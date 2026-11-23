"""M1 capability smoke against pinned, locally deployed vLLM snapshots."""

from dataclasses import asdict
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .requests import ModelRequest
from .validation import canonical_hash
from .vllm import VllmModel


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def record_lock(contract, role, snapshot, lock_path):
    settings = contract.models["models"][role]
    model_directory = "models--" + settings["model_id"].replace("/", "--")
    snapshot = snapshot.resolve(strict=True)
    if (not re.fullmatch(r"[0-9a-f]{40}", snapshot.name) or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name != model_directory):
        raise ValueError("Provide the exact cached Hugging Face snapshots/<40-character-commit> directory")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    template = getattr(processor, "chat_template", None) or getattr(processor.tokenizer, "chat_template", None)
    if not template:
        raise ValueError("The snapshot processor has no chat template")
    try:
        dependencies = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        hardware = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Cannot capture deployment metadata: {exc.cmd[0]} failed") from exc
    lock = {
        "model_id": settings["model_id"], "role": role, "snapshot": str(snapshot),
        "revision": snapshot.name, "processor_revision": snapshot.name,
        "backend_version": version("vllm"), "chat_template_hash": canonical_hash(template),
        "generation_config": settings["generation"], "precision": settings["precision"],
        "quantization": settings["quantization"], "hardware_fingerprint": hardware,
        "dependency_lock_hash": _sha(dependencies.encode("utf-8")),
        "vision_processor_config": {**settings["vision"], "max_images": 2,
                                    "image_processor": json.loads(processor.image_processor.to_json_string())},
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise ValueError("Existing lock differs. Use a separate run directory for changed settings")
    else:
        with lock_path.open("x", encoding="utf-8") as stream:
            json.dump(lock, stream, indent=2)
    return lock


def run_smoke(contract, role, lock_path):
    settings = contract.models["models"][role]
    lock_data = lock_path.read_bytes()
    lock = json.loads(lock_data)
    for field in contract.models["preflight"]["required_lock_fields"]:
        if field not in lock or (field != "quantization" and not lock[field]):
            raise ValueError(f"Missing model lock field: {field}")
    if lock.get("role") != role or any(lock[key] != settings[key] for key in ("model_id", "precision", "quantization")) \
            or lock["generation_config"] != settings["generation"]:
        raise ValueError("Model lock differs from role configuration")
    for key in ("revision", "processor_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", lock[key]):
            raise ValueError(f"Model lock requires an immutable {key}")
    endpoint = os.environ.get(settings["endpoint_env"])
    if not endpoint:
        raise ValueError(f"Set {settings['endpoint_env']} to the deployed model's /v1 endpoint")
    client = VllmModel(settings, endpoint, os.environ.get(settings["token_env"]),
                       contract.models["response_contract"]["timeout_seconds"])
    client.verify_identity()
    cases = [
        ("text", "Return the color written here: green.", (), ["green"]),
        ("single_red", "Return the dominant color of each attached image in attachment order.", ("red",), ["red"]),
        ("single_blue", "Return the dominant color of each attached image in attachment order.", ("blue",), ["blue"]),
        ("two_red_blue", "Return the dominant color of each attached image in attachment order.", ("red", "blue"), ["red", "blue"]),
        ("two_blue_red", "Return the dominant color of each attached image in attachment order.", ("blue", "red"), ["blue", "red"]),
    ]
    from PIL import Image

    pngs = {}
    for color in ("red", "blue"):
        buffer = BytesIO()
        Image.new("RGB", (128, 96), color=color).save(buffer, format="PNG")
        pngs[color] = buffer.getvalue()

    results = {}
    for name, prompt, colors, expected in cases:
        images = tuple(pngs[color] for color in colors)
        assets = [{"mime_type": "image/png", "sha256": _sha(raw)} for raw in images]
        schema = {
            "type": "object", "additionalProperties": False, "required": ["colors"],
            "properties": {"colors": {"type": "array", "items": {"type": "string", "enum": ["red", "blue", "green"]},
                                       "minItems": len(expected), "maxItems": len(expected)}},
        }
        messages = (("user", prompt + "\nReturn one JSON object conforming to: " + json.dumps(schema)),)
        probe_id = canonical_hash([role, name, messages, assets, _sha(lock_data)])
        request = ModelRequest(role, "preflight", "m1_color_probe", {"images": assets}, canonical_hash(messages),
                               schema, canonical_hash(settings), probe_id, messages, images)
        result, reasoning_seen = client.complete(request)
        results[name] = {**asdict(result), "reasoning_separated": reasoning_seen,
                         "image_hashes": [item["sha256"] for item in assets], "request_id": probe_id,
                         "passed": result.schema_valid and result.parsed_payload == {"colors": expected}}
    return {"role": role, "model_id": settings["model_id"], "model_lock_sha256": _sha(lock_data),
            "checks": results, "passed": all(case["passed"] for case in results.values()) and
            (not settings["generation"]["thinking"] or any(case["reasoning_separated"] for case in results.values()))}
