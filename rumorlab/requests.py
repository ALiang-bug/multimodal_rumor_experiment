"""Stage-specific model requests built from explicit field allowlists."""

from dataclasses import dataclass
import hashlib
import json

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .validation import canonical_hash


@dataclass(frozen=True)
class ModelRequest:
    role: str
    stage: str
    task_id: str
    visible_inputs: dict
    rendered_prompt_hash: str
    response_schema: dict
    model_config_hash: str
    request_id: str
    messages: tuple
    images: tuple


def visible_material(sample, evidence, media_root, *, solver):
    evidence_by_id = {item["evidence_id"]: item for item in evidence}
    visible = []
    for evidence_id in sample["evidence_ids"]:
        item = evidence_by_id[evidence_id]
        if item["status"] != "AVAILABLE" or item["evidence_type"] == "construction_log":
            continue
        if solver and (item["visibility"] == "construction" or item["direct_answer_present"]):
            continue
        if not solver and item["visibility"] == "solver":
            continue
        visible.append({key: item[key] for key in ("evidence_id", "evidence_type", "summary", "snapshot_sha256")})
    root = media_root.resolve()
    assets, images = [], []
    for index, asset in enumerate(sample["media"], 1):
        path = (root / asset["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Media path escapes the configured root")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if asset["hash_status"] != "verified" or digest != asset["sha256"]:
            raise ValueError(f"Media bytes changed: image {index}")
        assets.append({"id": f"image_{index}", "mime_type": asset["mime_type"], "sha256": digest})
        images.append(raw)
    return {"text": sample["text"], "images": assets, "evidence": visible}, tuple(images)


DEFINITIONS = {
    "annotate": ("annotator", "annotation_payload"),
    "review_labels": ("reviewer", "review_payload"),
    "generate_qa": ("annotator", "qa_payload"),
    "review_qa": ("reviewer", "review_payload"),
}


def _response_schema(contract, definition, task_id):
    definitions = contract.schema["$defs"]

    def expand(node):
        if isinstance(node, list):
            return [expand(child) for child in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return expand(definitions[node["$ref"].removeprefix("#/$defs/")])
        result = {}
        for key, value in node.items():
            if key == "description":
                continue  # The source contract includes Chinese editorial notes.
            if key == "allOf":
                selected = []
                for clause in value:
                    target = clause.get("if", {}).get("properties", {}).get("task_id", {}).get("const")
                    if target is None or target == task_id:
                        selected.append(clause)
                value = selected
            result[key] = expand(value)
        if "task_id" in result.get("properties", {}):
            result["properties"]["task_id"] = {"const": task_id}
        return result

    return expand(definitions[definition])


def build_request(contract, stage, task_id, sample, claim, evidence, media_root, *, upstream=None):
    role, definition = DEFINITIONS[stage]
    if claim not in sample["target_claims"]:
        raise ValueError("Claim is outside the sample")
    solver = stage in {"generate_qa", "review_qa"}
    material, images = visible_material(sample, evidence, media_root, solver=solver)
    schema = _response_schema(contract, definition, task_id)
    if stage in {"annotate", "review_labels", "generate_qa", "review_qa"}:
        evidence_ids = [item["evidence_id"] for item in material["evidence"]]
        schema["properties"]["evidence_ids"] = (
            {"type": "array", "items": {"enum": evidence_ids}, "uniqueItems": True}
            if evidence_ids else {"const": []}
        )
    if stage == "annotate":
        schema["properties"]["target_claim"] = {"const": claim}
    elif stage == "review_labels":
        properties = schema["properties"]
        suggested = properties["suggested_annotation"]["anyOf"][0]["properties"]
        suggested["evidence_ids"] = properties["evidence_ids"]
        suggested["target_claim"] = {"const": claim}
    elif stage == "review_qa":
        schema["properties"]["suggested_annotation"] = {"type": "null"}
    environment = Environment(
        loader=FileSystemLoader(contract.root / "prompts"),
        undefined=StrictUndefined, autoescape=False,
    )
    task = contract.labels["tasks"][task_id]
    values = {
        "task_id": task_id,
        "task_instructions": environment.loader.get_source(environment, f"tasks/{task_id}.txt")[0],
        "task_definition_json": {
            "cardinality": task["cardinality"], "origin": task["origin"],
            "labels": {item["code"]: item["name_en"] for item in task["labels"]},
        },
        "visible_input_json": material,
        "target_claim_json": claim,
        "response_schema_json": schema,
    }
    if stage != "annotate":
        expected = "qa_payload" if stage == "review_qa" else "annotation_payload"
        payload = upstream["payload"]
        contract.validator(expected).validate(payload)
        if payload["task_id"] != task_id or upstream["sample_id"] != sample["sample_id"]:
            raise ValueError("Upstream sample or task does not match the request")
        if expected == "annotation_payload" and payload["target_claim"] != claim:
            raise ValueError("Upstream claim does not match the request")
        if stage == "review_labels":
            values["candidate_annotation_json"] = payload
        elif stage == "generate_qa":
            if upstream["status"] not in {"ACCEPTED_SILVER", "ACCEPTED_GOLD"}:
                raise ValueError("QA generation requires an accepted annotation")
            schema["properties"]["target_labels"] = {"const": payload["labels"]}
            schema = {"anyOf": [schema, {"type": "object", "properties": {
                "skip_reason": {"enum": ["insufficient_solver_evidence", "no_supported_question"]},
            }, "required": ["skip_reason"], "additionalProperties": False}]}
            values["response_schema_json"] = schema
            values.update(accepted_annotation_json=payload, qa_policy_json=contract.experiment["qa"])
        else:
            values.update(candidate_qa_json=payload, solver_input_json={
                **material, "question": payload["question"],
                "options": [{"option_id": option["option_id"], "text": option["text"]}
                            for option in payload["options"]],
            })
    rendered = {key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if key.endswith("_json") else value for key, value in values.items()}
    messages = (
        ("system", environment.get_template("system.txt").render(role=role)),
        ("user", environment.get_template(f"{stage}.j2").render(**rendered)),
    )
    prompt_hash = canonical_hash(messages)
    model_hash = canonical_hash(contract.models["models"][role])
    request_id = canonical_hash([stage, task_id, material, schema, prompt_hash, model_hash, contract.config_hash])
    return ModelRequest(role, stage, task_id, material, prompt_hash, schema, model_hash, request_id, messages, images)
