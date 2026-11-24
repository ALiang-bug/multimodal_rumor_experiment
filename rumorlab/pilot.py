"""Run the independent label stages on a bounded MMFakeBench val batch."""

import json
from pathlib import Path
import re
import subprocess

from .grouping import apply_plan, freeze_plan
from .m1 import connect_role, load_lock
from .pipeline import JobFailed, execute
from .provenance import run_source_tasks
from .records import record_envelope, stage_record
from .requests import build_request
from .validation import ID_FIELDS, canonical_hash


def _inputs(state, run_id):
    row = state.db.execute(
        """SELECT imports.*, import_manifests.manifest FROM imports
           JOIN import_manifests USING(run_id) WHERE run_id = ?""", (run_id,),
    ).fetchone()
    if row is None:
        raise ValueError("Pilot requires an imported MMFakeBench val run")
    source = json.loads(row["manifest"])
    if (source["adapter"], source["source_split"], source["assigned_split"]) != (
        "mmfakebench", "MMFakeBench_val", "dev",
    ):
        raise ValueError("Pilot accepts only MMFakeBench val (dev)")
    records = json.loads(row["records"])
    samples = {item["sample_id"]: item for item in records if item["record_type"] == "sample"}
    if not samples or any(item["is_mock"] or item["split"] != "dev" for item in samples.values()):
        raise ValueError("Pilot requires real dev samples")
    evidence = {item["evidence_id"]: item for item in records if item["record_type"] == "evidence"}
    return row, source, records, samples, evidence


def _batch(state, run_id, batch_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", batch_id):
        raise ValueError("Invalid batch-id")
    return state.db.execute(
        "SELECT * FROM pilot_batches WHERE source_run_id = ? AND batch_id = ?", (run_id, batch_id),
    ).fetchone()


def _code_revision(contract):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=contract.root, capture_output=True, text=True, check=False,
    )
    revision = result.stdout.strip()
    if result.returncode or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Run the pilot from the checked-out Git repository")
    return revision


def _check_budget(contract, role, work_count):
    runtime = contract.experiment["runtime"]
    calls = work_count * (1 + runtime["technical_max_retries"] + runtime["format_repair_max_attempts"])
    tokens = calls * contract.models["models"][role]["generation"]["max_output_tokens"]
    if calls > runtime["max_calls_per_run"] or tokens > runtime["max_generated_tokens_per_run"]:
        raise ValueError("Pilot exceeds the worst-case call or output-token budget. Use a smaller --limit")


def _save_record(state, contract, record):
    kind = record["record_type"]
    contract.validator(kind).validate(record)
    with state.db:
        state.db.execute(
            "INSERT OR IGNORE INTO pilot_records VALUES (?, ?, ?)",
            (record["producer"]["run_id"], record[ID_FIELDS[kind]], json.dumps(record, ensure_ascii=False)),
        )


def _model_job(state, contract, client, manifest, stage, sample, claim, task, evidence, media_root, target=None):
    dependencies = [(sample["sample_id"], sample["content_hash"]),
                    (manifest["split_manifest_id"], manifest["split_manifest_id"].removeprefix("split_"))]
    available = [evidence[eid] for eid in sample["evidence_ids"]]
    dependencies.extend((item["evidence_id"], item["content_hash"]) for item in available)
    request = build_request(contract, stage, task, sample, claim, available, media_root, upstream=target)
    if target is not None:
        annotation_id = target["annotation_id"]
        dependencies.extend([(annotation_id, target["content_hash"]),
                             (annotation_id.removeprefix("annotation_"), target["content_hash"])])
    report = {"sample_id": sample["sample_id"], "claim_id": claim["claim_id"], "task_id": task}
    try:
        result, cached = execute(
            state, contract, request, sample["sample_id"], dependencies, client,
            script_hash=manifest["run_id"], claim=claim,
        )
    except JobFailed as exc:
        event = state.db.execute(
            "SELECT detail FROM events WHERE job_id = ? AND state = 'FAILED' ORDER BY sequence DESC LIMIT 1",
            (exc.job_id,),
        ).fetchone()
        error = json.loads(event["detail"]) if event else {}
        failure = error.get("error") or error
        return {**report, "job_id": exc.job_id, "status": "FAILED",
                "category": failure.get("category"),
                "technical_error": (failure.get("response") or {}).get("technical_error")}
    lock = manifest["models"][request.role]
    producer = {
        "kind": "model", "role": request.role, "model_id": lock["model_id"],
        "model_revision": lock["revision"], "run_id": manifest["run_id"],
        "prompt_version": manifest["prompt_versions"][stage], "prompt_hash": request.rendered_prompt_hash,
    }
    record = stage_record(
        result, sample, producer, [item["evidence_id"] for item in request.visible_inputs["evidence"]],
        kind="review" if target else "annotation", target=target,
    )
    if target is None:
        record["label_schema_version"] = manifest["label_schema_version"]
    _save_record(state, contract, record)
    return {**report, "job_id": result["job_id"], "cached": cached, "status": record["status"],
            "record_id": record[ID_FIELDS[record["record_type"]]],
            **({"decision": record["payload"]["decision"]} if target else {})}


def _finish(state, manifest, stage, results):
    report = {"stage": stage, "run_id": manifest["runtime"]["source_run_id"],
              "batch_id": manifest["runtime"]["batch_id"], "manifest_id": manifest["run_id"],
              "samples": manifest["runtime"]["sample_count"], "results": results,
              "failed": sum(item["status"] == "FAILED" for item in results)}
    with state.db:
        if stage == "annotate":
            state.db.execute("UPDATE pilot_batches SET annotate_report = ? WHERE manifest_id = ?",
                             (json.dumps(report), manifest["run_id"]))
        else:
            state.db.execute("UPDATE pilot_batches SET review_report = ? WHERE manifest_id = ?",
                             (json.dumps(report), manifest["run_id"]))
    return report


def annotate(state, contract, run_id, batch_id, limit, annotator_lock, reviewer_lock):
    existing = _batch(state, run_id, batch_id)
    row, source, records, samples, evidence = _inputs(state, run_id)
    if not 1 <= limit <= len(samples):
        raise ValueError(f"--limit must be between 1 and {len(samples)}")
    sample_ids = list(dict.fromkeys(item["sample_id"] for item in source["selection"]))
    if set(sample_ids) != set(samples):
        raise ValueError("Import manifest and sample IDs disagree")
    if limit < len(samples):
        sample_ids.sort(key=lambda sid: canonical_hash([contract.experiment["seed"], sid]))
    selected = [samples[sid] for sid in sample_ids[:limit]]
    tasks = contract.experiment["annotation"]["model_tasks"]
    work_count = sum(len(sample["target_claims"]) for sample in selected) * len(tasks)
    for role in ("annotator", "reviewer"):
        _check_budget(contract, role, work_count)
    locks = {
        "annotator": load_lock(contract, "annotator", annotator_lock),
        "reviewer": load_lock(contract, "reviewer", reviewer_lock),
    }
    revision = _code_revision(contract)
    plan_digest, plan = freeze_plan(state, run_id, records, contract.experiment["splitting"])
    apply_plan(records, plan)
    if any(sample["split"] != "dev" for sample in selected):
        raise ValueError("Pilot selection must remain entirely in dev")
    split_id = "split_" + plan_digest
    input_hash = canonical_hash([run_id, batch_id, row["digest"], sample_ids[:limit], plan_digest,
                                 revision, contract.config_hash, [digest for _, digest in locks.values()]])
    manifest_id = "pilot_" + input_hash
    if existing and existing["manifest_id"] != manifest_id:
        raise ValueError("Batch inputs changed. Use a new --batch-id")
    model_fields = ("model_id", "revision", "processor_revision", "backend_version",
                    "chat_template_hash", "precision", "quantization")
    model_settings = contract.models["models"]
    data = {
        "run_id": manifest_id, "mode": "real", "code_revision": revision,
        "config_hash": contract.config_hash, "label_schema_version": contract.labels["label_schema_version"],
        "dataset_version": "sha256:" + source["annotation_sha256"],
        "evidence_version": "sha256:" + canonical_hash(sorted((eid, ev["content_hash"]) for eid, ev in evidence.items())),
        "split_manifest_id": split_id,
        "models": {role: {**{key: lock[key] for key in model_fields}, "backend": model_settings[role]["backend"],
                          "thinking_enabled": model_settings[role]["generation"]["thinking"]}
                   for role, (lock, _) in locks.items()},
        "prompt_versions": {stage: contract.experiment["annotation"]["prompt_version"]
                            for stage in ("annotate", "review_labels")},
        "runtime": {"source_run_id": run_id, "batch_id": batch_id, "sample_count": limit, "workers": 1,
                    "decoding_schema": "structural-with-task-enums; full schema validated locally",
                    **{f"{role}_lock_sha256": digest for role, (_, digest) in locks.items()}},
        "seed": contract.experiment["seed"], "status": "RUNNING",
    }
    manifest = {**record_envelope("run_manifest", data, input_hash, is_mock=False), **data}
    contract.validator("run_manifest").validate(manifest)
    client = connect_role(contract, "annotator")
    with state.db:
        state.db.execute(
            """INSERT OR IGNORE INTO pilot_batches (source_run_id, batch_id, manifest_id, manifest)
               VALUES (?, ?, ?, ?)""", (run_id, batch_id, manifest_id, json.dumps(manifest)),
        )
        existing = _batch(state, run_id, batch_id)
        if existing["manifest_id"] != manifest_id:
            raise ValueError("Batch inputs changed. Use a new --batch-id")
        manifest = json.loads(existing["manifest"])
    output = []
    results = run_source_tasks(
        state, contract, selected, evidence.values(), (split_id, plan_digest), manifest, output,
    )
    for record in output:
        _save_record(state, contract, record)
    for sample in selected:
        for claim in sample["target_claims"]:
            for task in tasks:
                results.append(_model_job(state, contract, client, manifest, "annotate", sample, claim, task,
                                          evidence, Path(row["media_root"])))
    return _finish(state, manifest, "annotate", results)


def review_labels(state, contract, run_id, batch_id, reviewer_lock):
    batch = _batch(state, run_id, batch_id)
    if batch is None or batch["annotate_report"] is None:
        raise ValueError("Finish annotate for this --run-id and --batch-id before reviewing")
    manifest = json.loads(batch["manifest"])
    _, lock_hash = load_lock(contract, "reviewer", reviewer_lock)
    if lock_hash != manifest["runtime"]["reviewer_lock_sha256"]:
        raise ValueError("Reviewer model lock differs from the annotated batch")
    if manifest["config_hash"] != contract.config_hash or manifest["code_revision"] != _code_revision(contract):
        raise ValueError("Use the same code and configuration as the annotated batch")
    row, _, _, samples, evidence = _inputs(state, run_id)
    ids = [item["record_id"] for item in json.loads(batch["annotate_report"])["results"]
           if item.get("record_id") and item["status"] == "VALIDATED"]
    if not ids:
        raise ValueError("No validated model annotations available for review")
    _check_budget(contract, "reviewer", len(ids))
    client = connect_role(contract, "reviewer")
    results = []
    for record_id in ids:
        record = state.db.execute("SELECT record FROM pilot_records WHERE record_id = ?", (record_id,)).fetchone()
        target = json.loads(record["record"])
        claim, task = target["payload"]["target_claim"], target["payload"]["task_id"]
        results.append(_model_job(state, contract, client, manifest, "review_labels", samples[target["sample_id"]],
                                  claim, task, evidence, Path(row["media_root"]), target))
    return _finish(state, manifest, "review_labels", results)
