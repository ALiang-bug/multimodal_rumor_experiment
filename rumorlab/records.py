"""Build contract records from persisted mock responses."""

from datetime import datetime, timezone

from .validation import ID_FIELDS, canonical_hash


def mock_envelope(kind, payload, input_hash):
    return {
        "schema_version": "1.0.0", "record_type": kind, "is_mock": True,
        "mock_note": "Scripted offline workflow output.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "record_version": 1, "supersedes_id": None, "input_hash": input_hash,
        "content_hash": canonical_hash(payload),
    }


def mock_manifest(contract, run_id, split_id):
    models = {}
    for role, settings in contract.models["models"].items():
        models[role] = {
            "model_id": settings["model_id"], "backend": "mock",
            "revision": None, "backend_version": None, "processor_revision": None,
            "chat_template_hash": None, "precision": None, "quantization": None,
            "thinking_enabled": settings["generation"]["thinking"],
        }
    data = {
        "run_id": "mock_" + canonical_hash([run_id, split_id, contract.config_hash]),
        "mode": "mock", "code_revision": None, "config_hash": contract.config_hash,
        "label_schema_version": contract.labels["label_schema_version"],
        "dataset_version": contract.experiment["source"]["dataset_version"] or "synthetic",
        "evidence_version": "synthetic", "split_manifest_id": split_id,
        "models": models, "prompt_versions": {}, "runtime": {},
        "seed": contract.experiment["seed"], "status": "COMPLETED",
    }
    return {**mock_envelope("run_manifest", data, contract.config_hash), **data}


def stage_record(result, sample, producer, evidence_ids, *, kind, target=None, solver=None):
    payload = result["payload"]
    record = {
        **mock_envelope(kind, payload, result["input_hash"]),
        ID_FIELDS[kind]: f"{kind}_" + result["job_id"],
        "payload": payload, "producer": producer, "available_evidence_ids": evidence_ids,
    }
    if "created_at" in result:
        record["created_at"] = result["created_at"]
    if kind == "review":
        target_kind = target["record_type"]
        return {**record, "target_record_type": target_kind, "target_id": target[ID_FIELDS[target_kind]],
                "target_content_hash": target["content_hash"], "review_mode": "candidate_aware",
                "stage": "review_labels" if target_kind == "annotation" else "review_qa",
                "status": "COMPLETED", "blind_prediction": None}
    record.update(sample_id=sample["sample_id"], status="VALIDATED", quality_tier="unresolved",
                  review_ids=[], human_vote_ids=[], adjudication_id=None)
    if kind == "qa":
        record.update(source_annotation_id=target["annotation_id"],
                      source_annotation_content_hash=target["content_hash"], solver_input=solver)
    elif payload["applicability"] == "not_applicable":
        record["status"] = "SKIPPED"
    return record
