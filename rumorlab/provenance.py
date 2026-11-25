"""Resolve source labels from imported records without model inference."""

from .validation import canonical_hash
from .records import stage_record


FIELDS = {
    "text_reasoning": ("text_provenance", "OT", "FT"),
    "image_perception": ("image_provenance", "OI", "FI"),
}


def source_annotation(sample, claim, task, evidence):
    field, original, altered = FIELDS[task]
    if claim not in sample["target_claims"] or sample["status"] != "VALIDATED":
        raise ValueError("Source annotation requires a validated sample and registered claim")
    label = sample[field]
    ids = []
    if label == altered:
        operation = sample["operation"]
        if not sample["parent_sample_ids"] or not operation or operation["verification_status"] != "verified":
            raise ValueError("Controlled provenance requires verified manipulation and parent samples")
        ids = operation["verification_evidence_ids"]
        available = {row["evidence_id"] for row in evidence
                     if row["status"] == "AVAILABLE" and sample["sample_id"] in row["sample_ids"]}
        if not ids or not set(ids) <= available:
            raise ValueError("Controlled provenance requires available verification evidence")
    elif label != original:
        raise ValueError(f"Invalid source provenance label: {label}")
    return {
        "task_id": task, "target_claim": claim, "applicability": "applicable",
        "labels": [label], "unknown_reason": None, "evidence_ids": ids,
        "rationale": "Source label derived from the validated sample manifest.",
    }


def prepare_source_tasks(contract, samples, evidence, split_dependency, manifest):
    evidence_hashes = {row["evidence_id"]: row["content_hash"] for row in evidence}
    for sample in samples:
        for claim in sample["target_claims"]:
            for task in contract.experiment["annotation"]["provenance_tasks"]:
                payload = source_annotation(sample, claim, task, evidence)
                contract.validator("annotation_payload").validate(payload)
                deps = [(sample["sample_id"], sample["content_hash"]), split_dependency]
                deps.extend((eid, evidence_hashes[eid]) for eid in payload["evidence_ids"])
                identity = [contract.config_hash, "manifest", payload]
                if not sample["is_mock"]:
                    identity.append(manifest["run_id"])
                input_hash = canonical_hash(identity)
                yield sample, claim, task, payload, deps, input_hash


def save_source_tasks(state, contract, manifest, prepared, output):
    report = []
    for sample, claim, task, payload, deps, input_hash in prepared:
        job_id, lease, cached = state.claim(
            "annotate", sample["sample_id"], task, input_hash,
            deps, contract.experiment["runtime"]["lease_seconds"],
        )
        if cached is None:
            if lease is None:
                raise ValueError(f"Source annotation job is unavailable: {job_id}")
            state.finish(job_id, lease, result={
                "is_mock": sample["is_mock"], "producer": "manifest",
                "content_hash": canonical_hash(payload), "payload": payload,
            })
        record = stage_record(
            {"payload": payload, "input_hash": input_hash, "job_id": job_id,
             "created_at": manifest["created_at"]}, sample,
            {"kind": "manifest", "role": "system", "model_id": None, "model_revision": None,
             "run_id": manifest["run_id"], "prompt_version": None, "prompt_hash": None},
            payload["evidence_ids"], kind="annotation",
        )
        record["label_schema_version"] = contract.labels["label_schema_version"]
        output.append(record)
        report.append({
            "stage": "annotate", "job_id": job_id, "sample_id": sample["sample_id"],
            "claim_id": claim["claim_id"], "task_id": task, "labels": payload["labels"],
            "producer": "manifest", "status": "VALIDATED", "cached": cached is not None,
        })
    return report


def run_source_tasks(state, contract, samples, evidence, split_dependency, manifest, output):
    prepared = prepare_source_tasks(contract, samples, evidence, split_dependency, manifest)
    return save_source_tasks(state, contract, manifest, prepared, output)
