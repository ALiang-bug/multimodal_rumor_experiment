"""Restore a completed val baseline and recover verifiable cached model results."""

import argparse
from contextlib import ExitStack, closing
import json
from pathlib import Path
import sqlite3

from jsonschema import Draft202012Validator

from . import pilot
from .config import Contract
from .grouping import apply_plan
from .pipeline import model_job_spec
from .requests import build_request, visible_material
from .state import State
from .timeout_recovery import _key, _reports, _timed_out
from .validation import ID_FIELDS, canonical_hash


def _reader(path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return closing(db)


def _read_one(db, sql, identifier):
    if db is not None:
        try:
            row = db.execute(sql, (identifier,)).fetchone()
            return json.loads(row[0]) if row else None
        except (sqlite3.Error, ValueError, TypeError):
            return None
    return None


def _cached_result(record, job_id):
    return {
        "job_id": job_id, "is_mock": record["is_mock"], "input_hash": record["input_hash"],
        "content_hash": record["content_hash"], "payload": record["payload"],
        "created_at": record["created_at"], "restored_from": "snapshot_record",
    }


def _salvaged_result(db, job_id, request, input_hash, claim):
    kind = "annotation" if request.stage == "annotate" else "review"
    result = _read_one(db, "SELECT result FROM jobs WHERE id = ? AND state = 'SUCCEEDED'", job_id)
    if result is None:
        record_id = f"{kind}_{job_id}"
        record = _read_one(db, "SELECT record FROM pilot_records WHERE record_id = ?", record_id)
        if isinstance(record, dict) and record.get(ID_FIELDS[kind]) == record_id:
            try:
                result = _cached_result(record, job_id)
            except KeyError:
                return None
    if not isinstance(result, dict):
        return None
    payload = result.get("payload")
    if (result.get("job_id") != job_id or result.get("input_hash") != input_hash
            or result.get("is_mock") is not False or not isinstance(result.get("created_at"), str)
            or result.get("content_hash") != canonical_hash(payload)
            or not Draft202012Validator(request.response_schema).is_valid(payload)):
        return None
    annotation = payload if kind == "annotation" else payload.get("suggested_annotation")
    visible = {row["evidence_id"] for row in request.visible_inputs["evidence"]}
    if not set(payload["evidence_ids"]) <= visible:
        return None
    if annotation and (annotation["target_claim"] != claim or not set(annotation["evidence_ids"]) <= visible):
        return None
    return result


def _store_job(state, spec, job_id, result, reason):
    registered, created = state._register_job(*spec)
    if registered != job_id or not created:
        raise ValueError(f"Restored request does not match its original identity: {job_id}")
    status = "SUCCEEDED" if result is not None else "FAILED"
    state.db.execute(
        "UPDATE jobs SET state = ?, result = ? WHERE id = ?",
        (status, json.dumps(result, ensure_ascii=False) if result is not None else None, job_id),
    )
    detail = {"restored_from": reason}
    if result is None:
        detail["error"] = {"category": "technical", "response": {"technical_error": "TimeoutError"}}
    state._event(job_id, status, detail)


def restore_val(contract, source, baseline, output, run_id, salvage=()):
    annotations = _reports(baseline / "annotate.json", run_id, "annotate")
    reviews = _reports(baseline / "review-labels.json", run_id, "review_labels")
    if annotations.keys() != reviews.keys():
        raise ValueError("Baseline annotate and review batches differ")
    manifests = {batch["manifest_id"] for batch in annotations.values()}
    snapshot = {}
    with (baseline / "snapshot.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            kind = row["record_type"]
            if kind in {"annotation", "review", "run_manifest"}:
                identifier = row[ID_FIELDS[kind]]
                if identifier in manifests or row.get("producer", {}).get("run_id") in manifests:
                    if identifier in snapshot:
                        raise ValueError(f"Duplicate snapshot record: {identifier}")
                    snapshot[identifier] = row

    if output.exists() or not source.is_file():
        raise ValueError("Restore requires an existing import database and a new output file")
    with ExitStack() as stack:
        original = stack.enter_context(_reader(source))
        original.execute("BEGIN")
        source_row, import_manifest, records, samples, evidence = pilot._inputs(original, run_id)
        inputs = {}
        for table in ("imports", "import_manifests", "split_plans"):
            row = original.execute(f"SELECT * FROM {table} WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise ValueError(f"Frozen val data is missing: {table}")
            inputs[table] = tuple(row)
        _, split_digest, plan = inputs["split_plans"]
        apply_plan(records, json.loads(plan))
        media_root = Path(source_row["media_root"])
        revision = pilot._code_revision(contract)
        salvage_readers = [stack.enter_context(_reader(path)) for path in salvage]
        with output.open("xb"):
            pass
        state = stack.enter_context(closing(State(output)))
        restored, reused, remaining = 0, 0, 0
        with state.db:
            for table, values in inputs.items():
                marks = ", ".join("?" for _ in values)
                state.db.execute(f"INSERT INTO {table} VALUES ({marks})", values)
            for batch_id, a_report in annotations.items():
                r_report = reviews[batch_id]
                mid = a_report["manifest_id"]
                manifest = snapshot[mid]
                if (mid != r_report["manifest_id"] or manifest["runtime"]["batch_id"] != batch_id
                        or manifest["runtime"]["source_run_id"] != run_id
                        or manifest["config_hash"] != contract.config_hash
                        or manifest["dataset_version"] != "sha256:" + import_manifest["annotation_sha256"]
                        or manifest["split_manifest_id"] != "split_" + split_digest
                        or not pilot._can_resume_revision(contract, manifest["code_revision"], revision)):
                    raise ValueError(f"Baseline manifest changed: {batch_id}")
                state.db.execute(
                    "INSERT INTO pilot_batches VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, batch_id, mid, json.dumps(manifest), json.dumps(a_report), json.dumps(r_report)),
                )
                candidates = {}
                material_cache = {}
                for stage, report in (("annotate", a_report), ("review_labels", r_report)):
                    for item in report["results"]:
                        key = _key(item)
                        sid, claim_id, task = key
                        sample = samples[sid]
                        claim = next(c for c in sample["target_claims"] if c["claim_id"] == claim_id)
                        if sample["split"] != "dev":
                            raise ValueError(f"Expected a val sample: {sid}")
                        available = [evidence[eid] for eid in sample["evidence_ids"]]
                        target = candidates[key] if stage == "review_labels" else None
                        deps = pilot._model_dependencies(manifest, sample, available, target)
                        jid = item["job_id"]
                        if _timed_out(item):
                            if sid not in material_cache:
                                material_cache[sid] = visible_material(sample, available, media_root, solver=False)
                            request = build_request(contract, stage, task, sample, claim, available, media_root,
                                                    upstream=target, material=material_cache[sid])
                            spec = model_job_spec(request, sid, deps, mid)
                            cached = None
                            for reader in salvage_readers:
                                cached = _salvaged_result(reader, jid, request, spec[3], claim)
                                if cached is not None:
                                    break
                            _store_job(state, spec, jid, cached, "salvaged_result" if cached else "baseline_timeout_report")
                            reused += cached is not None
                            remaining += cached is None
                            continue
                        if item["status"] not in {"VALIDATED", "SKIPPED", "COMPLETED"}:
                            raise ValueError(f"Unexpected baseline outcome: {jid}")
                        kind = "annotation" if stage == "annotate" else "review"
                        rid = f"{kind}_{jid}"
                        record = snapshot[rid]
                        if (item.get("record_id", rid) != rid or record["producer"]["run_id"] != mid
                                or canonical_hash(record["payload"]) != record["content_hash"]):
                            raise ValueError(f"Baseline record changed: {rid}")
                        if kind == "annotation":
                            record = {**record, "status": item["status"], "quality_tier": "unresolved",
                                      "review_ids": [], "human_vote_ids": [], "adjudication_id": None}
                            if item.get("producer") == "manifest":
                                deps = [(sid, sample["content_hash"]), ("split_" + split_digest, split_digest)]
                                deps.extend((eid, evidence[eid]["content_hash"]) for eid in record["payload"]["evidence_ids"])
                            elif item["status"] == "VALIDATED":
                                candidates[key] = record
                        contract.validator(kind).validate(record)
                        state.db.execute("INSERT INTO pilot_records VALUES (?, ?, ?)", (mid, rid, json.dumps(record)))
                        _store_job(state, (stage, sid, task, record["input_hash"], deps), jid,
                                   _cached_result(record, jid), "baseline_snapshot")
                        restored += 1
        checks = state.db.execute("PRAGMA integrity_check").fetchall()
        if [row[0] for row in checks] != ["ok"] or state.db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Reconstructed val database failed integrity checks")
    return {"database": str(output), "baseline_records": restored, "reused_recovery_jobs": reused,
            "remaining_original_timeouts": remaining, "integrity_check": "ok"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source-db", "baseline", "output-db"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--salvage-db", type=Path, action="append", default=[])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    report = restore_val(Contract(args.config), args.source_db, args.baseline, args.output_db, args.run_id,
                         salvage=args.salvage_db)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
