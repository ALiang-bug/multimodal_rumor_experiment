"""Export an imported snapshot with its current version status."""

from collections import defaultdict, deque
from contextlib import nullcontext
import json

from .validation import ID_FIELDS
from .grouping import apply_plan
from .quality import project_gold


def snapshot_records(state, run_id, *, gold_votes=3):
    owns_transaction = not state.db.in_transaction
    with state.db if owns_transaction else nullcontext():
        if owns_transaction:
            state.db.execute("BEGIN")
        snapshot = state.db.execute("SELECT records FROM imports WHERE run_id = ?", (run_id,)).fetchone()
        if snapshot is None:
            raise ValueError(f"No imported snapshot for {run_id!r}")
        retired = {tuple(row) for row in state.db.execute(
            "SELECT record_id, content_hash FROM invalidated_versions"
        )}
        split = state.db.execute("SELECT plan FROM split_plans WHERE run_id = ?", (run_id,)).fetchone()
        generated = state.db.execute("SELECT records FROM mock_exports WHERE run_id = ?", (run_id,)).fetchone()
        batches = state.db.execute(
            "SELECT * FROM pilot_batches WHERE source_run_id = ? ORDER BY batch_id", (run_id,),
        ).fetchall()
        pilot = []
        for batch in batches:
            manifest = json.loads(batch["manifest"])
            reports = [json.loads(batch[key]) for key in ("annotate_report", "review_report") if batch[key]]
            if any(report["failed"] for report in reports):
                manifest["status"] = "FAILED"
            elif len(reports) == 2:
                manifest["status"] = "COMPLETED"
            pilot.append(manifest)
            pilot.extend(json.loads(row["record"]) for row in state.db.execute(
                "SELECT record FROM pilot_records WHERE manifest_id = ? ORDER BY record_id", (batch["manifest_id"],),
            ))
        stale_jobs = {row["id"] for row in state.db.execute("SELECT id FROM jobs WHERE state = 'STALE'")}
        human = [json.loads(row["record"]) for row in state.db.execute(
            "SELECT record FROM human_submissions WHERE run_id = ? ORDER BY record_id", (run_id,),
        )]
    records = json.loads(snapshot["records"])
    if generated:
        records.extend(json.loads(generated["records"]))
    records.extend(pilot)
    records.extend(human)
    if split:
        apply_plan(records, json.loads(split["plan"]))
    by_id = {row[ID_FIELDS[row["record_type"]]]: row for row in records}
    dependents = defaultdict(set)
    for rid, row in by_id.items():
        kind = row["record_type"]
        parents = []
        if kind == "sample":
            parents = row["parent_sample_ids"]
        elif kind == "annotation":
            parents = [row["sample_id"], *row["available_evidence_ids"]]
        elif kind == "qa":
            solver = row["solver_input"]
            parents = [row["source_annotation_id"], row["sample_id"], *row["available_evidence_ids"],
                       *solver["reference_sample_ids"], *solver["evidence_ids"]]
        elif kind == "review":
            parents = [row["target_id"], *row["available_evidence_ids"]]
        for parent in parents:
            dependents[parent].add(rid)
    stale = {rid for rid, row in by_id.items() if (rid, row["content_hash"]) in retired}
    if generated or pilot:
        stale.update(rid for rid in by_id if rid.partition("_")[2] in stale_jobs)
    pending = deque(stale)
    while pending:
        for child in dependents[pending.popleft()] - stale:
            stale.add(child)
            pending.append(child)
    for rid in stale:
        row = by_id[rid]
        kind = row["record_type"]
        if kind in {"annotation", "qa", "review"}:
            row["status"] = "STALE"
        elif kind in {"sample", "evidence"}:
            row["status"] = "EXCLUDED"
        elif kind == "run_manifest":
            row["status"] = "FAILED"
    for review in pilot:
        if review["record_type"] != "review" or review["status"] != "COMPLETED":
            continue
        target = by_id[review["target_id"]]
        if target["status"] != "VALIDATED" or target["content_hash"] != review["target_content_hash"]:
            continue
        decision = review["payload"]["decision"]
        target["status"] = {"PASS": "ACCEPTED_SILVER", "NEEDS_HUMAN": "HUMAN_PENDING", "REJECT": "EXCLUDED"}[decision]
        target["quality_tier"] = "silver" if decision == "PASS" else "unresolved"
        target["review_ids"].append(review["review_id"])
    project_gold(records, gold_votes)
    return records
