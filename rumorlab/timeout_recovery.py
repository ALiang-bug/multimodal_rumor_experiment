"""Recover terminal timeouts from a completed real val attempt."""

from dataclasses import replace
import json
from pathlib import Path
from time import monotonic

from . import pilot
from .grouping import apply_plan
from .m1 import connect_role, load_lock
from .pipeline import model_job_spec
from .requests import build_request, visible_material


TIMEOUT = 360


def _key(row):
    return row["sample_id"], row["claim_id"], row["task_id"]


def _timed_out(row):
    return (row["status"], row.get("category"), row.get("technical_error")) == (
        "FAILED", "technical", "TimeoutError",
    )


def _reports(path, run_id, stage):
    report = json.loads(path.read_text(encoding="utf-8"))
    batches = report["batches"]
    if (report["stage"] != stage or not batches
            or any(batch["run_id"] != run_id or batch["stage"] != stage for batch in batches)
            or len({batch["batch_id"] for batch in batches}) != len(batches)):
        raise ValueError(f"Expected a completed {stage} report for {run_id}: {path}")
    return {batch["batch_id"]: batch for batch in batches}


def recover_timeouts(state, contract, run_id, stage, attempt, annotator_lock, reviewer_lock, workers):
    annotation = _reports(attempt / "annotate.json", run_id, "annotate")
    baseline = annotation if stage == "annotate" else _reports(
        attempt / "review-labels.json", run_id, "review_labels",
    )
    if baseline.keys() != annotation.keys():
        raise ValueError("Original annotate and review-labels batch IDs differ")
    return _recover_batches(state, contract, run_id, stage, baseline, annotator_lock, reviewer_lock, workers,
                            annotation=annotation)


def recover_stage_report(state, contract, run_id, report, annotator_lock, reviewer_lock, workers):
    """Retry the just-finished phase against its still-running model endpoint."""
    batches = report.get("batches", [report])
    if not any(_timed_out(row) for batch in batches for row in batch["results"]):
        return report

    baseline = {batch["batch_id"]: batch for batch in batches}
    recovery = _recover_batches(
        state, contract, run_id, report["stage"], baseline, annotator_lock, reviewer_lock, workers,
    )
    stats = {key: value for key, value in recovery.items() if key not in {"stage", "batches"}}
    stats["first_pass_failed"] = report["failed"]
    updated = {batch["batch_id"]: batch for batch in recovery["batches"]}
    merged = []
    for batch_id, batch in baseline.items():
        final = updated.get(batch_id, batch)
        merged.append({**batch, "results": final["results"], "failed": final["failed"]})
    if "batches" not in report:
        return {**merged[0], "timeout_recovery": stats}
    return {**report, "batches": merged, "failed": sum(batch["failed"] for batch in merged),
            "timeout_recovery": stats}


def _recover_batches(state, contract, run_id, stage, baseline, annotator_lock, reviewer_lock, workers,
                     *, annotation=None):
    started = monotonic()
    role = "annotator" if stage == "annotate" else "reviewer"
    source, _, records, samples, evidence = pilot._inputs(state.db, run_id)
    plan = state.db.execute("SELECT * FROM split_plans WHERE run_id = ?", (run_id,)).fetchone()
    if plan is None:
        raise ValueError("Timeout recovery requires an existing frozen split")
    apply_plan(records, json.loads(plan["plan"]))
    locks = {name: load_lock(contract, name, path)[1] for name, path in (
        ("annotator", annotator_lock), ("reviewer", reviewer_lock),
    ) if path is not None}
    revision = pilot._code_revision(contract)
    report_column = "annotate_report" if stage == "annotate" else "review_report"
    selected, existing, manifests = [], {}, {}
    for batch_id, old in baseline.items():
        batch = pilot._batch(state, run_id, batch_id)
        if batch is None or not batch[report_column]:
            raise ValueError(f"Missing completed batch: {batch_id}")
        manifest = json.loads(batch["manifest"])
        first_annotation = annotation[batch_id] if annotation is not None else None
        if (manifest["run_id"] != old["manifest_id"]
                or (first_annotation is not None and manifest["run_id"] != first_annotation["manifest_id"])
                or manifest["split_manifest_id"] != "split_" + plan["digest"]
                or manifest["config_hash"] != contract.config_hash
                or any(manifest["runtime"][f"{name}_lock_sha256"] != digest for name, digest in locks.items())
                or not pilot._can_resume_revision(contract, manifest["code_revision"], revision)):
            raise ValueError(f"Use the original configuration, model locks and compatible code: {batch_id}")
        manifests[batch_id] = manifest
        existing[batch_id] = {_key(row): row for row in json.loads(batch[report_column])["results"]}
        old_timeouts = {_key(row): row["job_id"] for row in old["results"] if _timed_out(row)}
        if stage == "annotate":
            targets = [(key, job_id, None) for key, job_id in old_timeouts.items()]
        else:
            annotation_timeouts = {_key(row) for row in (first_annotation or {}).get("results", ()) if _timed_out(row)}
            original_review_keys = {_key(row) for row in old["results"]}
            targets = [
                (_key(row), old_timeouts.get(_key(row)), row["record_id"])
                for row in json.loads(batch["annotate_report"])["results"]
                if row["status"] == "VALIDATED" and row.get("record_id")
                and (_key(row) in old_timeouts
                     or (_key(row) in annotation_timeouts and _key(row) not in original_review_keys))
            ]
            if old_timeouts.keys() - {key for key, _, _ in targets}:
                raise ValueError(f"Missing original review candidates: {batch_id}")
        pilot._check_budget(contract, role, len(targets))
        selected.extend((batch_id, *target) for target in targets)

    client = connect_role(contract, role, timeout=TIMEOUT) if selected else None
    media_root = Path(source["media_root"])
    jobs, specs, material_cache = [], [], {}
    for batch_id, (sample_id, claim_id, task), previous_id, record_id in selected:
        sample = samples[sample_id]
        if sample["split"] != "dev" or task not in contract.experiment["annotation"]["model_tasks"]:
            raise ValueError("Timeout recovery requires a val model task")
        claim = next((claim for claim in sample["target_claims"] if claim["claim_id"] == claim_id), None)
        if claim is None:
            raise ValueError(f"Missing frozen claim: {claim_id}")
        manifest, target = manifests[batch_id], None
        if record_id is not None:
            record = state.db.execute(
                "SELECT record FROM pilot_records WHERE manifest_id = ? AND record_id = ?",
                (manifest["run_id"], record_id),
            ).fetchone()
            if record is None:
                raise ValueError(f"Missing review target: {record_id}")
            target = json.loads(record["record"])
        available = [evidence[eid] for eid in sample["evidence_ids"]]
        if sample_id not in material_cache:
            material_cache[sample_id] = visible_material(sample, available, media_root, solver=False)
        request = build_request(contract, stage, task, sample, claim, available, media_root,
                                upstream=target, material=material_cache[sample_id])
        dependencies = pilot._model_dependencies(manifest, sample, available, target)
        specs.append((model_job_spec(request, sample_id, dependencies, manifest["run_id"]), previous_id))
        jobs.append((batch_id, sample, claim, task, target, replace(request, images=()), client.prepare(request)))
    material_cache.clear()
    state.register_timeout_recovery(specs, TIMEOUT)

    def run(worker_state, job):
        batch_id, sample, claim, task, target, request, body = job
        return pilot._model_job(worker_state, contract, client, manifests[batch_id], stage, sample, claim, task,
                                evidence, media_root, target, (), request=request, prepared=body)

    results = pilot._dispatch_jobs(state, jobs, workers, run)
    if any(item.get("technical_error") in {"lease_busy", "lease_lost"} for item in results):
        raise ValueError("Timeout recovery encountered an active or lost lease. Resume after the other run exits.")
    updated = set()
    for (batch_id, *_), result in zip(jobs, results):
        existing[batch_id][_key(result)] = result
        updated.add(batch_id)
    reports = [pilot._finish(state, manifests[batch_id], stage, list(existing[batch_id].values()),
                             workers, started, execution_revision=revision,
                             timings=baseline[batch_id].get("run_timings"))
               for batch_id in manifests if batch_id in updated]
    failed = sum(row["status"] == "FAILED" for row in results)
    return {
        "stage": stage, "timeout_seconds": TIMEOUT, "workers": workers,
        "elapsed_seconds": round(monotonic() - started, 3),
        "selected": len(selected), "original_timeouts": sum(previous_id is not None for _, _, previous_id, _ in selected),
        "new_candidates": sum(previous_id is None for _, _, previous_id, _ in selected),
        "cached": sum(result.get("cached", False) for result in results), "failed": failed,
        "batches": reports,
    }
