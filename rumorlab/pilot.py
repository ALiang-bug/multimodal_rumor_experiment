"""Run independent model stages on bounded MMFakeBench val batches."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import re
import subprocess
from time import monotonic

from .grouping import apply_plan, freeze_plan
from .m1 import connect_role, load_lock
from .pipeline import JobFailed, execute
from .provenance import run_source_tasks
from .qa import solver_projection
from .records import record_envelope, stage_record
from .requests import build_request
from .snapshot import snapshot_records
from .state import State
from .validation import ID_FIELDS, canonical_hash


def _inputs(db, run_id):
    row = db.execute(
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


def _can_resume_revision(contract, pinned, current):
    if pinned == current:
        return True
    # A direct patch can opt in to preserving batch IDs if it changes only execution code.
    def git(*args):
        return subprocess.run(["git", *args], cwd=contract.root, capture_output=True, text=True)

    metadata = git("log", "-1", "--format=%P%n%(trailers:key=Runtime-Resume-From,valueonly)", current)
    if metadata.returncode or metadata.stdout.split() != [pinned, pinned]:
        return False
    changed = git("diff", "--name-only", pinned, current)
    allowed = {"rumorlab/state.py", "rumorlab/pipeline.py", "rumorlab/pilot.py"}
    if changed.returncode or any(path not in allowed and not path.startswith(("docs/", "tests/"))
                                 for path in changed.stdout.splitlines()):
        return False
    return git("diff", "--quiet", "HEAD", "--", "rumorlab", "configs", "prompts", "schemas").returncode == 0


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


def _model_job(state, contract, client, manifest, stage, sample, claim, task, evidence, media_root, target, approvals):
    dependencies = [(sample["sample_id"], sample["content_hash"]),
                    (manifest["split_manifest_id"], manifest["split_manifest_id"].removeprefix("split_"))]
    available = [evidence[eid] for eid in sample["evidence_ids"]]
    dependencies.extend((item["evidence_id"], item["content_hash"]) for item in available)
    request = build_request(contract, stage, task, sample, claim, available, media_root, upstream=target)
    if target is not None:
        dependencies.extend(_record_dependencies(target))
    for approval in approvals:
        dependencies.extend(_record_dependencies(approval))
    if stage == "review_qa" and solver_projection(request, sample) != target["solver_input"]:
        raise ValueError("QA solver input changed since generation")
    report = {"sample_id": sample["sample_id"], "claim_id": claim["claim_id"], "task_id": task}
    try:
        result, cached = execute(
            state, contract, request, sample["sample_id"], dependencies, client,
            script_hash=manifest["run_id"], claim=claim,
            labels=target["payload"]["labels"] if stage == "generate_qa" else None,
        )
    except JobFailed as exc:
        event = state.db.execute(
            "SELECT detail FROM events WHERE job_id = ? AND state = 'FAILED' ORDER BY sequence DESC LIMIT 1",
            (exc.job_id,),
        ).fetchone()
        error = json.loads(event["detail"]) if event else {}
        failure = error.get("error") or error
        return {**report, "job_id": exc.job_id, "status": "FAILED",
                "category": "technical" if exc.technical_error else failure.get("category"),
                "technical_error": exc.technical_error or (failure.get("response") or {}).get("technical_error")}
    lock = manifest["models"][request.role]
    if stage == "generate_qa" and "skip_reason" in result["payload"]:
        return {**report, "job_id": result["job_id"], "cached": cached, "status": "SKIPPED",
                "reason": result["payload"]["skip_reason"]}
    producer = {
        "kind": "model", "role": request.role, "model_id": lock["model_id"],
        "model_revision": lock["revision"], "run_id": manifest["run_id"],
        "prompt_version": manifest["prompt_versions"][stage], "prompt_hash": request.rendered_prompt_hash,
    }
    kind = {"annotate": "annotation", "generate_qa": "qa"}.get(stage, "review")
    record = stage_record(
        result, sample, producer, [item["evidence_id"] for item in request.visible_inputs["evidence"]],
        kind=kind, target=target, solver=solver_projection(request, sample) if kind == "qa" else None,
    )
    if kind != "review":
        record["label_schema_version"] = manifest["label_schema_version"]
    _save_record(state, contract, record)
    return {**report, "job_id": result["job_id"], "cached": cached, "status": record["status"],
            "record_id": record[ID_FIELDS[record["record_type"]]],
            **({"decision": record["payload"]["decision"]} if kind == "review" else {})}


def _record_dependencies(record):
    kind = record["record_type"]
    record_id = record[ID_FIELDS[kind]]
    yield record_id, record["content_hash"]
    if kind in {"annotation", "qa", "review"}:
        yield record_id.removeprefix(kind + "_"), record["content_hash"]


def _run_model_jobs(state, contract, client, manifest, stage, evidence, media_root, jobs, workers):
    if workers == 1:
        return [_model_job(state, contract, client, manifest, stage, sample, claim, task, evidence,
                           media_root, target, approvals) for sample, claim, task, target, approvals in jobs]

    def run(job):
        sample, claim, task, target, approvals = job
        # SQLite connections are thread-local, while requests wait outside write transactions.
        with closing(State(state.path)) as worker_state:
            return _model_job(worker_state, contract, client, manifest, stage, sample, claim, task,
                              evidence, media_root, target, approvals)

    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        return list(executor.map(run, jobs))
    finally:
        executor.shutdown(cancel_futures=True)


def _finish(state, manifest, stage, results, workers, started, *, execution_revision=None):
    report = {"stage": stage, "run_id": manifest["runtime"]["source_run_id"],
              "batch_id": manifest["runtime"]["batch_id"], "manifest_id": manifest["run_id"],
              "samples": manifest["runtime"]["sample_count"], "results": results,
              "workers": workers, "elapsed_seconds": round(monotonic() - started, 3),
              "failed": sum(item["status"] == "FAILED" for item in results)}
    if execution_revision is not None:
        report["execution_code_revision"] = execution_revision
    with state.db:
        if stage == "annotate":
            state.db.execute("UPDATE pilot_batches SET annotate_report = ? WHERE manifest_id = ?",
                             (json.dumps(report), manifest["run_id"]))
        elif stage == "review_labels":
            state.db.execute("UPDATE pilot_batches SET review_report = ? WHERE manifest_id = ?",
                             (json.dumps(report), manifest["run_id"]))
        else:
            state.db.execute("INSERT OR REPLACE INTO pilot_qa_reports VALUES (?, ?, ?)",
                             (manifest["run_id"], stage, json.dumps(report)))
    return report


def annotate(state, contract, run_id, batch_id, limit, annotator_lock, reviewer_lock, *, offset=0, workers=1):
    started = monotonic()
    existing = _batch(state, run_id, batch_id)
    row, source, records, samples, evidence = _inputs(state.db, run_id)
    if not 1 <= limit <= len(samples):
        raise ValueError(f"--limit must be between 1 and {len(samples)}")
    if not 0 <= offset <= len(samples) - limit:
        raise ValueError(f"--offset must be between 0 and {len(samples) - limit} for --limit {limit}")
    sample_ids = list(dict.fromkeys(item["sample_id"] for item in source["selection"]))
    if set(sample_ids) != set(samples):
        raise ValueError("Import manifest and sample IDs disagree")
    if limit < len(samples):
        sample_ids.sort(key=lambda sid: canonical_hash([contract.experiment["seed"], sid]))
    selected_ids = sample_ids[offset:offset + limit]
    selected = [samples[sid] for sid in selected_ids]
    tasks = contract.experiment["annotation"]["model_tasks"]
    work_count = sum(len(sample["target_claims"]) for sample in selected) * len(tasks)
    for role in ("annotator", "reviewer"):
        _check_budget(contract, role, work_count)
    locks = {
        "annotator": load_lock(contract, "annotator", annotator_lock),
        "reviewer": load_lock(contract, "reviewer", reviewer_lock),
    }
    execution_revision = _code_revision(contract)
    revision = execution_revision
    if existing:
        pinned = json.loads(existing["manifest"])["code_revision"]
        if _can_resume_revision(contract, pinned, execution_revision):
            revision = pinned
    plan_digest, plan = freeze_plan(state, run_id, records, contract.experiment["splitting"])
    apply_plan(records, plan)
    if any(sample["split"] != "dev" for sample in selected):
        raise ValueError("Pilot selection must remain entirely in dev")
    split_id = "split_" + plan_digest
    input_hash = canonical_hash([run_id, batch_id, row["digest"], selected_ids, plan_digest,
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
        "runtime": {"source_run_id": run_id, "batch_id": batch_id, "sample_count": limit, "workers": workers,
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
    jobs = ((sample, claim, task, None, ()) for sample in selected for claim in sample["target_claims"] for task in tasks)
    results.extend(_run_model_jobs(state, contract, client, manifest, "annotate", evidence,
                                  Path(row["media_root"]), jobs, workers))
    return _finish(state, manifest, "annotate", results, workers, started, execution_revision=execution_revision)


def review_labels(state, contract, run_id, batch_id, reviewer_lock, *, workers=1):
    started = monotonic()
    batch = _batch(state, run_id, batch_id)
    if batch is None or batch["annotate_report"] is None:
        raise ValueError("Finish annotate for this --run-id and --batch-id before reviewing")
    manifest = json.loads(batch["manifest"])
    _, lock_hash = load_lock(contract, "reviewer", reviewer_lock)
    if lock_hash != manifest["runtime"]["reviewer_lock_sha256"]:
        raise ValueError("Reviewer model lock differs from the annotated batch")
    execution_revision = _code_revision(contract)
    if manifest["config_hash"] != contract.config_hash or not _can_resume_revision(
        contract, manifest["code_revision"], execution_revision
    ):
        raise ValueError("Use the same code and configuration as the annotated batch")
    row, _, _, samples, evidence = _inputs(state.db, run_id)
    ids = [item["record_id"] for item in json.loads(batch["annotate_report"])["results"]
           if item.get("record_id") and item["status"] == "VALIDATED"]
    if not ids:
        raise ValueError("No validated model annotations available for review")
    _check_budget(contract, "reviewer", len(ids))
    client = connect_role(contract, "reviewer")
    jobs = []
    for record_id in ids:
        record = state.db.execute("SELECT record FROM pilot_records WHERE record_id = ?", (record_id,)).fetchone()
        target = json.loads(record["record"])
        claim, task = target["payload"]["target_claim"], target["payload"]["task_id"]
        jobs.append((samples[target["sample_id"]], claim, task, target, ()))
    results = _run_model_jobs(state, contract, client, manifest, "review_labels", evidence,
                              Path(row["media_root"]), jobs, workers)
    return _finish(state, manifest, "review_labels", results, workers, started, execution_revision=execution_revision)


def generate_qa(state, contract, run_id, batch_id, label_batch_id, limit, annotator_lock, reviewer_lock,
                *, workers, dry_run=False):
    started = monotonic()
    row, _, _, samples, evidence = _inputs(state.db, run_id)
    if limit < 1 or batch_id == label_batch_id:
        raise ValueError("QA requires a positive --limit and a new --batch-id distinct from --label-batch-id")
    existing = _batch(state, run_id, batch_id)
    source_batch = _batch(state, run_id, label_batch_id)
    if source_batch is None or not source_batch["review_report"]:
        raise ValueError("Finish review-labels for --label-batch-id before generating QA")
    source = json.loads(source_batch["manifest"])
    locks = {role: load_lock(contract, role, path)[1]
             for role, path in (("annotator", annotator_lock), ("reviewer", reviewer_lock))}
    if any(digest != source["runtime"].get(f"{role}_lock_sha256") for role, digest in locks.items()):
        raise ValueError("QA requires the model locks of the source label batch")
    if source["label_schema_version"] != contract.labels["label_schema_version"]:
        raise ValueError("Source labels use a different label schema version")

    by_id = {record[ID_FIELDS[record["record_type"]]]: record
             for record in snapshot_records(state, run_id, gold_votes=contract.gold_votes)}
    jobs = []
    seen = set()
    model_tasks = contract.experiment["annotation"]["model_tasks"]
    allowed_tiers = contract.experiment["qa"]["label_tiers_allowed"]
    for item in json.loads(source_batch["annotate_report"])["results"]:
        annotation = by_id.get(item.get("record_id"))
        if not annotation or annotation["status"] not in {"ACCEPTED_SILVER", "ACCEPTED_GOLD"}:
            continue
        if annotation["quality_tier"] not in allowed_tiers:
            continue
        sample_id = annotation["sample_id"]
        payload = annotation["payload"]
        task = payload["task_id"]
        if task not in model_tasks or (sample_id, task) in seen:
            continue
        approvals = [by_id[rid] for rid in annotation["review_ids"]]
        if annotation["quality_tier"] == "gold":
            approvals.extend(by_id[rid] for rid in [annotation["adjudication_id"], *annotation["human_vote_ids"]])
        seen.add((sample_id, task))
        jobs.append((samples[sample_id], payload["target_claim"], task, annotation, approvals))
        if len(jobs) == limit:
            break
    if not jobs:
        raise ValueError("No accepted model labels in the source batch are eligible for QA")
    for role in locks:
        _check_budget(contract, role, len(jobs))
    revision = _code_revision(contract)
    input_hash = canonical_hash([
        source["run_id"], batch_id, limit, revision, contract.config_hash,
        [list(_record_dependencies(target)) + [ref for approval in approvals for ref in _record_dependencies(approval)]
         for _, _, _, target, approvals in jobs],
    ])
    data = {key: source[key] for key in ("mode", "label_schema_version", "dataset_version", "evidence_version",
                                        "split_manifest_id", "models", "seed")}
    data.update(run_id="pilot_" + input_hash, code_revision=revision, config_hash=contract.config_hash,
                prompt_versions={stage: contract.experiment["qa"]["prompt_version"]
                                 for stage in ("generate_qa", "review_qa")},
                runtime={"source_run_id": run_id, "batch_id": batch_id, "label_batch_id": label_batch_id,
                         "source_label_manifest_id": source["run_id"], "sample_count": len({job[0]["sample_id"] for job in jobs}),
                         "workers": workers, **{f"{role}_lock_sha256": digest for role, digest in locks.items()}},
                status="RUNNING")
    manifest = {**record_envelope("run_manifest", data, input_hash, is_mock=False), **data}
    contract.validator("run_manifest").validate(manifest)
    if existing and existing["manifest_id"] != manifest["run_id"]:
        raise ValueError("QA batch inputs changed. Use a new --batch-id")
    if dry_run:
        return {"stage": "generate_qa", "dry_run": True, "jobs": len(jobs), "workers": workers, "failed": 0}
    client = connect_role(contract, "annotator")
    with state.db:
        state.db.execute(
            "INSERT OR IGNORE INTO pilot_batches (source_run_id, batch_id, manifest_id, manifest) VALUES (?, ?, ?, ?)",
            (run_id, batch_id, manifest["run_id"], json.dumps(manifest)),
        )
        existing = _batch(state, run_id, batch_id)
        if existing["manifest_id"] != manifest["run_id"]:
            raise ValueError("QA batch inputs changed. Use a new --batch-id")
        manifest = json.loads(existing["manifest"])
    results = _run_model_jobs(state, contract, client, manifest, "generate_qa", evidence,
                              Path(row["media_root"]), jobs, workers)
    return _finish(state, manifest, "generate_qa", results, workers, started)


def review_qa(state, contract, run_id, batch_id, reviewer_lock, *, workers, dry_run=False):
    started = monotonic()
    batch = _batch(state, run_id, batch_id)
    generated = None if batch is None else state.db.execute(
        "SELECT report FROM pilot_qa_reports WHERE manifest_id = ? AND stage = 'generate_qa'", (batch["manifest_id"],),
    ).fetchone()
    if generated is None:
        raise ValueError("Finish generate-qa for this batch before reviewing")
    manifest = json.loads(batch["manifest"])
    if manifest["config_hash"] != contract.config_hash or manifest["code_revision"] != _code_revision(contract):
        raise ValueError("Use the same code and configuration as the QA generation batch")
    _, digest = load_lock(contract, "reviewer", reviewer_lock)
    if digest != manifest["runtime"]["reviewer_lock_sha256"]:
        raise ValueError("Reviewer model lock differs from the QA generation batch")
    row, _, _, samples, evidence = _inputs(state.db, run_id)
    by_id = {record[ID_FIELDS[record["record_type"]]]: record
             for record in snapshot_records(state, run_id, gold_votes=contract.gold_votes)}
    jobs = []
    for item in json.loads(generated["report"])["results"]:
        if item["status"] != "VALIDATED":
            continue
        question = by_id[item["record_id"]]
        annotation = by_id[question["source_annotation_id"]]
        if question["status"] == "STALE" or annotation["status"] not in {"ACCEPTED_SILVER", "ACCEPTED_GOLD"}:
            continue
        jobs.append((samples[question["sample_id"]], annotation["payload"]["target_claim"],
                     question["payload"]["task_id"], question, ()))
    _check_budget(contract, "reviewer", len(jobs))
    if dry_run:
        return {"stage": "review_qa", "dry_run": True, "jobs": len(jobs), "workers": workers, "failed": 0}
    results = []
    if jobs:
        client = connect_role(contract, "reviewer")
        results = _run_model_jobs(state, contract, client, manifest, "review_qa", evidence,
                                  Path(row["media_root"]), jobs, workers)
    return _finish(state, manifest, "review_qa", results, workers, started)
