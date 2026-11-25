"""Execute version-bound model jobs and the bundled offline smoke run."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from .models import MockModel
from .grouping import apply_plan, freeze_plan
from .provenance import run_source_tasks
from .qa import solver_projection, validate_candidate
from .requests import build_request
from .records import mock_manifest, stage_record
from .state import LeaseBusy, LeaseLost
from .validation import canonical_hash


class JobFailed(ValueError):
    def __init__(self, request, job_id, *, technical_error=None):
        super().__init__(f"Job failed: {request.stage}/{request.task_id} ({job_id})")
        self.job_id = job_id
        self.technical_error = technical_error


def model_job_spec(request, subject, dependencies, script_hash):
    return request.stage, subject, request.task_id, canonical_hash([request.request_id, script_hash]), dependencies


def execute(state, contract, request, subject, dependencies, client, *, script_hash, claim=None, labels=None,
            prepared=None):
    settings = contract.experiment["runtime"]
    spec = model_job_spec(request, subject, dependencies, script_hash)
    input_hash = spec[3]
    while True:
        try:
            job_id, lease, cached = state.claim(*spec, settings["lease_seconds"])
        except LeaseBusy as exc:
            raise JobFailed(request, exc.job_id, technical_error="lease_busy") from exc
        if cached is not None:
            return cached, True
        if lease is None:
            raise JobFailed(request, job_id)
        try:
            with state.keep_lease(job_id, lease, settings["lease_seconds"], settings["heartbeat_seconds"]):
                response = client.generate(request) if prepared is None else client.generate(request, prepared=prepared)
                category = None
                if response.technical_error:
                    category = "technical"
                    retry_limit = settings["technical_max_retries"]
                elif not response.schema_valid:
                    category = "format"
                    retry_limit = settings["format_repair_max_attempts"]
                    if request.stage in {"generate_qa", "review_qa"}:
                        retry_limit = min(retry_limit, settings["qa_format_revision_max_attempts"])
                elif request.stage == "generate_qa" and "skip_reason" in response.parsed_payload:
                    payload = response.parsed_payload
                else:
                    payload = response.parsed_payload
                    visible = {item["evidence_id"] for item in request.visible_inputs["evidence"]}
                    suggested = payload.get("suggested_annotation")
                    cites = set(payload["evidence_ids"])
                    if suggested:
                        cites.update(suggested["evidence_ids"])
                    invalid = not cites <= visible
                    annotation = payload if request.stage == "annotate" else suggested
                    if annotation is not None and request.stage in {"annotate", "review_labels"}:
                        invalid |= (request.stage == "annotate" or claim is not None) and annotation["target_claim"] != claim
                        invalid |= annotation["applicability"] == "not_applicable" and request.task_id != "reality_plausibility"
                    if request.stage == "generate_qa":
                        try:
                            allowed = {item["code"] for item in contract.labels["tasks"][request.task_id]["labels"]}
                            validate_candidate(payload, request.visible_inputs, allowed,
                                               seed=contract.experiment["seed"], qa_id=request.request_id)
                        except ValueError:
                            invalid = True
                        invalid |= set(payload["target_labels"]) != set(labels)
                    if invalid:
                        category = "semantic"
                        retry_limit = 0
                if category:
                    state.finish(job_id, lease, error={"category": category, "response": asdict(response)}, retry_limit=retry_limit)
                    continue
                result = {
                    "is_mock": response.is_mock, "job_id": job_id, "input_hash": input_hash,
                    "payload": payload, "content_hash": canonical_hash(payload), "response": asdict(response),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                state.finish(job_id, lease, result=result)
                return result, False
        except LeaseLost as exc:
            raise JobFailed(request, job_id, technical_error="lease_lost") from exc


def _fixture_cases(by_type):
    annotations = {row["annotation_id"]: row for row in by_type.get("annotation", [])}
    reviews = {row["target_id"]: row for row in by_type.get("review", [])}
    for question in by_type.get("qa", []):
        annotation = annotations[question["source_annotation_id"]]
        qa_review = {
            "decision": "PASS", "issue_codes": [], "rationale": "Synthetic QA workflow fixture.",
            "evidence_ids": question["payload"]["evidence_ids"],
            "suggested_annotation": None, "suggested_qa_revision": None,
        }
        yield {
            "sample_id": annotation["sample_id"],
            "claim_id": annotation["payload"]["target_claim"]["claim_id"],
            "task_id": annotation["payload"]["task_id"],
            "responses": {
                "annotate": annotation["payload"],
                "review_labels": reviews[annotation["annotation_id"]]["payload"],
                "generate_qa": question["payload"], "review_qa": qa_review,
            },
            "dependencies": {
                "review_labels": (annotation["annotation_id"], annotation["content_hash"]),
                "review_qa": (question["qa_id"], question["content_hash"]),
            },
        }


def mock_run(state, contract, run_id, *, cases=None, provenance=False):
    snapshot = state.db.execute("SELECT * FROM imports WHERE run_id = ?", (run_id,)).fetchone()
    if snapshot is None:
        raise ValueError(f"No imported snapshot for {run_id!r}")
    records = json.loads(snapshot["records"])
    if any(row["is_mock"] is not True for row in records):
        raise ValueError("Mock smoke requires a synthetic snapshot")
    plan_digest, plan = freeze_plan(state, run_id, records, contract.experiment["splitting"])
    apply_plan(records, plan)
    split_dependency = ("split_" + plan_digest, plan_digest)
    manifest = mock_manifest(contract, run_id, split_dependency[0])
    output = [manifest]
    by_type = {}
    for record in records:
        by_type.setdefault(record["record_type"], []).append(record)
    external = cases is not None
    cases = list(cases if external else _fixture_cases(by_type))
    if not cases and not provenance:
        raise ValueError("Mock run requires scripted cases or a bundled QA fixture")
    samples = {row["sample_id"]: row for row in by_type.get("sample", [])}
    stages = ("annotate", "review_labels", "generate_qa", "review_qa")
    seen = set()
    report = run_source_tasks(
        state, contract, samples.values(), by_type.get("evidence", []), split_dependency, manifest, output,
    ) if provenance else []
    for case in cases:
        key = (case["sample_id"], case["claim_id"], case["task_id"])
        if key in seen:
            raise ValueError(f"Duplicate mock case: {key}")
        seen.add(key)
        sample_id, claim_id, task = key
        sample = samples[sample_id]
        claims = [item for item in sample["target_claims"] if item["claim_id"] == claim_id]
        if len(claims) != 1:
            raise ValueError(f"Expected one registered claim for mock case {key}")
        claim = claims[0]
        if task not in contract.experiment["annotation"]["model_tasks"]:
            raise ValueError(f"Mock scripted annotation requires a model task: {task}")
        responses = case["responses"]
        if not isinstance(responses, dict) or not set(responses) <= set(stages):
            raise ValueError(f"Invalid mock stage names: {key}")
        deps = [(sample["sample_id"], sample["content_hash"]), split_dependency]
        deps.extend((row["evidence_id"], row["content_hash"]) for row in by_type.get("evidence", [])
                    if row["evidence_id"] in sample["evidence_ids"])
        upstream = None
        candidate = None
        candidate_record = None
        annotation_record = None
        labels = None
        for stage in stages:
            if stage not in responses:
                report.append({"stage": stage, "status": "MISSING_SCRIPT",
                               "sample_id": sample_id, "claim_id": claim_id, "task_id": task})
                break
            script = responses[stage]
            if not external and stage in case["dependencies"]:
                deps.append(case["dependencies"][stage])
            request = build_request(
                contract, stage, task, sample, claim, by_type.get("evidence", []),
                Path(snapshot["media_root"]), upstream=upstream,
            )
            result, cached = execute(
                state, contract, request, sample["sample_id"], deps,
                MockModel({request.request_id: script}), script_hash=canonical_hash(script),
                claim=claim, labels=labels,
            )
            deps.append((result["job_id"], result["content_hash"]))
            producer = {
                "kind": "mock", "role": request.role, "model_id": contract.models["models"][request.role]["model_id"],
                "model_revision": None, "run_id": manifest["run_id"],
                "prompt_version": contract.experiment["annotation"]["prompt_version"],
                "prompt_hash": request.rendered_prompt_hash,
            }
            evidence_ids = [item["evidence_id"] for item in request.visible_inputs["evidence"]]
            report.append({"stage": stage, "job_id": result["job_id"], "cached": cached,
                           "sample_id": sample_id, "claim_id": claim_id, "task_id": task})
            if stage == "generate_qa" and "skip_reason" in result["payload"]:
                report[-1].update(status="SKIPPED", reason=result["payload"]["skip_reason"])
                break
            if stage in {"annotate", "generate_qa"}:
                candidate = {**result, "sample_id": sample["sample_id"], "status": "GENERATED"}
                upstream = candidate
                if stage == "annotate":
                    labels = result["payload"]["labels"]
                    candidate_record = stage_record(result, sample, producer, evidence_ids, kind="annotation")
                    annotation_record = candidate_record
                else:
                    candidate_record = stage_record(
                        result, sample, producer, evidence_ids, kind="qa", target=annotation_record,
                        solver=solver_projection(request, sample),
                    )
                candidate_record["label_schema_version"] = contract.labels["label_schema_version"]
                output.append(candidate_record)
                deps.append((candidate_record[f"{candidate_record['record_type']}_id"], candidate_record["content_hash"]))
                if stage == "annotate" and result["payload"]["applicability"] == "not_applicable":
                    report[-1]["status"] = "SKIPPED"
                    break
            else:
                decision = result["payload"]["decision"]
                review_record = stage_record(result, sample, producer, evidence_ids, kind="review", target=candidate_record)
                output.append(review_record)
                candidate["status"] = {
                    "PASS": "ACCEPTED_SILVER", "NEEDS_HUMAN": "HUMAN_PENDING", "REJECT": "EXCLUDED",
                }[decision]
                candidate_record["status"] = candidate["status"]
                candidate_record["review_ids"].append(review_record["review_id"])
                if decision == "PASS":
                    candidate_record["quality_tier"] = "silver"
                report[-1]["status"] = candidate["status"]
                if decision != "PASS":
                    break
    for record in output:
        contract.validator(record["record_type"]).validate(record)
    with state.db:
        state.db.execute(
            "INSERT INTO mock_exports VALUES (?, ?) ON CONFLICT(run_id) DO UPDATE SET records = excluded.records",
            (run_id, json.dumps(output, ensure_ascii=False)),
        )
    return {"is_mock": True, "run_id": run_id, "split_manifest_id": split_dependency[0], "stages": report}
