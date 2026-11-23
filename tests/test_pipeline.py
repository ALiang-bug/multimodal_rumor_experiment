from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from rumorlab.cli import read_jsonl
from rumorlab.config import Contract
from rumorlab.ingest import ingest
from rumorlab.pipeline import execute, mock_run
from rumorlab.models import MockModel
from rumorlab.requests import build_request
from rumorlab.provenance import source_annotation
from rumorlab.release import check_release
from rumorlab.snapshot import snapshot_records
from rumorlab.state import State
from rumorlab.validation import validate_rows
from rumorlab.validation import canonical_hash


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(TestCase):
    def test_t07_qa_format_repair_is_bounded(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        rows = [row for _, row in read_jsonl(ROOT / "examples/records.jsonl")]
        annotation = next(row for row in rows if row["record_type"] == "annotation")
        sample = next(row for row in rows if row["record_type"] == "sample"
                      and row["sample_id"] == annotation["sample_id"])
        payload = annotation["payload"]
        request = build_request(
            contract, "generate_qa", payload["task_id"], sample, payload["target_claim"],
            [row for row in rows if row["record_type"] == "evidence"], ROOT, upstream=annotation,
        )
        with TemporaryDirectory() as directory:
            state = State(Path(directory) / "state.db")
            try:
                for limit in (0, 1):
                    contract.experiment["runtime"]["qa_format_revision_max_attempts"] = limit
                    with self.assertRaisesRegex(ValueError, "Job failed"):
                        execute(state, contract, request, f"case_{limit}", [],
                                MockModel({request.request_id: "invalid JSON"}), script_hash=str(limit))
                    result = state.db.execute(
                        "SELECT state, format_repairs, result FROM jobs WHERE subject = ?", (f"case_{limit}",),
                    ).fetchone()
                    self.assertEqual(tuple(result), ("FAILED", limit, None))
            finally:
                state.close()

    def test_t09_evidence_revision_import_invalidates_cached_downstream(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        rows = list(read_jsonl(ROOT / "examples/records.jsonl"))
        old = next(row for _, row in rows if row["record_type"] == "evidence")
        updated = {
            **old, "evidence_id": "synthetic_evidence_v2", "record_version": 2,
            "supersedes_id": old["evidence_id"], "summary": "Synthetic revised evidence summary.",
        }
        updated["content_hash"] = canonical_hash(updated)
        with TemporaryDirectory() as directory:
            state = State(Path(directory) / "state.db")
            try:
                ingest(state, rows, contract, ROOT, "fixture", fixture=True)
                job_ids = {row["job_id"] for row in mock_run(state, contract, "fixture")["stages"]}
                self.assertEqual(len(job_ids), 4)
                ingest(state, [*rows, (len(rows) + 1, updated)], contract, ROOT, "revision", fixture=True)
                stale = {row["id"] for row in state.db.execute("SELECT id FROM jobs WHERE state = 'STALE'")}
                self.assertEqual(stale, job_ids)
                with self.assertRaisesRegex(ValueError, "Stale dependency"):
                    mock_run(state, contract, "fixture")
                projected = snapshot_records(state, "fixture")
                self.assertTrue(all(row["status"] == "STALE" for row in projected
                                    if row["record_type"] in {"annotation", "review", "qa"}))
                revised = snapshot_records(state, "revision")
                self.assertIn(updated, revised)
                self.assertIn({**old, "status": "EXCLUDED"}, revised)
            finally:
                state.close()

    def test_t03_t04_invalid_outputs_exhaust_only_their_own_retry_budget(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        rows = list(read_jsonl(ROOT / "examples/records.jsonl"))
        sample = next(row for _, row in rows if row["record_type"] == "sample")
        evidence = [row for _, row in rows if row["record_type"] == "evidence"]
        claim = sample["target_claims"][0]
        request = build_request(contract, "annotate", "persuasion_strategy", sample, claim, evidence, ROOT)
        payload = {
            "task_id": "persuasion_strategy", "target_claim": claim, "applicability": "applicable",
            "labels": ["UNKNOWN"], "rationale": "Synthetic fixture.",
            "evidence_ids": [], "unknown_reason": "evidence_insufficient",
        }
        failures = [
            ("timeout", TimeoutError("fixture timeout"), "technical", 3),
            ("json", "{broken", "format", 2),
            ("mixed", {**payload, "labels": ["UNKNOWN", "AUTHORITY"]}, "format", 2),
            ("citation", {**payload, "evidence_ids": ["missing"]}, "semantic", 1),
        ]
        with TemporaryDirectory() as directory:
            state = State(Path(directory) / "state.db")
            try:
                for name, script, category, attempts in failures:
                    with self.subTest(name=name):
                        with self.assertRaisesRegex(ValueError, "Job failed"):
                            execute(state, contract, request, name, [], MockModel({request.request_id: script}),
                                    script_hash=name, claim=claim)
                        job = state.db.execute("SELECT * FROM jobs WHERE subject = ?", (name,)).fetchone()
                        self.assertEqual(job["state"], "FAILED")
                        self.assertIsNone(job["result"])
                        events = state.db.execute(
                            "SELECT state, detail FROM events WHERE job_id = ? ORDER BY sequence", (job["id"],),
                        ).fetchall()
                        self.assertEqual(sum(event["state"] == "RUNNING" for event in events), attempts)
                        self.assertIn(category, events[-1]["detail"])
            finally:
                state.close()

    def test_t03_t04_skip_and_insufficient_evidence_review(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        rows = list(read_jsonl(ROOT / "examples/records.jsonl"))
        sample = next(row for _, row in rows if row["record_type"] == "sample")
        claim = sample["target_claims"][0]
        key = {"sample_id": sample["sample_id"], "claim_id": claim["claim_id"], "task_id": "reality_plausibility"}
        payload = {"task_id": key["task_id"], "target_claim": claim, "applicability": "not_applicable",
                   "labels": [], "rationale": "Synthetic verified sample.", "evidence_ids": [], "unknown_reason": None}
        review = {"decision": "NEEDS_HUMAN", "issue_codes": ["EVIDENCE_MISSING"],
                  "rationale": "Synthetic example without evidence for EF.", "evidence_ids": [],
                  "suggested_annotation": None, "suggested_qa_revision": None}
        with TemporaryDirectory() as directory:
            state = State(Path(directory) / "state.db")
            try:
                ingest(state, rows, contract, ROOT, "fixture", fixture=True)
                skip = mock_run(state, contract, "fixture", cases=[{**key, "responses": {"annotate": payload}}])
                self.assertEqual([row["status"] for row in skip["stages"]], ["SKIPPED"])
                active = {**payload, "applicability": "applicable", "labels": ["EF"]}
                routed = mock_run(state, contract, "fixture", cases=[{
                    **key, "responses": {"annotate": active, "review_labels": review},
                }])
                self.assertEqual([row["stage"] for row in routed["stages"]], ["annotate", "review_labels"])
                self.assertEqual(routed["stages"][-1]["status"], "HUMAN_PENDING")
                rejected = mock_run(state, contract, "fixture", cases=[{
                    **key, "responses": {"annotate": active, "review_labels": {**review, "decision": "REJECT"}},
                }])
                self.assertEqual([row["stage"] for row in rejected["stages"]], ["annotate", "review_labels"])
                self.assertEqual(rejected["stages"][-1]["status"], "EXCLUDED")
                rejected_id = rejected["stages"][-1]["job_id"]
                attempts = state.db.execute(
                    "SELECT count(*) FROM events WHERE job_id = ? AND state = 'RUNNING'", (rejected_id,),
                ).fetchone()[0]
                self.assertEqual(attempts, 1)
                consistency = {**key, "task_id": "image_text_consistency"}
                c1 = {**active, "task_id": consistency["task_id"], "labels": ["C1"]}
                routed_c1 = mock_run(state, contract, "fixture", cases=[{
                    **consistency, "responses": {"annotate": c1, "review_labels": review},
                }])
                self.assertEqual(len(routed_c1["stages"]), 2)
                self.assertEqual(routed_c1["stages"][-1]["status"], "HUMAN_PENDING")
            finally:
                state.close()

    def test_manifest_provenance_batch_uses_no_model_scripts(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        rows = list(read_jsonl(ROOT / "examples/records.jsonl"))
        with TemporaryDirectory() as directory:
            state = State(Path(directory) / "state.db")
            try:
                ingest(state, rows, contract, ROOT, "fixture", fixture=True)
                fresh = mock_run(state, contract, "fixture", cases=[], provenance=True)["stages"]
                cached = mock_run(state, contract, "fixture", cases=[], provenance=True)["stages"]
            finally:
                state.close()
        self.assertEqual([row["labels"][0] for row in fresh], ["OT", "OI", "FT", "OI"])
        self.assertTrue(all(row["producer"] == "manifest" and row["status"] == "VALIDATED" for row in fresh))
        self.assertEqual(cached, [{**row, "cached": True} for row in fresh])

        original = next(row for _, row in rows if row.get("text_provenance") == "OT")
        evidence = [row for _, row in rows if row["record_type"] == "evidence"]
        self.assertEqual(source_annotation({**original, "source": {"source_veracity": False}},
                                           original["target_claims"][0], "text_reasoning", evidence)["labels"], ["OT"])
        with self.assertRaisesRegex(ValueError, "verified manipulation"):
            source_annotation({**original, "image_provenance": "FI"},
                              original["target_claims"][0], "image_perception", evidence)
        derivative = next(row for _, row in rows if row.get("text_provenance") == "FT")
        # Exercise the FI provenance mapper with a synthetic verified operation envelope.
        fi = {**derivative, "text_provenance": "OT", "image_provenance": "FI"}
        contract.validator("sample").validate(fi)
        mapped = source_annotation(fi, fi["target_claims"][0], "image_perception", evidence)
        self.assertEqual(mapped["labels"], ["FI"])
        self.assertEqual(mapped["evidence_ids"], fi["operation"]["verification_evidence_ids"])

    def test_scripted_batch_keeps_tasks_independent(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        rows = list(read_jsonl(ROOT / "examples/records.jsonl"))
        samples = [row for _, row in rows if row["record_type"] == "sample"]
        cases = []
        for sample, task in zip(samples, ("intent_analysis", "persuasion_strategy")):
            claim = sample["target_claims"][0]
            cases.append({"sample_id": sample["sample_id"], "claim_id": claim["claim_id"], "task_id": task,
                          "responses": {"annotate": {
                              "task_id": task, "target_claim": claim, "applicability": "applicable",
                              "labels": ["UNKNOWN"], "rationale": "Synthetic script for workflow test.",
                              "evidence_ids": [], "unknown_reason": "evidence_insufficient",
                          }}})
        with TemporaryDirectory() as directory:
            state = State(Path(directory) / "state.db")
            try:
                ingest(state, rows, contract, ROOT, "fixture", fixture=True)
                stages = mock_run(state, contract, "fixture", cases=cases)["stages"]
            finally:
                state.close()
        self.assertEqual([row["task_id"] for row in stages],
                         ["intent_analysis", "intent_analysis", "persuasion_strategy", "persuasion_strategy"])
        self.assertEqual([row["status"] for row in stages if row["stage"] == "review_labels"],
                         ["MISSING_SCRIPT", "MISSING_SCRIPT"])

    def test_snapshot_projects_stale_annotations_reviews_and_qa(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        rows = list(read_jsonl(ROOT / "examples/records.jsonl"))
        annotation = next(row for _, row in rows if row["record_type"] == "annotation")
        with TemporaryDirectory() as directory:
            state = State(Path(directory) / "state.db")
            try:
                ingest(state, rows, contract, ROOT, "fixture", fixture=True)
                self.assertEqual(snapshot_records(state, "fixture"), [row for _, row in rows])
                state.invalidate(annotation["annotation_id"], annotation["content_hash"])
                exported = snapshot_records(state, "fixture")
            finally:
                state.close()
        snapshots = list(enumerate(exported, 1))
        errors, _ = validate_rows(snapshots, contract.schema, contract.labels, contract.gold_votes, ROOT)
        self.assertEqual(errors, [])
        self.assertEqual({row["record_type"] for row in exported if row["status"] == "STALE"},
                         {"annotation", "review", "qa"})
        self.assertEqual(sum("unusable release status STALE" in error
                             for error in check_release(snapshots, contract, ROOT)), 3)
        self.assertEqual(annotation["status"], "ACCEPTED_SILVER")

    def test_import_and_four_stages_resume_after_reopen(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        rows = list(read_jsonl(ROOT / "examples/records.jsonl"))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.db"
            state = State(path)
            try:
                self.assertEqual(ingest(state, rows, contract, ROOT, "fixture", fixture=True), (8, True))
                self.assertEqual(ingest(state, rows, contract, ROOT, "fixture", fixture=True), (8, False))
                report = mock_run(state, contract, "fixture")
                fresh = report["stages"]
                split_id = report["split_manifest_id"]
                self.assertEqual(state.db.execute(
                    "SELECT count(*) FROM dependencies WHERE record_id = ?", (split_id,),
                ).fetchone()[0], 4)
                exported = snapshot_records(state, "fixture")
                errors, _ = validate_rows(
                    list(enumerate(exported, 1)), contract.schema, contract.labels, contract.gold_votes, ROOT,
                )
                self.assertEqual(errors, [])
                generated = [row for row in exported if row["mock_note"] == "Scripted offline workflow output."]
                self.assertEqual([row["record_type"] for row in generated],
                                 ["run_manifest", "annotation", "review", "qa", "review"])
                self.assertEqual(generated[3]["quality_tier"], "silver")
                self.assertEqual(generated[4]["target_id"], generated[3]["qa_id"])
                samples = [row for row in exported if row["record_type"] == "sample"]
                self.assertEqual({row["split"] for row in samples}, {"dev"})
                self.assertEqual(len({row["leakage_group_id"] for row in samples}), 1)
                self.assertNotEqual(samples[0]["leakage_group_id"], "group_mock_001")
                event_count = state.db.execute("SELECT count(*) FROM events").fetchone()[0]
            finally:
                state.close()
            reopened = State(path)
            try:
                resumed = mock_run(reopened, contract, "fixture")
                self.assertEqual(resumed["split_manifest_id"], split_id)
                cached = resumed["stages"]
                self.assertEqual(event_count, reopened.db.execute("SELECT count(*) FROM events").fetchone()[0])
            finally:
                reopened.close()
        self.assertEqual([row["stage"] for row in fresh], ["annotate", "review_labels", "generate_qa", "review_qa"])
        self.assertTrue(all(not row["cached"] for row in fresh))
        self.assertEqual(cached, [{**row, "cached": True} for row in fresh])
        self.assertEqual(fresh[1]["status"], "ACCEPTED_SILVER")
        self.assertEqual(fresh[3]["status"], "ACCEPTED_SILVER")
