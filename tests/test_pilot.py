from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from rumorlab.cli import main
from rumorlab.config import Contract
from rumorlab.grouping import freeze_plan
from rumorlab.ingest import import_snapshot
from rumorlab.m1 import load_lock
from rumorlab.mmfakebench import prepare_mmfakebench
from rumorlab.records import record_envelope, stage_record
from rumorlab.snapshot import snapshot_records
from rumorlab.state import State
from rumorlab.validation import validate_rows


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiment.yaml"
REVISION = "f" * 40
RUN = "val-data"
BATCH = "pilot-1"
TASK_LABELS = {
    "reality_plausibility": "KE", "image_text_consistency": "C1",
    "persuasion_strategy": "SOCIAL_PROOF", "intent_analysis": "PUBLIC_INTEREST",
}


class PilotFixture:
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.db_path = self.root / "state.sqlite"
        self.state = State(self.db_path)
        self.addCleanup(self.state.close)
        self.contract = Contract(CONFIG)
        self.locks = {}
        for role, settings in self.contract.models["models"].items():
            lock = {
                "model_id": settings["model_id"], "role": role, "revision": REVISION,
                "processor_revision": REVISION, "backend_version": "0.32.0+cu129",
                "chat_template_hash": "1" * 64, "generation_config": settings["generation"],
                "precision": settings["precision"], "quantization": settings["quantization"],
                "hardware_fingerprint": "Synthetic deployment fixture", "dependency_lock_hash": "2" * 64,
                "vision_processor_config": {"max_images": 2},
            }
            self.locks[role] = self.root / f"{role}.json"
            self.locks[role].write_text(json.dumps(lock), encoding="utf-8")
        path = self.root / "MMFakeBench_val" / "real" / "a.png"
        path.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "examples/images/blue_circle.png", path)
        source = self.root / "MMFakeBench_val.json"
        source.write_text(json.dumps([
            {"text": text, "image_path": "/real/a.png", "gt_answers": "True", "fake_cls": "original"}
            for text in ("A blue circle.", "A blue symbol.")
        ]), encoding="utf-8")
        rows, sampling = prepare_mmfakebench(source, self.root, 42)
        import_snapshot(self.state, rows, self.contract, self.root, RUN, sampling=sampling)
        self.samples = [record for _, record in rows]
        self.sample = self.samples[0]
        self.calls = []
        for mock in (patch("rumorlab.pilot._code_revision", return_value=REVISION),
                     patch.dict("os.environ", {"ANNOTATOR_BASE_URL": "http://127.0.0.1:8000/v1",
                                               "REVIEWER_BASE_URL": "http://127.0.0.1:8000/v1"})):
            mock.start()
            self.addCleanup(mock.stop)

    def command(self, stage, *, expected=0):
        args = ["rumorlab", stage, "--run-id", RUN, "--batch-id", BATCH, "--config", str(CONFIG),
                "--state-db", str(self.db_path), "--reviewer-lock", str(self.locks["reviewer"])]
        if stage == "annotate":
            args += ["--annotator-lock", str(self.locks["annotator"]), "--limit", "1"]
        out, err = StringIO(), StringIO()
        with patch("sys.argv", args), redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(main(), expected, err.getvalue() or out.getvalue())
        return json.loads(out.getvalue()) if out.getvalue() else err.getvalue()

    def payload(self, task, sample=None):
        return {
            "task_id": task, "target_claim": (sample or self.sample)["target_claims"][0], "applicability": "applicable",
            "labels": [TASK_LABELS[task]], "rationale": "Fixture rationale.",
            "evidence_ids": [], "unknown_reason": None,
        }

    def transport(self, reply_payload):
        def request(client, path, body=None):
            if path == "/models":
                return {"data": [{"id": client.settings["model_id"]}]}
            self.calls.append(body)
            return {"id": "fixture", "choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps(reply_payload(body)), "reasoning": "never persisted",
            }}]}
        return patch("rumorlab.vllm.VllmModel._request", new=request)

    def export(self):
        rows = snapshot_records(self.state, RUN)
        errors, _ = validate_rows(list(enumerate(rows, 1)), self.contract.schema, self.contract.labels,
                                  self.contract.gold_votes, self.root)
        self.assertEqual(errors, [])
        return rows


class AnnotateTests(PilotFixture, TestCase):
    def test_real_command_limits_work_and_resumes_without_another_generation(self):
        def reply(body):
            schema = body["response_format"]["json_schema"]["schema"]
            task = schema["properties"]["task_id"]["const"]
            self.assertNotIn("allOf", schema)
            self.assertIn(TASK_LABELS[task], schema["properties"]["labels"]["items"]["enum"])
            self.assertTrue(body["messages"][-1]["content"][0]["image_url"]["url"].startswith("data:image/png;"))
            messages = json.dumps(body["messages"])
            self.assertNotIn("source_veracity", messages)
            samples = [sample for sample in self.samples if sample["text"] in messages]
            self.assertEqual(len(samples), 1)
            return self.payload(task, samples[0])

        with self.transport(reply):
            first = self.command("annotate")
            rerun = self.command("annotate")
        self.assertEqual((first["samples"], first["failed"], len(self.calls)), (1, 0, 4))
        self.assertTrue(all(row["cached"] for row in rerun["results"]))
        records = self.export()
        labels = [record for record in records if record["record_type"] == "annotation"]
        self.assertEqual(len(labels), 6)
        self.assertEqual({record["producer"]["kind"] for record in labels}, {"model", "manifest"})
        self.assertEqual({record["payload"]["labels"][0] for record in labels}, {*TASK_LABELS.values(), "OT", "OI"})
        self.assertTrue(all(not record["is_mock"] and record["quality_tier"] == "unresolved" for record in labels))

    def test_rejects_test_snapshot_before_connecting(self):
        row = self.state.db.execute("SELECT manifest FROM import_manifests").fetchone()
        manifest = json.loads(row["manifest"])
        manifest["source_split"] = "MMFakeBench_test"
        with self.state.db:
            self.state.db.execute("UPDATE import_manifests SET manifest = ?", (json.dumps(manifest),))
        with patch("rumorlab.pilot.connect_role") as connect:
            self.assertIn("only MMFakeBench val", self.command("annotate", expected=1))
        connect.assert_not_called()

    def test_mixed_unknown_is_not_a_candidate(self):
        def reply(body):
            task = body["response_format"]["json_schema"]["schema"]["properties"]["task_id"]["const"]
            messages = json.dumps(body["messages"])
            sample = next(sample for sample in self.samples if sample["text"] in messages)
            payload = self.payload(task, sample)
            if task == "persuasion_strategy":
                payload.update(labels=["UNKNOWN", "SOCIAL_PROOF"], unknown_reason="Unresolved boundary.")
            return payload

        with self.transport(reply):
            report = self.command("annotate", expected=1)
        failure, = [item for item in report["results"] if item["status"] == "FAILED"]
        self.assertEqual((report["failed"], failure["task_id"], failure["category"], len(self.calls)),
                         (1, "persuasion_strategy", "format", 5))
        self.assertEqual(len([row for row in self.export() if row["record_type"] == "annotation"]), 5)


class ReviewTests(PilotFixture, TestCase):
    def seed_annotation(self):
        digest, _ = freeze_plan(self.state, RUN, self.samples, self.contract.experiment["splitting"])
        models, hashes = {}, {}
        for role, path in self.locks.items():
            lock, hashes[role] = load_lock(self.contract, role, path)
            models[role] = {key: lock[key] for key in (
                "model_id", "revision", "backend_version", "processor_revision",
                "chat_template_hash", "precision", "quantization",
            )}
            models[role].update(backend="vllm", thinking_enabled=True)
        data = {
            "run_id": "pilot_seed", "mode": "real", "code_revision": REVISION,
            "config_hash": self.contract.config_hash, "label_schema_version": self.contract.labels["label_schema_version"],
            "dataset_version": self.sample["source"]["dataset_version"], "evidence_version": "none",
            "split_manifest_id": "split_" + digest, "models": models, "seed": 42, "status": "RUNNING",
            "prompt_versions": {"annotate": "1.1.1", "review_labels": "1.1.1"},
            "runtime": {"source_run_id": RUN, "batch_id": BATCH, "sample_count": 1,
                        "reviewer_lock_sha256": hashes["reviewer"]},
        }
        manifest = {**record_envelope("run_manifest", data, digest, is_mock=False), **data}
        record = stage_record(
            {"job_id": "a" * 64, "input_hash": digest, "payload": self.payload("image_text_consistency")},
            self.sample, {"kind": "model", "role": "annotator", "model_id": models["annotator"]["model_id"],
                          "model_revision": REVISION, "run_id": manifest["run_id"],
                          "prompt_version": "1.1.1", "prompt_hash": digest}, [], kind="annotation",
        )
        record["label_schema_version"] = self.contract.labels["label_schema_version"]
        with self.state.db:
            self.state.db.execute(
                "INSERT INTO pilot_batches VALUES (?, ?, ?, ?, ?, NULL)",
                (RUN, BATCH, manifest["run_id"], json.dumps(manifest), json.dumps({
                    "failed": 0, "results": [{"status": "VALIDATED", "record_id": record["annotation_id"]}],
                })),
            )
            self.state.db.execute(
                "INSERT INTO pilot_records VALUES (?, ?, ?)",
                (manifest["run_id"], record["annotation_id"], json.dumps(record)),
            )
        return record

    def test_reviews_saved_candidate_and_projects_silver_without_rewriting_it(self):
        candidate = self.seed_annotation()

        def reply(body):
            self.assertEqual(body["model"], "google/gemma-4-31B-it")
            self.assertIn("Fixture rationale.", json.dumps(body["messages"]))
            return {"decision": "PASS", "issue_codes": [], "rationale": "Review fixture.",
                    "evidence_ids": [], "suggested_annotation": None, "suggested_qa_revision": None}

        with self.transport(reply):
            first = self.command("review-labels")
            rerun = self.command("review-labels")
        self.assertEqual((first["failed"], len(self.calls)), (0, 1))
        self.assertTrue(rerun["results"][0]["cached"])
        indexed = {row["record_type"]: row for row in self.export()}
        annotation, review = indexed["annotation"], indexed["review"]
        self.assertEqual(annotation["annotation_id"], candidate["annotation_id"])
        self.assertEqual(annotation["payload"], candidate["payload"])
        self.assertEqual((annotation["status"], annotation["quality_tier"]), ("ACCEPTED_SILVER", "silver"))
        self.assertEqual(review["target_content_hash"], candidate["content_hash"])
        self.state.invalidate(review["review_id"], review["content_hash"])
        annotation = next(row for row in self.export() if row["record_type"] == "annotation")
        self.assertEqual((annotation["status"], annotation["quality_tier"]), ("VALIDATED", "unresolved"))

    def test_structural_decoder_does_not_bypass_local_conditional_validation(self):
        self.seed_annotation()
        with self.transport(lambda _: {"decision": "PASS", "issue_codes": ["INSUFFICIENT_EVIDENCE"],
                                       "rationale": "Invalid PASS.", "evidence_ids": [],
                                       "suggested_annotation": None, "suggested_qa_revision": None}):
            result = self.command("review-labels", expected=1)
        self.assertEqual((result["failed"], result["results"][0]["category"], len(self.calls)), (1, "format", 2))
        self.assertFalse(any(row["record_type"] == "review" for row in self.export()))
