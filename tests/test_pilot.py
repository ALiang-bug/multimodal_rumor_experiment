from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from threading import Barrier, Event
from unittest import TestCase
from unittest.mock import patch

from jsonschema import Draft202012Validator

from rumorlab.cli import main
from rumorlab.config import Contract
from rumorlab.grouping import freeze_plan
from rumorlab.ingest import import_snapshot
from rumorlab.m1 import load_lock
from rumorlab.mmfakebench import prepare_mmfakebench
from rumorlab.provenance import prepare_source_tasks, save_source_tasks
from rumorlab.qa import solver_projection
from rumorlab.records import record_envelope, stage_record
from rumorlab.requests import build_request
from rumorlab.snapshot import snapshot_records
from rumorlab.state import LeaseLost, State
from rumorlab.validation import validate_rows
from rumorlab.vllm import VllmModel


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

    def command(self, stage, *, expected=0, workers=None, batch=BATCH, extra=()):
        args = ["rumorlab", stage, "--run-id", RUN, "--batch-id", batch, "--config", str(CONFIG),
                "--state-db", str(self.db_path), "--reviewer-lock", str(self.locks["reviewer"])]
        if stage in {"annotate", "generate-qa"}:
            args += ["--annotator-lock", str(self.locks["annotator"])]
            args += ["--limit", "1"] if stage == "annotate" else ["--label-batch-id", BATCH]
        if workers is not None:
            args += ["--workers", str(workers)]
        args.extend(extra)
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
            if isinstance(body, bytes):
                body = json.loads(body)
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
    def test_prepares_one_queue_and_overlaps_source_computation_without_source_writes(self):
        overlap = Barrier(8, timeout=5)
        source_started, model_started = Event(), Event()
        prepared = []
        original_prepare = VllmModel.prepare

        def prepare(client, request):
            wire = original_prepare(client, request)
            self.assertEqual(wire, json.dumps(client._body(request, None)).encode("utf-8"))
            prepared.append(request.request_id)
            return wire

        def reply(body):
            self.assertEqual(len(prepared), 8)
            model_started.set()
            self.assertTrue(source_started.wait(5))
            overlap.wait()
            properties = body["response_format"]["json_schema"]["schema"]["properties"]
            sample = next(row for row in self.samples if row["target_claims"][0] == properties["target_claim"]["const"])
            return self.payload(properties["task_id"]["const"], sample)

        def prepare_sources(*args):
            source_started.set()
            self.assertTrue(model_started.wait(5))
            return prepare_source_tasks(*args)

        def save_sources(*args):
            self.assertEqual(len(self.calls), 8)
            count = args[0].db.execute("""
                SELECT COUNT(*) FROM jobs WHERE state='SUCCEEDED'
                AND task NOT IN ('text_reasoning', 'image_perception')
            """).fetchone()[0]
            self.assertEqual(count, 8)
            return save_source_tasks(*args)

        offsets = ("--batch-offsets", "0", "1")
        with (self.transport(reply), patch.object(VllmModel, "prepare", new=prepare),
              patch("rumorlab.pilot.prepare_source_tasks", new=prepare_sources),
              patch("rumorlab.pilot.save_source_tasks", new=save_sources)):
            group = self.command("annotate", workers=8, extra=offsets)
        self.assertEqual(group["failed"], 0)
        self.assertEqual(group["run_timings"]["prepared_requests"], 8)
        self.assertEqual([report["batch_id"] for report in group["batches"]], [f"{BATCH}-0", f"{BATCH}-1"])

        with self.transport(lambda _: self.fail("Cached batch must not generate again")):
            resumed = self.command("annotate", workers=8, extra=offsets)
        self.assertTrue(all(row["cached"] for report in resumed["batches"] for row in report["results"]))

    def test_lease_failure_isolated_and_resume_reuses_completed_results(self):
        original_finish = State.finish
        interrupted = set()

        def expire_one(state, job_id, lease, **kwargs):
            payload = (kwargs.get("result") or {}).get("payload", {})
            if payload.get("task_id") == "intent_analysis" and not interrupted:
                with state.db:
                    state.db.execute("UPDATE jobs SET expires = 0 WHERE id = ?", (job_id,))
                interrupted.add(job_id)
                raise LeaseLost(f"Job lease lost: {job_id}")
            return original_finish(state, job_id, lease, **kwargs)

        def reply(body):
            properties = body["response_format"]["json_schema"]["schema"]["properties"]
            sample = next(row for row in self.samples if row["target_claims"][0] == properties["target_claim"]["const"])
            return self.payload(properties["task_id"]["const"], sample)

        with self.transport(reply), patch.object(State, "finish", new=expire_one):
            first = self.command("annotate", expected=1)
            self.assertEqual(first["failed"], 1)
            self.assertEqual([row["technical_error"] for row in first["results"] if row["status"] == "FAILED"],
                             ["lease_lost"])
            self.assertEqual(len(self.calls), 4)
            with (patch("rumorlab.pilot._code_revision", return_value="e" * 40),
                  patch("rumorlab.pilot._can_resume_revision", return_value=True)):
                resumed = self.command("annotate")
        self.assertEqual((resumed["manifest_id"], resumed["failed"], len(self.calls)), (first["manifest_id"], 0, 5))
        self.assertEqual(resumed["execution_code_revision"], "e" * 40)
        self.assertEqual({row["job_id"] for row in first["results"]}, {row["job_id"] for row in resumed["results"]})

    def test_real_command_limits_work_and_resumes_without_another_generation(self):
        overlap = Barrier(2, timeout=5)

        def reply(body):
            overlap.wait()
            schema = body["response_format"]["json_schema"]["schema"]
            task = schema["properties"]["task_id"]["const"]
            self.assertNotIn("allOf", schema)
            self.assertIn(TASK_LABELS[task], schema["properties"]["labels"]["items"]["enum"])
            self.assertTrue(body["messages"][-1]["content"][0]["image_url"]["url"].startswith("data:image/png;"))
            messages = json.dumps(body["messages"])
            self.assertNotIn("source_veracity", messages)
            samples = [sample for sample in self.samples if sample["text"] in messages]
            self.assertEqual(len(samples), 1)
            payload = self.payload(task, samples[0])
            decoder = Draft202012Validator(schema)
            decoder.validate(payload)
            self.assertFalse(decoder.is_valid({**payload, "evidence_ids": ["image_1"]}))
            changed_claim = {**payload["target_claim"], "text": "Changed caption"}
            self.assertFalse(decoder.is_valid({**payload, "target_claim": changed_claim}))
            return payload

        with self.transport(reply):
            first = self.command("annotate", workers=2)
            rerun = self.command("annotate")
        self.assertEqual((first["samples"], first["workers"], first["failed"], len(self.calls)), (1, 2, 0, 4))
        self.assertEqual(rerun["workers"], 48)
        self.assertEqual([row["task_id"] for row in first["results"] if row["task_id"] in TASK_LABELS], list(TASK_LABELS))
        self.assertTrue(all(row["cached"] for row in rerun["results"]))
        records = self.export()
        labels = [record for record in records if record["record_type"] == "annotation"]
        self.assertEqual(len(labels), 6)
        self.assertEqual({record["producer"]["kind"] for record in labels}, {"model", "manifest"})
        self.assertEqual({record["payload"]["labels"][0] for record in labels}, {*TASK_LABELS.values(), "OT", "OI"})
        self.assertTrue(all(not record["is_mock"] and record["quality_tier"] == "unresolved" for record in labels))

    def test_val_offsets_select_disjoint_samples_and_freeze_the_batch(self):
        def reply(body):
            task = body["response_format"]["json_schema"]["schema"]["properties"]["task_id"]["const"]
            sample = next(sample for sample in self.samples if sample["text"] in json.dumps(body["messages"]))
            return self.payload(task, sample)

        with self.transport(reply):
            first = self.command("annotate")
            second = self.command("annotate", batch="pilot-2", extra=("--offset", "1"))
            resumed = self.command("annotate", batch="pilot-2", extra=("--offset", "1"))
            changed = self.command("annotate", batch="pilot-2", expected=1)
            invalid = self.command("annotate", extra=("--offset", "2"), expected=1)

        self.assertEqual({row["sample_id"] for row in first["results"] + second["results"]},
                         {sample["sample_id"] for sample in self.samples})
        self.assertEqual(second["manifest_id"], resumed["manifest_id"])
        self.assertEqual(len(self.calls), 8)
        self.assertIn("Batch inputs changed", changed)
        self.assertIn("--offset must be between 0 and 1", invalid)

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


class LabelFixture(PilotFixture):
    def seed_annotations(self, *tasks):
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
            "prompt_versions": {stage: self.contract.experiment["annotation"]["prompt_version"]
                                for stage in ("annotate", "review_labels")},
            "runtime": {"source_run_id": RUN, "batch_id": BATCH, "sample_count": 1,
                        **{f"{role}_lock_sha256": digest for role, digest in hashes.items()}},
        }
        manifest = {**record_envelope("run_manifest", data, digest, is_mock=False), **data}
        records = []
        for index, task in enumerate(tasks or ("image_text_consistency",), start=1):
            record = stage_record(
                {"job_id": f"{index:064x}", "input_hash": f"{index:064x}", "payload": self.payload(task)},
                self.sample, {"kind": "model", "role": "annotator", "model_id": models["annotator"]["model_id"],
                              "model_revision": REVISION, "run_id": manifest["run_id"],
                              "prompt_version": data["prompt_versions"]["annotate"], "prompt_hash": digest},
                [], kind="annotation",
            )
            record["label_schema_version"] = self.contract.labels["label_schema_version"]
            records.append(record)
        with self.state.db:
            self.state.db.execute(
                "INSERT INTO pilot_batches VALUES (?, ?, ?, ?, ?, NULL)",
                (RUN, BATCH, manifest["run_id"], json.dumps(manifest), json.dumps({
                    "failed": 0, "results": [{"status": "VALIDATED", "record_id": row["annotation_id"]} for row in records],
                })),
            )
            self.state.db.executemany(
                "INSERT INTO pilot_records VALUES (?, ?, ?)",
                [(manifest["run_id"], row["annotation_id"], json.dumps(row)) for row in records],
            )
        return records


class ReviewTests(LabelFixture, TestCase):
    def test_reviews_saved_candidate_and_projects_silver_without_rewriting_it(self):
        candidate, = self.seed_annotations()

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
        self.seed_annotations()
        with self.transport(lambda _: {"decision": "PASS", "issue_codes": ["INSUFFICIENT_EVIDENCE"],
                                       "rationale": "Invalid PASS.", "evidence_ids": [],
                                       "suggested_annotation": None, "suggested_qa_revision": None}):
            result = self.command("review-labels", expected=1)
        self.assertEqual((result["failed"], result["results"][0]["category"], len(self.calls)), (1, "format", 2))
        self.assertFalse(any(row["record_type"] == "review" for row in self.export()))

    def test_reviews_two_candidates_concurrently(self):
        candidates = self.seed_annotations("image_text_consistency", "intent_analysis")
        overlap = Barrier(2, timeout=5)

        def reply(_):
            overlap.wait()
            return {"decision": "PASS", "issue_codes": [], "rationale": "Review fixture.",
                    "evidence_ids": [], "suggested_annotation": None, "suggested_qa_revision": None}

        with self.transport(reply):
            report = self.command("review-labels", workers=2)
        self.assertEqual((report["workers"], report["failed"], len(self.calls)), (2, 0, 2))
        reviews = [row for row in self.export() if row["record_type"] == "review"]
        self.assertEqual({(row["target_id"], row["target_content_hash"]) for row in reviews},
                         {(row["annotation_id"], row["content_hash"]) for row in candidates})


QA_BATCH = "questions-1"
PASS_REVIEW = {"decision": "PASS", "issue_codes": [], "rationale": "Review fixture.",
               "evidence_ids": [], "suggested_annotation": None, "suggested_qa_revision": None}


class QAFixture(LabelFixture):
    def seed_reviewed_labels(self, *tasks, decision="PASS"):
        annotations = self.seed_annotations(*tasks)
        results = []
        with self.state.db:
            for index, annotation in enumerate(annotations, 100):
                payload = PASS_REVIEW if decision == "PASS" else {
                    **PASS_REVIEW, "decision": decision, "issue_codes": ["EVIDENCE_MISSING"],
                }
                producer = {**annotation["producer"], "role": "reviewer", "model_id": "google/gemma-4-31B-it"}
                review = stage_record({"job_id": f"{index:064x}", "input_hash": f"{index:064x}", "payload": payload},
                                      self.sample, producer, [], kind="review", target=annotation)
                self.state.db.execute("INSERT INTO pilot_records VALUES (?, ?, ?)",
                                      (producer["run_id"], review["review_id"], json.dumps(review)))
                results.append({"record_id": review["review_id"], "decision": decision, "status": "COMPLETED"})
            self.state.db.execute("UPDATE pilot_batches SET review_report = ? WHERE batch_id = ?",
                                  (json.dumps({"failed": 0, "results": results}), BATCH))
        return annotations

    def qa_payload(self, task):
        return {
            "task_id": task, "question": "Which option fits the visible post?", "question_type": "single_choice",
            "options": [{"option_id": "one", "text": "The first interpretation", "label_codes": [TASK_LABELS[task]]},
                        {"option_id": "two", "text": "A different interpretation", "label_codes": []}],
            "answer_option_ids": ["one"], "target_labels": [TASK_LABELS[task]],
            "rationale": "Question fixture.", "evidence_ids": [], "required_inputs": ["text", "image"],
            "qa_mode": "content",
        }


class GenerateQATests(QAFixture, TestCase):
    def test_generates_resumable_candidates_concurrently_with_explicit_skip(self):
        self.seed_reviewed_labels(*TASK_LABELS)
        # The input label batch predates the QA implementation.
        with self.state.db:
            row = self.state.db.execute("SELECT manifest FROM pilot_batches").fetchone()
            source = json.loads(row["manifest"])
            source.update(code_revision="e" * 40, config_hash="3" * 64)
            self.state.db.execute("UPDATE pilot_batches SET manifest = ?", (json.dumps(source),))
        args = ("--limit", "4")
        with patch("rumorlab.pilot.connect_role") as connect:
            plan = self.command("generate-qa", batch=QA_BATCH, extra=(*args, "--dry-run"))
            connect.assert_not_called()
        self.assertEqual((plan["jobs"], plan["workers"]), (4, 48))
        overlap = Barrier(4, timeout=5)

        def reply(body):
            overlap.wait()
            schema = body["response_format"]["json_schema"]["schema"]
            task = schema["anyOf"][0]["properties"]["task_id"]["const"]
            payload = {"skip_reason": "no_supported_question"} if task == "intent_analysis" else self.qa_payload(task)
            Draft202012Validator(schema).validate(payload)
            return payload

        with self.transport(reply):
            generated = self.command("generate-qa", batch=QA_BATCH, extra=args)
            resumed = self.command("generate-qa", batch=QA_BATCH, extra=args)
        self.assertEqual((generated["failed"], len(self.calls)), (0, 4))
        self.assertEqual([item["status"] for item in generated["results"]], ["VALIDATED"] * 3 + ["SKIPPED"])
        self.assertTrue(all(item["cached"] for item in resumed["results"]))
        questions = [item for item in self.export() if item["record_type"] == "qa"]
        self.assertEqual(len(questions), 3)
        self.assertTrue(all(not item["is_mock"] and "answer_option_ids" not in item["solver_input"] for item in questions))
        source = next(item for item in self.export() if item["record_type"] == "review"
                      and item["target_id"] == questions[0]["source_annotation_id"])
        self.state.invalidate(source["review_id"], source["content_hash"])
        question = next(item for item in self.export() if item.get("qa_id") == questions[0]["qa_id"])
        self.assertEqual(question["status"], "STALE")

    def test_human_pending_label_never_reaches_the_model(self):
        self.seed_reviewed_labels(decision="NEEDS_HUMAN")
        with patch("rumorlab.pilot.connect_role") as connect:
            result = self.command("generate-qa", batch=QA_BATCH, extra=("--limit", "1"), expected=1)
            connect.assert_not_called()
        self.assertIn("No accepted model labels", result)


class ReviewQATests(QAFixture, TestCase):
    def test_reviews_saved_candidate_independently_and_binds_the_key(self):
        self.seed_reviewed_labels()
        annotation = next(item for item in self.export() if item["record_type"] == "annotation")
        task = annotation["payload"]["task_id"]
        row = self.state.db.execute("SELECT manifest FROM pilot_batches").fetchone()
        manifest = json.loads(row["manifest"])
        manifest.update(run_id="pilot_questions", prompt_versions={"review_qa": "1.1.2"})
        manifest["runtime"]["batch_id"] = QA_BATCH
        request = build_request(self.contract, "generate_qa", task, self.sample,
                                annotation["payload"]["target_claim"], [], self.root, upstream=annotation)
        payload = self.qa_payload(task)
        question = stage_record({"job_id": "4" * 64, "input_hash": "5" * 64, "payload": payload},
                                self.sample, {**annotation["producer"], "run_id": manifest["run_id"]}, [],
                                kind="qa", target=annotation, solver=solver_projection(request, self.sample))
        question["label_schema_version"] = self.contract.labels["label_schema_version"]
        with self.state.db:
            self.state.db.execute("INSERT INTO pilot_batches VALUES (?, ?, ?, ?, NULL, NULL)",
                                  (RUN, QA_BATCH, manifest["run_id"], json.dumps(manifest)))
            self.state.db.execute("INSERT INTO pilot_records VALUES (?, ?, ?)",
                                  (manifest["run_id"], question["qa_id"], json.dumps(question)))
            self.state.db.execute("INSERT INTO pilot_qa_reports VALUES (?, ?, ?)", (
                manifest["run_id"], "generate_qa", json.dumps({"failed": 0, "results": [
                    {"record_id": question["qa_id"], "status": "VALIDATED"},
                ]}),
            ))

        def reply(body):
            self.assertEqual(body["model"], "google/gemma-4-31B-it")
            self.assertIn("Candidate and proposed key:", json.dumps(body["messages"]))
            return PASS_REVIEW

        with self.transport(reply):
            report = self.command("review-qa", batch=QA_BATCH)
            resumed = self.command("review-qa", batch=QA_BATCH)
        self.assertEqual((report["failed"], report["workers"], len(self.calls)), (0, 48, 1))
        self.assertTrue(resumed["results"][0]["cached"])
        exported = self.export()
        approved = next(item for item in exported if item["record_type"] == "qa")
        review = next(item for item in exported if item.get("target_id") == approved["qa_id"])
        self.assertEqual(approved["status"], "ACCEPTED_SILVER")
        self.assertEqual(approved["payload"], payload)
        self.assertEqual(review["target_content_hash"], question["content_hash"])
