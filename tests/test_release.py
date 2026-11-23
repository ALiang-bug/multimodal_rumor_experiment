import json
from pathlib import Path
from unittest import TestCase

from rumorlab.config import Contract
from rumorlab.qa import project_options, score_options, score_snapshot
from rumorlab.release import check_release
from rumorlab.validation import canonical_hash


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(TestCase):
    def test_t12_release_gate_accepts_complete_train_snapshot_and_rejects_regressions(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        # In-memory simulation of production-shaped inputs, never written as research records.
        records = [json.loads(line) for line in (ROOT / "examples/records.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in records:
            row.update(is_mock=False, mock_note=None, input_hash=canonical_hash(row))
            if row["record_type"] == "run_manifest":
                row.update(mode="real", code_revision="unit-test", status="COMPLETED")
                for lock in row["models"].values():
                    lock.update(backend="unit-test", revision="unit-test", backend_version="unit-test",
                                processor_revision="unit-test", chat_template_hash="a" * 64, precision="fp32")
            if row["record_type"] == "sample":
                row["split"] = "train"
                row["source"]["redistribution_status"] = "permitted"
        question = next(row for row in records if row["record_type"] == "qa")
        annotation_review = next(row for row in records if row["record_type"] == "review")
        qa_review = {
            **annotation_review, "review_id": "test_qa_review", "stage": "review_qa",
            "target_record_type": "qa", "target_id": question["qa_id"],
            "target_content_hash": question["content_hash"],
        }
        records.append(qa_review)
        question.update(status="ACCEPTED_SILVER", quality_tier="silver", review_ids=[qa_review["review_id"]])
        rows = list(enumerate(records, 1))
        self.assertEqual(check_release(rows, contract, ROOT), [])
        sample = next(row for row in records if row["record_type"] == "sample")
        sample["source"]["redistribution_status"] = "restricted"
        self.assertTrue(any("redistribution permission" in error for error in check_release(rows, contract, ROOT)))
        sample["source"]["redistribution_status"] = "permitted"
        sample["split"] = "test"
        self.assertTrue(any("cross leakage group or split" in error for error in check_release(rows, contract, ROOT)))
        sample["split"] = "train"
        question["payload"]["answer_option_ids"] = ["absent"]
        question["content_hash"] = qa_review["target_content_hash"] = canonical_hash(question["payload"])
        self.assertTrue(any("nonexistent option" in error for error in check_release(rows, contract, ROOT)))

    def test_t12_qa_denominator_includes_every_failure(self):
        rows = [json.loads(line) for line in (ROOT / "examples/records.jsonl").read_text(encoding="utf-8").splitlines()]
        qa = next(row for row in rows if row["record_type"] == "qa")
        states = ["correct", "incorrect", "invalid", "timeout", "refused", "technical_error", "missing"]
        questions = [{**qa, "qa_id": name} for name in states]
        predictions = [{"qa_id": name, "status": name} for name in ("timeout", "refused", "technical_error")]
        for name in ("correct", "incorrect", "invalid"):
            _, mapping = project_options(qa["payload"], seed=42, qa_id=name)
            correct = qa["payload"]["answer_option_ids"]
            letters = [letter for letter, oid in mapping.items() if (oid in correct) == (name == "correct")]
            predictions.append({"qa_id": name, "status": "answered", "letters": letters[:1] if name != "invalid" else ["Z"]})
        with self.assertRaisesRegex(ValueError, "No eligible QA"):
            score_snapshot(questions, [], seed=42)
        result = score_snapshot(questions, predictions, seed=42, mock=True)
        self.assertEqual(result["counts"], dict.fromkeys(states, 1))
        self.assertEqual(result["total"], 7)
        self.assertEqual(result["accuracy"], 1 / 7)

    def test_fixture_cannot_pass_formal_release(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        rows = [(n, json.loads(line)) for n, line in enumerate(
            (ROOT / "examples/records.jsonl").read_text(encoding="utf-8").splitlines(), 1
        )]
        errors = check_release(rows, contract, ROOT)
        self.assertEqual(sum("mock record" in error for error in errors), len(rows))

    def test_t10_t12_test_requires_gold_and_licensed_media(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        rows = [(n, json.loads(line)) for n, line in enumerate(
            (ROOT / "examples/records.jsonl").read_text(encoding="utf-8").splitlines(), 1
        )]
        for _, record in rows:
            if record["record_type"] == "sample":
                record["split"] = "test"
        errors = check_release(rows, contract, ROOT)
        self.assertTrue(any("test release cannot use ACCEPTED_SILVER/silver" in error for error in errors))
        self.assertTrue(any("test release cannot use GENERATED/unresolved" in error for error in errors))
        self.assertEqual(sum("redistribution permission" in error for error in errors), 2)

    def test_shuffle_keeps_single_and_multiple_answers_stable(self):
        options = [{"option_id": key, "text": key.upper()} for key in ("first", "second", "third")]
        for shape, answers in [("single_choice", ["first"]), ("multiple_choice", ["first", "third"])]:
            payload = {"question_type": shape, "options": options, "answer_option_ids": answers}
            display, mapping = project_options(payload, seed=42, qa_id="qa")
            letters = [letter for letter, key in mapping.items() if key in answers]
            self.assertTrue(score_options(list(reversed(letters)), mapping, answers))
            self.assertFalse(score_options(letters + ["Z"], mapping, answers))
            self.assertTrue(all(set(option) == {"letter", "text"} for option in display))
            with self.assertRaisesRegex(ValueError, "Unknown"):
                project_options({**payload, "answer_option_ids": ["missing"]}, seed=42, qa_id="qa")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                project_options({**payload, "options": options + [options[0]]}, seed=42, qa_id="qa")
            with self.assertRaisesRegex(ValueError, "answer marker"):
                project_options({**payload, "question": "Synthetic question。正确答案：A"}, seed=42, qa_id="qa")
