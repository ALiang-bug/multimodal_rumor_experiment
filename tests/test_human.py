from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from rumorlab.cli import read_jsonl
from rumorlab.config import Contract
from rumorlab.human import blind_review_tasks, import_human
from rumorlab.ingest import ingest
from rumorlab.records import mock_envelope
from rumorlab.qa import project_options
from rumorlab.snapshot import snapshot_records
from rumorlab.state import State
from rumorlab.validation import validate_rows
from rumorlab.quality import project_gold


ROOT = Path(__file__).resolve().parents[1]


class HumanTests(TestCase):
    def test_t08_blind_letters_round_trip_to_stable_option_ids(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        rows = list(read_jsonl(ROOT / "examples/records.jsonl"))
        qa = next(row for _, row in rows if row["record_type"] == "qa")
        with TemporaryDirectory() as directory:
            state = State(Path(directory) / "state.db")
            try:
                ingest(state, rows, contract, ROOT, "fixture", fixture=True)
                assignment, = blind_review_tasks(state, contract, "fixture", [qa["qa_id"]], "synthetic")
                _, mapping = project_options(qa["payload"], seed=contract.experiment["seed"], qa_id=qa["qa_id"])
                selected = [letter for letter, oid in mapping.items() if oid in qa["payload"]["answer_option_ids"]]
                data = {
                    "human_vote_id": "mock_qa_letters", "target_record_type": "qa",
                    "target_id": assignment["target_id"], "target_content_hash": assignment["target_content_hash"],
                    "annotator_id": "mock_annotator", "guideline_version": "synthetic", "vote_mode": "independent",
                    "blinded_to_model": True, "blinded_to_other_humans": True, "annotation_answer": None,
                    "qa_answer": {"letters": selected, "decision": "ACCEPTABLE",
                                  "rationale": "Synthetic selection.", "evidence_ids": []},
                    "audit_selection_id": None, "elapsed_seconds": 0, "status": "SUBMITTED",
                }
                submission = {**mock_envelope("human_vote", data, qa["content_hash"]), **data}
                self.assertEqual(import_human(state, contract, "fixture", [(1, submission)]), 1)
                self.assertEqual(import_human(state, contract, "fixture", [(1, submission)]), 0)
                stored = next(row for row in snapshot_records(state, "fixture") if row.get("human_vote_id") == data["human_vote_id"])
            finally:
                state.close()
        self.assertEqual(stored["target_id"], qa["qa_id"])
        self.assertEqual(stored["qa_answer"]["answer_option_ids"], qa["payload"]["answer_option_ids"])

    def test_blind_export_separates_material_from_candidate_and_answer(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        rows = list(read_jsonl(ROOT / "examples/records.jsonl"))
        annotation = next(row for _, row in rows if row["record_type"] == "annotation")
        qa = next(row for _, row in rows if row["record_type"] == "qa")
        with TemporaryDirectory() as directory:
            state = State(Path(directory) / "state.db")
            try:
                ingest(state, rows, contract, ROOT, "fixture", fixture=True)
                labels, question = blind_review_tasks(
                    state, contract, "fixture", [annotation["annotation_id"], qa["qa_id"]], "test-guideline",
                )
            finally:
                state.close()
        self.assertNotIn(annotation["payload"]["rationale"], repr(labels))
        self.assertNotIn("source_veracity", repr(labels))
        self.assertNotIn(".png", repr(labels))
        self.assertEqual(question["question"], qa["payload"]["question"])
        self.assertEqual(question["target_content_hash"], qa["content_hash"])
        self.assertTrue(all(set(x) == {"letter", "text"} for x in question["options"]))
        self.assertNotIn("answer_option_ids", repr(question))

    def test_gold_confirmation_requires_three_blind_votes_and_does_not_promote_qa(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        rows = list(read_jsonl(ROOT / "examples/records.jsonl"))
        target = next(row for _, row in rows if row["record_type"] == "annotation")
        votes = []
        for index in range(3):
            vote = {
                "human_vote_id": f"mock_vote_{index}", "target_record_type": "annotation",
                "target_id": target["annotation_id"], "target_content_hash": target["content_hash"],
                "annotator_id": f"mock_annotator_{index}", "guideline_version": "synthetic",
                "blinded_to_model": True, "blinded_to_other_humans": True, "vote_mode": "independent",
                "annotation_answer": target["payload"], "qa_answer": None, "audit_selection_id": None,
                "elapsed_seconds": 0, "status": "SUBMITTED",
            }
            votes.append({**mock_envelope("human_vote", vote, target["content_hash"]), **vote})
        rulings = []
        for count in (2, 3):
            ruling = {
                "adjudication_id": f"mock_ruling_{count}", "target_record_type": "annotation",
                "target_id": target["annotation_id"], "target_content_hash": target["content_hash"],
                "human_vote_ids": [vote["human_vote_id"] for vote in votes[:count]],
                "adjudicator_id": "mock_adjudicator", "guideline_version": "synthetic",
                "outcome": "CONFIRMED", "final_annotation_payload": target["payload"],
                "final_qa_payload": None, "rationale": "Synthetic workflow test.",
                "evidence_ids": [], "eligible_for_gold": True, "status": "FINALIZED",
            }
            rulings.append({**mock_envelope("adjudication", ruling, target["content_hash"]), **ruling})
        with TemporaryDirectory() as directory:
            state = State(Path(directory) / "state.db")
            try:
                ingest(state, rows, contract, ROOT, "fixture", fixture=True)
                import_human(state, contract, "fixture", list(enumerate([*votes[:2], rulings[0]], 1)))
                initial = snapshot_records(state, "fixture")
                self.assertEqual(next(row for row in initial if row["record_type"] == "annotation")["quality_tier"], "silver")
                import_human(state, contract, "fixture", [(1, votes[2]), (2, rulings[1])])
                promoted = snapshot_records(state, "fixture")
                correction = {
                    **rulings[1], "adjudication_id": "mock_correction", "outcome": "CORRECTED",
                    "final_annotation_payload": {
                        **target["payload"], "rationale": "Synthetic corrected rationale.",
                    },
                }
                self.assertEqual(import_human(state, contract, "fixture", [(1, correction)]), 1)
                self.assertEqual(import_human(state, contract, "fixture", [(1, correction)]), 0)
                revised = snapshot_records(state, "fixture")
            finally:
                state.close()
        annotation = next(row for row in promoted if row["record_type"] == "annotation")
        qa = next(row for row in promoted if row["record_type"] == "qa")
        self.assertEqual(annotation["status"], "ACCEPTED_GOLD")
        self.assertEqual(annotation["content_hash"], target["content_hash"])
        self.assertEqual(qa["quality_tier"], "unresolved")
        other_task = {
            **annotation, "annotation_id": "mock_other_task", "status": "VALIDATED",
            "quality_tier": "unresolved", "human_vote_ids": [], "adjudication_id": None,
            "payload": {**annotation["payload"], "task_id": "reality_plausibility", "labels": ["EF"]},
        }
        project_gold([*promoted, other_task], contract.gold_votes)
        self.assertEqual(other_task["quality_tier"], "unresolved")
        errors, _ = validate_rows(list(enumerate(promoted, 1)), contract.schema, contract.labels, contract.gold_votes, ROOT)
        self.assertEqual(errors, [])
        replacement = next(row for row in revised if row["supersedes_id"] == target["annotation_id"])
        self.assertEqual(replacement["status"], "ACCEPTED_GOLD")
        self.assertNotEqual(replacement["content_hash"], target["content_hash"])
        self.assertEqual(replacement["record_version"], target["record_version"] + 1)
        self.assertEqual(next(row for row in revised if row["record_type"] == "qa")["status"], "STALE")

    def test_idempotency_version_and_atomic_batch(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        rows = list(read_jsonl(ROOT / "examples/records.jsonl"))
        target = next(row for _, row in rows if row["record_type"] == "annotation")
        data = {
            "human_vote_id": "test_vote", "target_record_type": "annotation",
            "target_id": target["annotation_id"], "target_content_hash": target["content_hash"],
            "annotator_id": "test_only_annotator", "guideline_version": "synthetic",
            "blinded_to_model": True, "blinded_to_other_humans": True,
            "vote_mode": "independent", "annotation_answer": target["payload"], "qa_answer": None,
            "audit_selection_id": None, "elapsed_seconds": 0, "status": "SUBMITTED",
        }
        vote = {**mock_envelope("human_vote", data, target["content_hash"]), **data}
        with TemporaryDirectory() as directory:
            state = State(Path(directory) / "state.db")
            try:
                ingest(state, rows, contract, ROOT, "fixture", fixture=True)
                bad = {**vote, "human_vote_id": "bad", "target_content_hash": "f" * 64}
                with self.assertRaisesRegex(ValueError, "stale or invalid target"):
                    import_human(state, contract, "fixture", [(1, vote), (2, bad)])
                self.assertEqual(import_human(state, contract, "fixture", [(1, vote)]), 1)
                self.assertEqual(import_human(state, contract, "fixture", [(1, vote)]), 0)
                with self.assertRaisesRegex(ValueError, "duplicate vote"):
                    import_human(state, contract, "fixture", [(1, {**vote, "human_vote_id": "duplicate"})])
                exported = snapshot_records(state, "fixture")
                self.assertIn(vote, exported)
                self.assertEqual(next(row for row in exported if row.get("annotation_id") == target["annotation_id"])["quality_tier"], "silver")
            finally:
                state.close()
