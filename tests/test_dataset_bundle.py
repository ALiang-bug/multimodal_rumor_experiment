"""Targeted checks for the seven-label export and stratified TEST gate."""

from hashlib import sha256
import json
from unittest import TestCase

from rumorlab.dataset_bundle import join_samples, summarize
from rumorlab.mmfakebench import STRATIFIED_METHOD
from rumorlab.pilot import _inputs
from rumorlab.validation import canonical_hash


class DatasetBundleTests(TestCase):
    def test_seven_labels_keep_source_truth_and_unresolved_model_labels_separate(self):
        classes = ("original", "mismatch", "textual_veracity_distortion", "visual_veracity_distortion")
        source = [{"text": name, "fake_cls": name, "gt_answers": "True" if name == "original" else "Fake"}
                  for name in classes]
        samples = [{"record_type": "sample", "sample_id": name, "text": name,
                    "text_sha256": sha256(name.encode()).hexdigest(),
                    "media": [{"path": f"{name}.png", "sha256": name}]}
                   for name in classes]
        manifest = {"annotation_sha256": "source-file", "seed": 42, "selection": [
            {"sample_id": row["sample_id"], "source_row": index, "source_row_sha256": canonical_hash(raw)}
            for index, (row, raw) in enumerate(zip(samples, source), 1)
        ]}
        annotations = []
        for sample, decision in zip(samples, ("PASS", "NEEDS_HUMAN", "REJECT")):
            key = sample["sample_id"]
            annotation = {"record_type": "annotation", "sample_id": key, "annotation_id": key,
                          "status": "ACCEPTED_SILVER" if decision == "PASS" else "HUMAN_PENDING",
                          "payload": {"task_id": "image_text_consistency", "labels": ["C1"],
                                      "rationale": "Visible match.", "evidence_ids": []},
                          "producer": {"prompt_version": "1.3.0"}, "is_mock": False, "content_hash": key}
            review = {"record_type": "review", "target_id": key, "target_content_hash": key,
                      "review_id": key, "status": "COMPLETED", "payload": {"decision": decision},
                      "producer": {"prompt_version": "1.3.0"}, "is_mock": False}
            annotations.extend((annotation, review))
        rows = join_samples(samples + annotations, manifest, source, (set(), {"mismatch"}))
        self.assertEqual([len(row["labels"]) for row in rows], [7] * 4)
        self.assertEqual([row["sample_id"] for row in rows], sorted(classes, key=lambda key: canonical_hash([42, key])))
        rows_by_id = {row["sample_id"]: row for row in rows}
        rows = [rows_by_id[key] for key in classes]
        self.assertEqual([row["labels"]["textual_veracity_distortion"] for row in rows], ["OT", "OT", "FT", "OT"])
        self.assertEqual([row["labels"]["visual_veracity_distortion"] for row in rows], ["OI", "OI", None, "FI"])
        self.assertEqual([row["labels"]["source_veracity"] for row in rows], ["real", "fake", "fake", "fake"])
        self.assertEqual([row["labels"]["image_text_consistency"] for row in rows], ["C1", None, None, None])
        self.assertEqual(rows[1]["model_annotations"]["image_text_consistency"]["candidate_label"], "C1")
        self.assertEqual(rows[1]["usable_labels"], [])
        stats = summarize(rows)
        self.assertEqual(stats["review_decisions"]["image_text_consistency"],
                         {"PASS": 1, "NEEDS_HUMAN": 1, "REJECT": 1, "ANNOTATION_MISSING": 1})
        self.assertEqual(stats["excluded_for_exact_val_overlap"], 1)

    def test_label_stages_accept_frozen_nonprefix_test_only_when_allowed(self):
        class Database:
            def execute(self, _query, _args):
                return self

            def fetchone(self):
                return {"manifest": json.dumps(manifest), "records": json.dumps(records)}

        manifest = {"adapter": "mmfakebench", "source_split": "MMFakeBench_test", "assigned_split": "test",
                    "method": STRATIFIED_METHOD, "input_rows": 10, "sample_size": 2,
                    "selection": [{"sample_id": "a", "source_row": 8}, {"sample_id": "b", "source_row": 2}]}
        records = [{"record_type": "sample", "split": "test", "is_mock": False, "sample_id": key}
                   for key in ("a", "b")]
        db = Database()
        self.assertEqual(set(_inputs(db, "test-run", allow_test=True)[3]), {"a", "b"})
        with self.assertRaisesRegex(ValueError, "Pilot accepts only"):
            _inputs(db, "test-run")
        manifest["selection"][1]["source_row"] = 8
        with self.assertRaisesRegex(ValueError, "frozen import disagree"):
            _inputs(db, "test-run", allow_test=True)
