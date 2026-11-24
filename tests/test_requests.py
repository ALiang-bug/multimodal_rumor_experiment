import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from unittest import TestCase

import jsonschema

from rumorlab.config import Contract
from rumorlab.models import MockModel
from rumorlab.requests import build_request


ROOT = Path(__file__).resolve().parents[1]


class RequestTests(TestCase):
    def test_t06_request_keys_change_with_each_versioned_input(self):
        baseline = self.request("annotate").request_id
        config = self.contract.models["models"]["annotator"]
        for field, value in [("revision", "a" * 40), ("vision", {"max_pixels": 1234})]:
            with self.subTest(field=field):
                original = config[field]
                config[field] = value
                self.assertNotEqual(self.request("annotate").request_id, baseline)
                config[field] = original
        candidate = deepcopy(self.annotation)
        candidate["payload"]["rationale"] += " Synthetic revision."
        self.assertNotEqual(self.request("review_labels", candidate).request_id,
                            self.request("review_labels", self.annotation).request_id)
        self.evidence[0]["summary"] += " Synthetic evidence revision."
        self.assertNotEqual(self.request("annotate").request_id, baseline)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            copytree(ROOT / "prompts", root / "prompts")
            self.contract.root = root
            previous = self.request("annotate").request_id
            path = root / "prompts" / "annotate.j2"
            path.write_text(path.read_text(encoding="utf-8") + "\nSynthetic template revision.", encoding="utf-8")
            prompt_changed = self.request("annotate").request_id
            self.assertNotEqual(previous, prompt_changed)
            asset = self.sample["media"][0]
            raw = (ROOT / asset["path"]).read_bytes() + b"synthetic changed snapshot"
            (root / "updated.png").write_bytes(raw)
            asset.update(path="updated.png", sha256=sha256(raw).hexdigest())
            actual = build_request(self.contract, "annotate", self.annotation["payload"]["task_id"],
                                   self.sample, self.claim, self.evidence, root)
            self.assertNotEqual(actual.request_id, prompt_changed)
            self.assertEqual(actual.images[0], raw)

    def setUp(self):
        self.contract = Contract(ROOT / "configs/experiment.yaml")
        rows = [json.loads(line) for line in (ROOT / "examples/records.jsonl").read_text(encoding="utf-8").splitlines()]
        self.qa = next(row for row in rows if row["record_type"] == "qa")
        self.sample = next(row for row in rows if row.get("sample_id") == self.qa["sample_id"] and row["record_type"] == "sample")
        self.claim = self.sample["target_claims"][0]
        self.evidence = [row for row in rows if row["record_type"] == "evidence"]
        self.annotation = next(row for row in rows if row.get("annotation_id") == self.qa["source_annotation_id"])

    def request(self, stage, upstream=None):
        return build_request(
            self.contract, stage, self.annotation["payload"]["task_id"],
            self.sample, self.claim, self.evidence, ROOT, upstream=upstream,
        )

    def test_annotation_request_hides_internal_fields_and_specializes_schema(self):
        before = self.request("annotate")
        self.sample["source"]["source_veracity"] = "SECRET_VERACITY"
        self.sample["other_task_predictions"] = "SECRET_PREDICTIONS"
        self.sample["operation"] = {"target_error": "SECRET_OPERATION"}
        self.sample["answer_key"] = "SECRET_KEY"
        after = self.request("annotate")
        self.assertEqual(before, after)
        self.assertNotIn("blue_circle.png", str(after.messages))
        self.assertNotIn("SECRET_", str(after.messages))
        self.assertEqual(len(after.images), 1)
        self.assertNotIn('"$ref"', json.dumps(after.response_schema))
        validator = jsonschema.Draft202012Validator(after.response_schema)
        validator.validate(self.annotation["payload"])
        self.assertFalse(validator.is_valid({**self.annotation["payload"], "labels": ["OT"]}))

    def test_stage_exposure_and_qa_gate(self):
        reviewer = self.request("review_labels", self.annotation)
        self.assertIn(self.annotation["payload"]["rationale"], reviewer.messages[1][1])
        self.assertNotIn(self.annotation["annotation_id"], reviewer.messages[1][1])
        rejected = {**self.annotation, "status": "HUMAN_PENDING"}
        with self.assertRaisesRegex(ValueError, "accepted annotation"):
            self.request("generate_qa", rejected)
        self.assertEqual(self.request("generate_qa", self.annotation).role, "annotator")
        self.assertEqual(self.request("review_qa", self.qa).role, "reviewer")

    def test_label_stage_citations_use_only_visible_evidence_ids(self):
        annotation = self.request("annotate")
        visible, = annotation.visible_inputs["evidence"]
        evidence_id = visible["evidence_id"]
        payload = {**self.annotation["payload"], "evidence_ids": [evidence_id]}
        jsonschema.Draft202012Validator(annotation.response_schema).validate(payload)
        review = self.request("review_labels", self.annotation)
        validator = jsonschema.Draft202012Validator(review.response_schema)
        reply = {"decision": "NEEDS_HUMAN", "issue_codes": ["EVIDENCE_MISSING"], "rationale": "Inspect the image.",
                 "evidence_ids": [evidence_id], "suggested_annotation": payload, "suggested_qa_revision": None}
        validator.validate(reply)
        self.assertFalse(validator.is_valid({**reply, "evidence_ids": ["image_1"]}))
        self.assertFalse(validator.is_valid({**reply, "suggested_annotation": {**payload, "evidence_ids": ["image_1"]}}))

    def test_messages_identify_role_and_include_actual_output_schema(self):
        for stage, role, candidate in (
            ("annotate", "Annotator", None), ("review_labels", "Reviewer", self.annotation),
            ("generate_qa", "Annotator", self.annotation), ("review_qa", "Reviewer", self.qa),
        ):
            with self.subTest(stage=stage):
                request = self.request(stage, candidate)
                self.assertTrue(request.messages[0][1].startswith(f"You are the {role} "))
                rendered_schema = request.messages[1][1].rsplit("Response schema: ", 1)[1]
                self.assertEqual(json.loads(rendered_schema), request.response_schema)

    def test_t02_all_task_templates_exclude_hidden_sample_fields(self):
        sample = {**self.sample, "source": {"source_veracity": "PRIVATE_SOURCE"},
                  "other_task_predictions": "PRIVATE_PREDICTION",
                  "answer_key": "PRIVATE_ANSWER", "operation": {"target_error": "PRIVATE_OPERATION"}}
        requests = [build_request(self.contract, "annotate", task, sample, self.claim, self.evidence, ROOT)
                    for task in self.contract.experiment["annotation"]["tasks"]]
        self.assertEqual(len(requests), 6)
        self.assertEqual(len({request.request_id for request in requests}), 6)
        for request in requests:
            self.assertNotIn("PRIVATE_", str(request.messages))
            self.assertNotIn("blue_circle.png", str(request.messages))

    def test_mock_final_json_and_failures_are_distinct(self):
        request = self.request("annotate")
        for scripted in (self.annotation["payload"], "{broken", TimeoutError("offline timeout")):
            with self.subTest(scripted=scripted):
                response = MockModel({request.request_id: scripted}).generate(request)
                self.assertTrue(response.is_mock)
                self.assertEqual(response.schema_valid, isinstance(scripted, dict))
                self.assertEqual(bool(response.technical_error), isinstance(scripted, TimeoutError))
                if not response.schema_valid:
                    self.assertIsNone(response.parsed_payload)

    def test_english_task_guides_reach_annotator_and_candidate_aware_reviewer(self):
        caption = "The photo shows a red square."
        claim = dict(self.claim, text=caption, text_span={"start": 0, "end": len(caption)})
        sample = dict(self.sample, text=caption, target_claims=[claim])
        evidence = [dict(item, summary="The image shows a blue circle.") for item in self.evidence]
        for task, settings in self.contract.labels["tasks"].items():
            guide = (ROOT / "prompts" / "tasks" / f"{task}.txt").read_text(encoding="utf-8").strip()
            label = settings["labels"][0]["code"]
            payload = dict(self.annotation["payload"], task_id=task, target_claim=claim,
                           labels=[label], rationale="Synthetic example.")
            candidate = dict(self.annotation, payload=payload)
            for stage in ("annotate", "review_labels"):
                with self.subTest(task=task, stage=stage):
                    request = build_request(self.contract, stage, task, sample, claim, evidence, ROOT, upstream=candidate)
                    self.assertIn(guide, request.messages[1][1])
                    self.assertNotRegex(str(request.messages), r"[\u3400-\u9fff]")
        # Source wording and offsets are copied verbatim even when the instructions are English.
        self.assertIn(self.claim["text"], self.request("annotate").messages[1][1])
