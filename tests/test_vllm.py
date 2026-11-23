from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from urllib.error import URLError

from rumorlab.config import Contract
from rumorlab.requests import ModelRequest
from rumorlab.vllm import VllmModel


ROOT = Path(__file__).resolve().parents[1]


class VllmTests(TestCase):
    def setUp(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        settings = contract.models["models"]["annotator"]
        self.client = VllmModel(settings, "http://127.0.0.1:8000/v1", None, 1)
        assets = {"images": [{"mime_type": "image/png"}, {"mime_type": "image/png"}]}
        self.request = ModelRequest("annotator", "preflight", "probe", assets, "prompt",
                                    {"type": "object"}, "model", "request",
                                    (("user", "Fixed caption"),), (b"first", b"second"))

    def test_attaches_ordered_images_and_reads_only_final_content(self):
        reply = {"id": "test", "choices": [{"finish_reason": "stop", "message": {
            "reasoning": "must not be persisted", "content": '{"ok": true}',
        }}]}
        with patch.object(self.client, "_request", return_value=reply) as call:
            result, reasoning_seen = self.client.complete(self.request)
        body = call.call_args.args[1]
        contents = body["messages"][0]["content"]
        self.assertEqual([entry["image_url"]["url"].split(",")[1] for entry in contents[:2]],
                         ["Zmlyc3Q=", "c2Vjb25k"])
        self.assertEqual(contents[-1]["text"], "Fixed caption")
        self.assertEqual(body["response_format"]["json_schema"]["schema"], self.request.response_schema)
        self.assertTrue(reasoning_seen)
        self.assertTrue(result.schema_valid)
        self.assertEqual(result.parsed_payload, {"ok": True})
        self.assertNotIn("must not be persisted", repr(result))

    def test_timeout_truncation_empty_and_invalid_json_fail(self):
        with patch.object(self.client, "_request", side_effect=URLError("timeout")):
            self.assertTrue(self.client.generate(self.request).technical_error)
        for finish, content in (("length", "{}"), ("stop", ""), ("stop", "bad json")):
            with self.subTest(finish=finish, content=content):
                reply = {"choices": [{"finish_reason": finish, "message": {"content": content}}]}
                with patch.object(self.client, "_request", return_value=reply):
                    self.assertFalse(self.client.generate(self.request).schema_valid)
