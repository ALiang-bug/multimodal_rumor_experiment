from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from threading import Lock
from unittest import TestCase
from unittest.mock import patch

from rumorlab.cli import main
from rumorlab.validation import canonical_hash
from rumorlab.vllm import VllmModel
from test_pilot import CONFIG, RUN, PilotFixture


class BenchmarkTests(PilotFixture, TestCase):
    def test_cli_replays_identical_seeded_requests_without_workflow_writes(self):
        def reply(body):
            properties = body["response_format"]["json_schema"]["schema"]["properties"]
            sample = next(item for item in self.samples if item["text"] == properties["target_claim"]["const"]["text"])
            return self.payload(properties["task_id"]["const"], sample)

        args = ["rumorlab", "benchmark-vllm", "--config", str(CONFIG), "--run-id", RUN,
                "--state-db", str(self.db_path), "--model-lock", str(self.locks["annotator"]),
                "--samples", "2", "--levels", "1", "4", "--timeout-seconds", "360"]
        output = StringIO()
        with self.transport(reply), patch("sys.argv", args), redirect_stdout(output), redirect_stderr(StringIO()):
            self.assertEqual(main(), 0)

        report = json.loads(output.getvalue())
        self.assertEqual((report["timeout_seconds"], report["requests_per_level"], report["repeats"]), (360, 8, 1))
        self.assertEqual([(row["concurrency"], row["succeeded"]) for row in report["results"]], [(1, 8), (4, 8)])
        self.assertIn(report["best_concurrency"], (1, 4))
        warmup = report["warmup_requests"]
        self.assertEqual(len(self.calls), warmup + 16)
        first = self.calls[warmup:warmup + 8]
        second = self.calls[warmup + 8:]
        self.assertEqual(Counter(map(canonical_hash, first)), Counter(map(canonical_hash, second)))
        self.assertTrue(all(call["seed"] == report["seed"] for call in self.calls))
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM pilot_records").fetchone()[0], 0)

    def test_128_level_has_enough_replayed_requests_and_reports_timeout_identity(self):
        guard = Lock()
        calls = 0

        def request(client, path, body=None):
            nonlocal calls
            self.assertEqual(client.timeout, 360)
            if path == "/models":
                return {"data": [{"id": client.settings["model_id"]}]}
            with guard:
                calls += 1
                if calls == 5:
                    raise TimeoutError("fixture timeout")
            properties = body["response_format"]["json_schema"]["schema"]["properties"]
            sample = next(item for item in self.samples if item["text"] == properties["target_claim"]["const"]["text"])
            payload = self.payload(properties["task_id"]["const"], sample)
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}]}

        args = ["rumorlab", "benchmark-vllm", "--config", str(CONFIG), "--run-id", RUN,
                "--state-db", str(self.db_path), "--model-lock", str(self.locks["annotator"]),
                "--samples", "2", "--repeats", "16", "--levels", "128", "--timeout-seconds", "360"]
        output = StringIO()
        with patch.object(VllmModel, "_request", request), patch("sys.argv", args), redirect_stdout(output), redirect_stderr(StringIO()):
            self.assertEqual(main(), 1)
        report = json.loads(output.getvalue())
        row, = report["results"]
        self.assertEqual((calls, row["attempted"], row["succeeded"], report["best_concurrency"]), (132, 128, 127, None))
        failure, = row["failed_requests"]
        self.assertEqual(failure["error"], "TimeoutError")
        self.assertIn(failure["sample_id"], report["sample_ids"])
        self.assertTrue(failure["task_id"] and failure["request_id"])

    def test_sample_size_cannot_silently_shrink(self):
        args = ["rumorlab", "benchmark-vllm", "--config", str(CONFIG), "--run-id", RUN,
                "--state-db", str(self.db_path), "--model-lock", str(self.locks["annotator"]), "--samples", "16"]
        error = StringIO()
        with patch("sys.argv", args), redirect_stderr(error), patch("rumorlab.benchmark.connect_role") as connect:
            self.assertEqual(main(), 1)
            connect.assert_not_called()
        self.assertIn("--samples must be between 1 and 2", error.getvalue())
