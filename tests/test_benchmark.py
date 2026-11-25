from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from unittest import TestCase
from unittest.mock import patch

from rumorlab.cli import main
from rumorlab.validation import canonical_hash
from test_pilot import CONFIG, RUN, PilotFixture


class BenchmarkTests(PilotFixture, TestCase):
    def test_cli_replays_identical_seeded_requests_without_workflow_writes(self):
        def reply(body):
            properties = body["response_format"]["json_schema"]["schema"]["properties"]
            sample = next(item for item in self.samples if item["text"] == properties["target_claim"]["const"]["text"])
            return self.payload(properties["task_id"]["const"], sample)

        args = ["rumorlab", "benchmark-vllm", "--config", str(CONFIG), "--run-id", RUN,
                "--state-db", str(self.db_path), "--model-lock", str(self.locks["annotator"]), "--levels", "1", "4"]
        output = StringIO()
        with self.transport(reply), patch("sys.argv", args), redirect_stdout(output), redirect_stderr(StringIO()):
            self.assertEqual(main(), 0)

        report = json.loads(output.getvalue())
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
