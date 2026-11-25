from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest import TestCase
from unittest.mock import patch

from rumorlab.state import State


class StateTests(TestCase):
    def test_heartbeat_keeps_slow_job_owned_past_initial_expiry(self):
        with TemporaryDirectory() as directory:
            state = State(Path(directory) / "state.sqlite3")
            try:
                job, lease, _ = state.claim("annotate", "sample", "intent_analysis", "a", [], .6)
                with state.keep_lease(job, lease, .6, .05):
                    Event().wait(.75)
                    state.finish(job, lease, result={"content_hash": "finished"})
                cached = state.claim("annotate", "sample", "intent_analysis", "a", [], .6)
                self.assertEqual(cached, (job, None, {"content_hash": "finished"}))
            finally:
                state.close()

    def test_invalidated_version_revokes_transitive_leases_and_cache(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            state = State(path)
            try:
                old = [("sample", "v1")]
                annotation, lease, _ = state.claim("annotate", "sample", "intent_analysis", "a", old, 10)
                state.finish(annotation, lease, result={"content_hash": "annotation-v1"})
                qa_deps = [(annotation, "annotation-v1")]
                qa, qa_lease, _ = state.claim("generate_qa", "sample", "intent_analysis", "b", qa_deps, 10)
                self.assertEqual(set(state.invalidate("sample", "v1")), {annotation, qa})
                self.assertEqual(state.invalidate("sample", "v1"), [])
                with self.assertRaisesRegex(ValueError, "lease lost"):
                    state.finish(qa, qa_lease, result={"content_hash": "qa-v1"})
            finally:
                state.close()
            reopened = State(path)
            try:
                with self.assertRaisesRegex(ValueError, "Stale dependency"):
                    reopened.claim("generate_qa", "sample", "intent_analysis", "b", qa_deps, 10)
                new, _, cached = reopened.claim("annotate", "sample", "intent_analysis", "a", [("sample", "v2")], 10)
                self.assertNotEqual(new, annotation)
                self.assertIsNone(cached)
                result = reopened.db.execute("SELECT result FROM jobs WHERE id = ?", (annotation,)).fetchone()[0]
                self.assertIn("annotation-v1", result)
            finally:
                reopened.close()

    def test_expired_lease_and_committed_cache(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            state = State(path)
            self.addCleanup(state.close)
            request = ("annotate", "sample", "intent_analysis", "hash", [("sample", "v1")], 10)
            with patch("rumorlab.state.time.time", return_value=100):
                job_id, old_lease, _ = state.claim(*request)
                with self.assertRaisesRegex(ValueError, "already leased"):
                    state.claim(*request)
            with patch("rumorlab.state.time.time", return_value=111):
                recovered_id, lease, _ = state.claim(*request)
                self.assertEqual(job_id, recovered_id)
                with self.assertRaisesRegex(ValueError, "lease lost"):
                    state.finish(job_id, old_lease, result={"obsolete": True})
                state.finish(job_id, lease, result={"is_mock": True})
            reopened = State(path)
            try:
                self.assertEqual(reopened.claim(*request), (job_id, None, {"is_mock": True}))
                states = [row["state"] for row in reopened.db.execute("SELECT state FROM events ORDER BY sequence")]
                self.assertEqual(states, ["QUEUED", "RUNNING", "RETRYABLE_ERROR", "RUNNING", "SUCCEEDED"])
            finally:
                reopened.close()
                state.close()
