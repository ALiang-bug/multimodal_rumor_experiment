from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from rumorlab.config import Contract
from rumorlab.ingest import ingest
from rumorlab.snapshot import snapshot_records
from rumorlab.state import State


ROOT = Path(__file__).resolve().parents[1]


class RawImportTests(TestCase):
    def test_generic_jsonl_decodes_media_and_preserves_source_provenance(self):
        contract = Contract(ROOT / "configs/experiment.yaml")
        contract.experiment["source"].update(
            dataset_name="synthetic", dataset_version="test",
            license_reference="Unit test fixture", source_veracity_mapping={"false": "false"},
        )
        row = {
            "source_id": "synthetic", "text": "蓝色圆形 🌐", "source_veracity": "false",
            "images": ["examples/images/blue_circle.png"], "split": "dev",
            "target_claims": [{"claim_id": "synthetic_claim", "text": "🌐", "text_span": {"start": 5, "end": 6}}],
        }
        with TemporaryDirectory() as directory:
            state = State(Path(directory) / "state.db")
            try:
                self.assertEqual(ingest(state, [(1, row)], contract, ROOT, "raw", fixture=False), (1, True))
                self.assertEqual(ingest(state, [(1, row)], contract, ROOT, "raw", fixture=False), (1, False))
                record, = snapshot_records(state, "raw")
                self.assertTrue(record["is_mock"])
                self.assertEqual(record["source"]["source_veracity"], "false")
                self.assertEqual((record["text_provenance"], record["image_provenance"]), ("OT", "OI"))
                self.assertEqual((record["media"][0]["width"], record["media"][0]["height"]), (256, 256))
                invalid = {**row, "images": [str(ROOT / row["images"][0])]}
                with self.assertRaisesRegex(ValueError, "relative to the media root"):
                    ingest(state, [(1, invalid)], contract, ROOT, "invalid", fixture=False)
                self.assertEqual(state.db.execute("SELECT count(*) FROM imports").fetchone()[0], 1)
            finally:
                state.close()
