"""MMFakeBench-shaped synthetic rows and ingest-only checks."""

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from rumorlab.cli import main
from rumorlab.mmfakebench import prepare_mmfakebench
from rumorlab.requests import visible_material
from rumorlab.state import State
from rumorlab.validation import canonical_hash


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261122


class MMFakeBenchTests(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "MMFakeBench_val"
        self.path = self.source / "source" / "MMFakeBench_val.json"
        self.path.parent.mkdir(parents=True)
        self.rows = []
        for label, fake_class, count in (("True", "original", 3), ("Fake", "mismatch", 2),
                                         ("Fake", "visual_veracity_distortion", 1)):
            for index in range(count):
                folder = "real" if label == "True" else "fake"
                relative = f"{folder}/{fake_class}_{index}.png"
                image = self.source / relative
                image.parent.mkdir(exist_ok=True)
                shutil.copyfile(ROOT / "examples/images/blue_circle.png", image)
                self.rows.append({"text": f"Synthetic {fake_class} {index} 🌐", "image_path": "/" + relative,
                                  "gt_answers": label, "fake_cls": fake_class})

    def write(self, rows):
        self.path.write_text(json.dumps(rows), encoding="utf-8")

    def test_ranked_strata_and_stable_source_ids(self):
        self.write(self.rows + self.rows[:1])
        records, manifest = prepare_mmfakebench(self.path, self.root, SEED, 4)
        self.assertEqual(len(records), 4)
        self.assertEqual(manifest["duplicate_source_rows"], 1)
        self.assertEqual({s["fake_cls"]: s["selected"] for s in manifest["strata"]}, {
            "mismatch": 1, "visual_veracity_distortion": 1, "original": 2,
        })
        self.assertEqual(manifest["annotation_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        for _, record in records:
            self.assertEqual((record["split"], record["text_provenance"], record["image_provenance"]), ("dev", "OT", "OI"))
            self.assertEqual(record["source"]["redistribution_status"], "restricted")
            self.assertIn(record["source"]["source_veracity"], {"True", "Fake"})
            self.assertEqual(record["target_claims"][0]["text_span"]["end"], len(record["text"]))
            self.assertIs(record["is_mock"], False)
            material, _ = visible_material(record, [], self.root, solver=False)
            self.assertEqual(set(material), {"text", "images", "evidence"})
            self.assertEqual(set(material["images"][0]), {"id", "mime_type", "sha256"})
        self.write(list(reversed(self.rows)))
        _, reversed_manifest = prepare_mmfakebench(self.path, self.root, SEED, 4)
        self.assertEqual([row["source_id"] for row in manifest["selection"]],
                         [row["source_id"] for row in reversed_manifest["selection"]])

    def test_ingest_rerun_with_corrupt_media_and_atomic_manifest(self):
        def rank(row):
            relative = self.source.name + row["image_path"]
            source_id = self.source.name.lower() + "_" + canonical_hash([relative, row["text"]])
            return canonical_hash([SEED, source_id])

        damaged = min(self.rows[:3], key=rank)
        self.write(self.rows)
        (self.source / damaged["image_path"].lstrip("/")).write_bytes(b"synthetic invalid image")
        db_path = self.root / "test.db"
        command = ["rumorlab", "ingest", "--adapter", "mmfakebench", "--run-id", "synthetic",
                   "--data-root", str(self.root), "--config", str(ROOT / "configs/experiment.yaml"),
                   "--state-db", str(db_path), "--sample-size", "4"]
        for inserted in (True, False):
            stdout = StringIO()
            with patch("sys.argv", command), redirect_stdout(stdout):
                self.assertEqual(main(), 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual((report["records"], report["inserted"]), (4, inserted))
            self.assertEqual(report["sampling"]["quarantined_rows"], 1)
        state = State(db_path)
        self.addCleanup(state.close)
        manifest, = state.db.execute("SELECT manifest FROM import_manifests WHERE run_id = 'synthetic'").fetchone()
        saved = json.loads(manifest)
        self.assertEqual(len(saved["selection"]), 4)
        self.assertEqual(saved["quarantine"][0]["status"], "QUARANTINED")

    def test_default_cli_imports_entire_val_split(self):
        self.write(self.rows)
        (self.root / "MMFakeBench_test.json").write_text("unread test pool", encoding="utf-8")
        command = ["rumorlab", "ingest", "--adapter", "mmfakebench", "--run-id", "synthetic-val",
                   "--data-root", str(self.root), "--config", str(ROOT / "configs/experiment.yaml"),
                   "--state-db", str(self.root / "test.db")]
        stdout = StringIO()
        with patch("sys.argv", command), redirect_stdout(stdout):
            self.assertEqual(main(), 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["records"], len(self.rows))
        self.assertEqual(report["sampling"]["source_split"], "MMFakeBench_val")
        self.assertEqual(report["sampling"]["assigned_split"], "dev")

    def test_rejects_path_traversal_and_short_population(self):
        self.write(self.rows)
        with self.assertRaisesRegex(ValueError, "Requested 2000 .* found 6"):
            prepare_mmfakebench(self.path, self.root, SEED, 2_000)
        self.write([{**self.rows[0], "image_path": "/real/../escape.png"}])
        with self.assertRaisesRegex(ValueError, "image_path must stay inside"):
            prepare_mmfakebench(self.path, self.root, SEED, 1)

    def test_cli_freezes_exact_test_prefix_in_source_order(self):
        split = self.root / "MMFakeBench_test"
        shutil.copytree(self.source, split)
        path = split / "MMFakeBench_test.json"
        ordered = [self.rows[index] for index in (4, 1, 3, 0, 2, 5)]
        path.write_text(json.dumps(ordered), encoding="utf-8")
        (split / ordered[4]["image_path"].lstrip("/")).unlink()
        db_path = self.root / "prefix.db"
        args = ["rumorlab", "ingest", "--adapter", "mmfakebench", "--run-id", "test-prefix",
                "--first-n", "3", "--records", str(path), "--data-root", str(self.root),
                "--config", str(ROOT / "configs/experiment.yaml"), "--state-db", str(db_path)]
        with patch("sys.argv", args), redirect_stdout(StringIO()):
            self.assertEqual(main(), 0)
        state = State(db_path)
        self.addCleanup(state.close)
        records, manifest = map(json.loads, state.db.execute(
            "SELECT records, manifest FROM imports JOIN import_manifests USING(run_id)"
        ).fetchone())
        self.assertEqual(manifest["method"], "source-prefix-v1")
        self.assertEqual([item["source_row"] for item in manifest["selection"]], [1, 2, 3])
        self.assertEqual([row["text"] for row in records], [row["text"] for row in ordered[:3]])
        self.assertEqual({row["split"] for row in records}, {"test"})
        with self.assertRaisesRegex(ValueError, "Cannot freeze MMFakeBench test row 5"):
            prepare_mmfakebench(path, self.root, SEED, first_n=5)
