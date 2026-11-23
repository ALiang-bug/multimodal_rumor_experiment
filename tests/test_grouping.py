from unittest import TestCase

from rumorlab.grouping import build_groups, plan_splits


def sample(sid, *, parents=(), digest=None, split="unassigned"):
    assets = []
    if digest is not None:
        assets.append({"media_id": f"asset_{sid}", "original_media_id": None,
                       "sha256": digest, "hash_status": "verified"})
    return {"record_type": "sample", "sample_id": sid, "family_id": sid,
            "leakage_group_id": sid, "media": assets, "parent_sample_ids": parents,
            "supersedes_id": None, "split": split}


class GroupingTests(TestCase):
    def test_t05_event_and_qa_reference_bind_the_same_component(self):
        rows = [sample("a", split="train"), sample("b"), sample("c")]
        rows.append({"record_type": "qa", "sample_id": "b",
                     "solver_input": {"reference_sample_ids": ["c"]}})
        links = [{"sample_ids": ["a", "b"], "reason": "same_event"}]
        report = build_groups(rows, links)
        self.assertEqual(list(report["groups"].values()), [["a", "b", "c"]])
        self.assertEqual({link["reason"] for link in report["links"]}, {"same_event", "qa_reference"})
        rows[2]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "Known split leakage"):
            build_groups(rows, links)

    def test_multilateral_leakage_and_seeded_plan(self):
        rows = [sample("a", digest="same", split="test"), sample("b", digest="same"),
                sample("c"), sample("d", parents=("b", "c")), sample("e"), sample("f")]
        links = [{"sample_ids": ["d", "e"], "reason": "confirmed_near_duplicate"}]
        policy = {"train": .7, "dev": .15, "test": .15, "seed": 314}
        report = build_groups(rows, links)
        self.assertIn(["a", "b", "c", "d", "e"], report["groups"].values())
        plan = plan_splits(report, policy)
        self.assertEqual(plan, plan_splits(build_groups(list(reversed(rows)), links), policy))
        self.assertEqual(plan["sample_counts"], {"train": 1, "dev": 0, "test": 5})
        rows[-2]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "Known split leakage"):
            build_groups(rows, links)
