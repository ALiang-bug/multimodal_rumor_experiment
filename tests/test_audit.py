from unittest import TestCase

from rumorlab.audit import draw_audit, reviewer_metrics


class AuditTests(TestCase):
    def test_weighted_detection_and_pass_residual_have_distinct_denominators(self):
        policy = {"strata": ["reviewer_decision"], "auto_pass_sampling_fraction_initial": .5,
                  "routed_sampling_fraction_initial": 1, "reviewer_alert_decisions": ["NEEDS_HUMAN", "REJECT"]}
        population = [{"record_id": str(i), "reviewer_decision": "PASS" if i < 4 else "NEEDS_HUMAN",
                       "routed_to_human": i == 4} for i in range(6)]
        selected = draw_audit(population, policy, seed=1)
        self.assertEqual(sorted(row["sampling_probability"] for row in selected), [.5, .5, 1, 1])
        passed = next(row["record_id"] for row in selected if row["reviewer_decision"] == "PASS")
        judgments = [{"record_id": row["record_id"], "annotation_error": row["record_id"] in {passed, "4"},
                      "needs_intervention": row["record_id"] in {"4", "5"}} for row in selected]
        result = reviewer_metrics(population, selected, judgments, policy)
        self.assertEqual(result["annotation_error"]["weighted_confusion"], {"tp": 1, "fp": 1, "fn": 2, "tn": 2})
        self.assertEqual(result["annotation_error"]["recall"], 1 / 3)
        self.assertEqual(result["needs_intervention"]["recall"], 1)
        self.assertEqual(result["pass_residual_error"], .5)
        self.assertEqual(result["routed_fraction"], 1 / 6)
        flagged = [row for row in selected if row["reviewer_decision"] != "PASS"]
        restricted = reviewer_metrics(population, flagged, [row for row in judgments if row["record_id"] in {"4", "5"}], policy)
        self.assertIsNone(restricted["pass_residual_error"])
        self.assertIsNone(restricted["annotation_error"]["recall"])
