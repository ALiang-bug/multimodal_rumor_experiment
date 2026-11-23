"""Stratified sampling and weighted reviewer diagnostics."""

from collections import defaultdict
from math import ceil
import random

from .validation import canonical_hash


DECISIONS = {"PASS", "NEEDS_HUMAN", "REJECT"}


def draw_audit(population, policy, *, seed):
    if "reviewer_decision" not in policy["strata"]:
        raise ValueError("Audit strata must distinguish reviewer decisions")
    groups = defaultdict(list)
    seen = set()
    for row in population:
        rid = row["record_id"]
        if rid in seen or row["reviewer_decision"] not in DECISIONS:
            raise ValueError(f"Duplicate or unreviewed audit population item: {rid}")
        seen.add(rid)
        stratum = canonical_hash([row[field] for field in policy["strata"]])
        groups[stratum].append(row)
    if not population:
        raise ValueError("Audit population is empty")
    randomizer = random.Random(seed)
    selections = []
    for stratum, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["record_id"])
        fraction_key = ("auto_pass_sampling_fraction_initial"
                        if rows[0]["reviewer_decision"] == "PASS" else "routed_sampling_fraction_initial")
        fraction = policy[fraction_key]
        if not 0 < fraction <= 1:
            raise ValueError(f"Invalid audit fraction: {fraction_key}")
        size = len(rows)
        count = ceil(size * fraction)
        for row in randomizer.sample(rows, count):
            selections.append({
                "record_id": row["record_id"], "reviewer_decision": row["reviewer_decision"],
                "stratum_id": stratum, "stratum_population_size": size,
                "population_size": len(population), "sampling_probability": count / size,
                "sampling_seed": seed,
            })
    return selections


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def reviewer_metrics(population, selections, judgments, policy):
    if "reviewer_decision" not in policy["strata"]:
        raise ValueError("Audit strata must distinguish reviewer decisions")
    population_map = {row["record_id"]: row for row in population}
    if not population_map or len(population_map) != len(population):
        raise ValueError("Expected a nonempty population with unique record IDs")
    for row in population:
        if row["reviewer_decision"] not in DECISIONS or type(row["routed_to_human"]) is not bool:
            raise ValueError("Population requires a valid decision and actual routing status")
    selected = {row["record_id"]: row for row in selections}
    answers = {row["record_id"]: row for row in judgments}
    if (len(selected) != len(selections) or len(answers) != len(judgments)
            or not selected.keys() <= population_map.keys() or selected.keys() != answers.keys()):
        raise ValueError("Require exactly one independent judgment per selected population record")
    strata = defaultdict(list)
    for rid, row in population_map.items():
        stratum = canonical_hash([row[field] for field in policy["strata"]])
        strata[stratum].append(rid)
    complete = all(any(rid in selected for rid in ids) for ids in strata.values())
    weighted = {key: dict.fromkeys(("tp", "fp", "fn", "tn"), 0.0)
                for key in ("annotation_error", "needs_intervention")}
    pass_weight = 0.0
    pass_error = 0.0
    for stratum, ids in strata.items():
        audited = [rid for rid in ids if rid in selected]
        for rid in audited:
            row, answer = selected[rid], answers[rid]
            decision = population_map[rid]["reviewer_decision"]
            probability = len(audited) / len(ids)
            if (row["sampling_probability"] != probability or row["stratum_id"] != stratum
                    or row["stratum_population_size"] != len(ids)
                    or row["population_size"] != len(population) or row["reviewer_decision"] != decision):
                raise ValueError(f"Invalid selection metadata for {rid}")
            weight = 1 / probability
            alert = decision in policy["reviewer_alert_decisions"]
            for key, matrix in weighted.items():
                truth = answer[key]
                if type(truth) is not bool:
                    raise ValueError(f"Expected boolean judgment {key} for {rid}")
                cell = ("t" if alert == truth else "f") + ("p" if alert else "n")
                matrix[cell] += weight
            if decision == "PASS":
                pass_weight += weight
                pass_error += weight * answer["annotation_error"]
    diagnostics = {}
    for key, matrix in weighted.items():
        tp, fp, fn, tn = (matrix[cell] for cell in ("tp", "fp", "fn", "tn"))
        diagnostics[key] = {
            "weighted_confusion": matrix if complete else None,
            "recall": _ratio(tp, tp + fn) if complete else None,
            "precision": _ratio(tp, tp + fp) if complete else None,
            "false_positive_rate": _ratio(fp, fp + tn) if complete else None,
        }
    pass_strata = [ids for ids in strata.values() if any(population_map[rid]["reviewer_decision"] == "PASS" for rid in ids)]
    pass_complete = all(any(rid in selected for rid in ids) for ids in pass_strata)
    routed = sum(row["routed_to_human"] for row in population)
    return {
        **diagnostics, "population": len(population), "audited": len(selections),
        "routed_count": routed, "routed_fraction": routed / len(population),
        "estimated_pass_weight": pass_weight,
        "pass_residual_error": _ratio(pass_error, pass_weight) if pass_complete else None,
        "all_strata_covered": complete,
    }
