"""Connected leakage groups and deterministic split plans."""

from collections import defaultdict
import json
import math
import random

from .validation import canonical_hash


def build_groups(records, confirmed_links=()):
    samples = {row["sample_id"]: row for row in records if row["record_type"] == "sample"}
    parents = {sid: sid for sid in samples}
    edges = []

    def root(sid):
        if sid not in parents:
            raise ValueError(f"Unknown grouping sample: {sid}")
        while parents[sid] != sid:
            parents[sid] = parents[parents[sid]]
            sid = parents[sid]
        return sid

    def connect(first, second, reason):
        left, right = sorted((root(first), root(second)))
        if first != second:
            edges.append({"sample_ids": sorted((first, second)), "reason": reason})
        parents[right] = left

    seen = {}
    for sid, sample in sorted(samples.items()):
        keys = [("declared_group", sample["leakage_group_id"]), ("family", sample["family_id"])]
        keys.extend(("media_hash", item["sha256"]) for item in sample["media"] if item["hash_status"] == "verified")
        for item in sample["media"]:
            keys.append(("media_lineage", item["media_id"]))
            if item["original_media_id"]:
                keys.append(("media_lineage", item["original_media_id"]))
        for key in keys:
            connect(seen.setdefault(key, sid), sid, key[0])
        for parent in sample["parent_sample_ids"]:
            connect(parent, sid, "parent")
        if sample["supersedes_id"]:
            connect(sample["supersedes_id"], sid, "supersedes")
    for link in confirmed_links:
        if link["reason"] not in {"same_event", "confirmed_near_duplicate"}:
            raise ValueError("Expected a confirmed event or near-duplicate relation")
        first, second = link["sample_ids"]
        connect(first, second, link["reason"])
    for row in records:
        if row["record_type"] == "qa":
            for reference in row["solver_input"]["reference_sample_ids"]:
                connect(row["sample_id"], reference, "qa_reference")

    members = defaultdict(list)
    for sid in sorted(samples):
        members[root(sid)].append(sid)
    groups = {"group_" + canonical_hash(ids): ids for ids in members.values()}
    splits = {}
    for group, ids in groups.items():
        assigned = {samples[sid]["split"] for sid in ids} - {"unassigned"}
        if len(assigned) > 1:
            raise ValueError(f"Known split leakage: {ids} occupy {sorted(assigned)}")
        if assigned:
            splits[group] = assigned.pop()
    return {"groups": groups, "source_splits": splits, "links": sorted(edges, key=lambda edge: (edge["sample_ids"], edge["reason"]))}


def plan_splits(grouping, policy):
    weights = {name: policy[name] for name in ("train", "dev", "test")}
    if any(not 0 <= weight <= 1 for weight in weights.values()) or not math.isclose(sum(weights.values()), 1):
        raise ValueError("Train, dev and test fractions must sum to one")
    groups = grouping["groups"]
    assignment = dict(grouping["source_splits"])
    total = sum(map(len, groups.values()))
    counts = {name: 0 for name in weights}
    for group, split in assignment.items():
        counts[split] += len(groups[group])
    pending = sorted(groups.keys() - assignment.keys())
    random.Random(policy["seed"]).shuffle(pending)
    pending.sort(key=lambda group: len(groups[group]), reverse=True)
    for group in pending:
        split = max(weights, key=lambda name: total * weights[name] - counts[name])
        assignment[group] = split
        counts[split] += len(groups[group])
    return {"seed": policy["seed"], "assignment": assignment, "sample_counts": counts}


def freeze_plan(state, run_id, records, policy, confirmed_links=None):
    if confirmed_links is None:
        saved = state.db.execute("SELECT digest, plan FROM split_plans WHERE run_id = ?", (run_id,)).fetchone()
        if saved:
            return saved["digest"], json.loads(saved["plan"])
    report = build_groups(records, confirmed_links or ())
    report.update(plan_splits(report, policy))
    digest = canonical_hash(report)
    with state.db:
        state.db.execute("INSERT OR IGNORE INTO split_plans VALUES (?, ?, ?)",
                         (run_id, digest, json.dumps(report, ensure_ascii=False)))
        saved = state.db.execute("SELECT digest, plan FROM split_plans WHERE run_id = ?", (run_id,)).fetchone()
        if saved["digest"] != digest:
            if confirmed_links is None:
                return saved["digest"], json.loads(saved["plan"])
            raise ValueError("Run already has a different frozen split plan")
    return digest, report


def apply_plan(records, plan):
    groups = {sid: group for group, members in plan["groups"].items() for sid in members}
    for sample in records:
        if sample["record_type"] == "sample":
            group = groups[sample["sample_id"]]
            sample["leakage_group_id"] = group
            sample["split"] = plan["assignment"][group]
