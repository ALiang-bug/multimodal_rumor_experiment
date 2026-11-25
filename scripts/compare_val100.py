"""Copy the frozen val input into a fresh state DB and compare matched pilot runs."""

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from rumorlab.config import Contract
from rumorlab.m1 import load_lock
from rumorlab.pilot import _inputs, _ordered_sample_ids
from rumorlab.state import State


RUN = "mmfakebench-val"
BASELINE = "m2-val100-baseline"
REVISED = "m2-val100-single-label"
CONFIG = Path("configs/experiment-single-label.yaml")
OFFSETS = (0, 25, 50, 75)
INPUT_TABLES = ("imports", "import_manifests", "split_plans")


def read_only(path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def batches(db, prefix):
    rows = []
    for offset in OFFSETS:
        batch_id = f"{prefix}-{offset}"
        row = db.execute("SELECT * FROM pilot_batches WHERE source_run_id = ? AND batch_id = ?",
                         (RUN, batch_id)).fetchone()
        if row is None or not row["annotate_report"] or not row["review_report"]:
            raise ValueError(f"Missing annotate or review report for {batch_id}")
        rows.append(row)
    return rows


def read_inputs(db):
    values = [db.execute(f"SELECT * FROM {table} WHERE run_id = ?", (RUN,)).fetchone()
              for table in INPUT_TABLES]
    if any(row is None for row in values):
        raise ValueError("Missing val import, import manifest or frozen split plan")
    return [tuple(row) for row in values]


def prepare(old, target, contract, locks):
    inputs = read_inputs(old)
    _, source, _, samples, _ = _inputs(old, RUN)
    ordered = _ordered_sample_ids(source, samples, contract.experiment["seed"], subset=True)
    for offset, row in zip(OFFSETS, batches(old, BASELINE)):
        old_ids = {result["sample_id"] for result in json.loads(row["annotate_report"])["results"]}
        if len(old_ids) != 25 or old_ids != set(ordered[offset:offset + 25]):
            raise ValueError(f"Frozen selection differs from {row['batch_id']}")
        old_locks = json.loads(row["manifest"])["runtime"]
        if any(old_locks[f"{role}_lock_sha256"] != digest for role, digest in locks.items()):
            raise ValueError(f"Model locks differ from {row['batch_id']}")
    with closing(State(target)) as state, state.db:
        for table, values in zip(INPUT_TABLES, inputs):
            placeholders = ", ".join("?" for _ in values)
            state.db.execute(f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})", values)
        if read_inputs(state.db) != inputs:
            raise ValueError("Existing comparison DB contains different val inputs")
    return {"samples_verified": 100, "run_id": RUN, "state_db": str(target)}


def outcomes(db, batch, tasks):
    def key(result):
        return result["sample_id"], result["claim_id"], result["task_id"]

    reviews = {key(item): item for item in json.loads(batch["review_report"])["results"]}
    data = {}
    for result in json.loads(batch["annotate_report"])["results"]:
        item_key = key(result)
        if item_key[-1] not in tasks:
            continue
        labels = None
        if result.get("record_id"):
            record = db.execute(
                "SELECT record FROM pilot_records WHERE manifest_id = ? AND record_id = ?",
                (batch["manifest_id"], result["record_id"]),
            ).fetchone()
            if record is None:
                raise ValueError(f"Missing annotation record for {item_key}")
            labels = json.loads(record["record"])["payload"]["labels"]
        review = reviews.pop(item_key, {})
        if item_key in data or (result["status"] == "VALIDATED") != bool(review):
            raise ValueError(f"Invalid annotation or review coverage for {item_key}")
        data[item_key] = (result["status"], labels, review.get("decision") or review.get("status"))
    if reviews:
        raise ValueError("Reviews found without matching annotations")
    return data


def compare(old, new, tasks):
    if read_inputs(old) != read_inputs(new):
        raise ValueError("The databases differ in imported or frozen input")
    summary = {task: {version: Counter() for version in (BASELINE, REVISED)} for task in tasks}
    transitions = {task: Counter() for task in tasks}
    for left, right in zip(batches(old, BASELINE), batches(new, REVISED)):
        old_manifest, new_manifest = (json.loads(row["manifest"]) for row in (left, right))
        versions = [row["prompt_versions"]["annotate"] for row in (old_manifest, new_manifest)]
        if versions != ["1.1.2", "1.2.0"]:
            raise ValueError(f"Unexpected annotation prompt versions: {versions}")
        for role in ("annotator", "reviewer"):
            field = f"{role}_lock_sha256"
            if old_manifest["runtime"][field] != new_manifest["runtime"][field]:
                raise ValueError(f"Model locks differ for {left['batch_id']} and {right['batch_id']}")
        before, after = outcomes(old, left, tasks), outcomes(new, right, tasks)
        if (set(before) != set(after) or len({key[0] for key in before}) != 25
                or len(before) != 25 * len(tasks)):
            raise ValueError(f"Unmatched val tasks in {left['batch_id']} and {right['batch_id']}")
        for key, (old_status, old_labels, old_review) in before.items():
            task = key[-1]
            new_status, new_labels, new_review = after[key]
            for version, status, labels, review in ((BASELINE, old_status, old_labels, old_review),
                                                     (REVISED, new_status, new_labels, new_review)):
                counts = summary[task][version]
                counts[f"annotate:{status}"] += 1
                counts[f"review:{review or 'NOT_RUN'}"] += 1
                for label in labels or ():
                    counts[f"label:{label}"] += 1
            old_result = "+".join(sorted(old_labels)) if old_labels else old_status
            new_result = "+".join(sorted(new_labels)) if new_labels else new_status
            transitions[task][f"{old_result} -> {new_result}"] += 1
    return {"samples": 100, "per_task": summary, "label_transitions": transitions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "compare"))
    parser.add_argument("--baseline-db", default=Path("runs/state.sqlite3"), type=Path)
    parser.add_argument("--revised-db", default=Path(f"runs/{REVISED}/state.sqlite3"), type=Path)
    args = parser.parse_args()
    if args.baseline_db.resolve() == args.revised_db.resolve():
        parser.error("Baseline and revised DBs must be different files")
    if args.revised_db.exists() and args.baseline_db.exists() and args.revised_db.samefile(args.baseline_db):
        parser.error("Baseline and revised DBs must be different files")
    try:
        contract = Contract(CONFIG)
        with closing(read_only(args.baseline_db)) as old:
            if args.action == "prepare":
                locks = {
                    role: load_lock(contract, role, Path(f"runs/m1-{tag}/model_lock.json"))[1]
                    for role, tag in (("annotator", "qwen"), ("reviewer", "gemma"))
                }
                result = prepare(old, args.revised_db, contract, locks)
            else:
                with closing(read_only(args.revised_db)) as new:
                    result = compare(old, new, contract.experiment["annotation"]["model_tasks"])
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
