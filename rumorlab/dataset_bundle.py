"""Export two sample-level datasets and construction statistics from frozen labeling runs."""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from .mmfakebench import VERACITY
from .prompts_v1_3 import TASKS, VERSION
from .validation import canonical_hash


SOURCE_AXES = {
    "original": ("OT", "OI"),
    "mismatch": ("OT", "OI"),
    "textual_veracity_distortion": ("FT", None),
    "visual_veracity_distortion": ("OT", "FI"),
}
SOURCE_FIELDS = ("textual_veracity_distortion", "visual_veracity_distortion", "source_veracity")
LABEL_FIELDS = (*TASKS, *SOURCE_FIELDS)


def join_samples(records, manifest, source_rows, val=None):
    samples = {row["sample_id"]: row for row in records if row["record_type"] == "sample"}
    annotations, reviews = {}, {}
    for record in records:
        if record["record_type"] not in {"annotation", "review"} or record["is_mock"]:
            continue
        if record["producer"].get("prompt_version") != VERSION or record["status"] == "STALE":
            continue
        if record["record_type"] == "annotation" and record["payload"]["task_id"] in TASKS:
            key = (record["sample_id"], record["payload"]["task_id"])
            if key in annotations:
                raise ValueError(f"Ambiguous current annotation: {key}")
            annotations[key] = record
        elif record["record_type"] == "review" and record["status"] == "COMPLETED":
            if record["target_id"] in reviews:
                raise ValueError(f"Ambiguous current review: {record['target_id']}")
            reviews[record["target_id"]] = record

    selection = manifest["selection"]
    if len(selection) != len(samples) or {item["sample_id"] for item in selection} != samples.keys():
        raise ValueError("Snapshot and frozen selection disagree")
    if val is not None:
        selection = sorted(selection, key=lambda item: canonical_hash([manifest["seed"], item["sample_id"]]))
    output = []
    for item in selection:
        sample = samples[item["sample_id"]]
        source = source_rows[item["source_row"] - 1]
        if canonical_hash(source) != item["source_row_sha256"] or source["text"] != sample["text"]:
            raise ValueError(f"Source row changed: {item['source_row']}")
        raw_veracity = source["gt_answers"]
        veracity = VERACITY[raw_veracity.lower()]
        fake_cls = source.get("fake_cls")
        text_label, image_label = SOURCE_AXES.get(fake_cls, (None, None))
        if fake_cls in SOURCE_AXES and (fake_cls == "original") != (veracity == "real"):
            raise ValueError(f"Conflicting official labels at source row {item['source_row']}")
        labels = {field: None for field in TASKS}
        labels.update(zip(SOURCE_FIELDS, (text_label, image_label, veracity)))
        media = [{"path": entry["path"], "sha256": entry["sha256"]} for entry in sample["media"]]
        overlaps = []
        if val is not None:
            val_text, val_images = val
            if sample["text_sha256"] in val_text:
                overlaps.append("text_sha256")
            if any(entry["sha256"] in val_images for entry in media):
                overlaps.append("image_sha256")
        decisions = {}
        for task in TASKS:
            annotation = annotations.get((sample["sample_id"], task))
            if annotation is None:
                decisions[task] = None
                continue
            payload = annotation["payload"]
            annotation_id = annotation["annotation_id"]
            review = reviews.get(annotation_id)
            if review and review["target_content_hash"] != annotation["content_hash"]:
                raise ValueError(f"Review targets a different annotation version: {annotation_id}")
            candidate, = payload["labels"]
            decision = review["payload"]["decision"] if review else None
            if decision == "PASS" and annotation["status"] in {"ACCEPTED_SILVER", "ACCEPTED_GOLD"}:
                labels[task] = candidate
            decisions[task] = {
                "annotation_id": annotation_id, "candidate_label": candidate,
                "rationale": payload["rationale"], "evidence_ids": payload["evidence_ids"],
                "status": annotation["status"], "decision": decision,
                "review_id": review["review_id"] if review else None,
                "review": review["payload"] if review else None,
            }
        output.append({
            "sample_id": sample["sample_id"], "split": "test" if val is not None else "val",
            "text": sample["text"], "text_sha256": sample["text_sha256"], "images": media,
            "labels": labels, "usable_labels": [] if overlaps else [key for key in LABEL_FIELDS if labels[key] is not None],
            "val_overlap": overlaps,
            "source": {**item, "fake_cls": fake_cls, "gt_answers": raw_veracity,
                       "annotation_sha256": manifest["annotation_sha256"]},
            "model_annotations": decisions,
        })
    return output


def summarize(rows):
    distributions, usable, reviews = {}, {}, {}
    for field in LABEL_FIELDS:
        distributions[field] = dict(Counter(row["labels"][field] or "MISSING" for row in rows))
        usable[field] = dict(Counter(row["labels"][field] for row in rows if field in row["usable_labels"]))
    for task in TASKS:
        counts = Counter()
        for row in rows:
            item = row["model_annotations"][task]
            counts["ANNOTATION_MISSING" if item is None else item["decision"] or "REVIEW_MISSING"] += 1
        reviews[task] = dict(counts)
    return {
        "samples": len(rows), "source_distribution": dict(Counter(row["source"]["fake_cls"] or "MISSING" for row in rows)),
        "labels": distributions, "usable_label_distribution": usable, "review_decisions": reviews,
        "all_four_model_labels_pass": sum(all(row["labels"][task] is not None for task in TASKS) for row in rows),
        "all_seven_labels_usable": sum(len(row["usable_labels"]) == len(LABEL_FIELDS) for row in rows),
        "excluded_for_exact_val_overlap": sum(bool(row["val_overlap"]) for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("media_root", type=Path)
    parser.add_argument("val_source", type=Path)
    parser.add_argument("test_source", type=Path)
    parser.add_argument("val_run")
    parser.add_argument("test_run")
    args = parser.parse_args()

    cohorts, stats = {}, {}
    for split, run_id, path in (("val", args.val_run, args.val_source), ("test", args.test_run, args.test_source)):
        folder = args.attempt / split
        with closing(sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            entry, = db.execute("SELECT manifest FROM import_manifests WHERE run_id = ?", (run_id,)).fetchone()
        manifest = json.loads(entry)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["annotation_sha256"]:
            raise ValueError(f"Official source changed since {run_id} was imported")
        with (folder / "snapshot.jsonl").open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        val = None
        if split == "test":
            val = ({row["text_sha256"] for row in cohorts["val"]},
                   {image["sha256"] for row in cohorts["val"] for image in row["images"]})
        rows = join_samples(records, manifest, json.loads(raw), val)
        cohorts[split] = rows
        stats[split] = summarize(rows)
        stats[split]["sampling"] = {key: manifest[key] for key in ("method", "seed", "annotation_sha256", "strata")}
        stats[split]["stages"] = {}
        for stage in ("annotate", "review-labels"):
            report = json.loads((folder / f"{stage}.json").read_text(encoding="utf-8"))
            stats[split]["stages"][stage] = {
                "remaining_failures": report["failed"], "timeout_recovery": report.get("timeout_recovery", {"selected": 0, "first_pass_failed": report["failed"]}),
            }
        (folder / "selection.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for split, rows in cohorts.items():
        destination = args.attempt / f"{split}.jsonl"
        temporary = destination.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(destination)
    report = json.dumps({"prompt_version": VERSION, "media_root": str(args.media_root.resolve()), **stats},
                        ensure_ascii=False, indent=2) + "\n"
    (args.attempt / "statistics.json").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
