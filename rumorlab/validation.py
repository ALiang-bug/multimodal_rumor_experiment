"""JSON Schema and closed-world record validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import jsonschema

ID_FIELDS = {
    "sample": "sample_id", "evidence": "evidence_id",
    "annotation": "annotation_id", "review": "review_id", "qa": "qa_id",
    "human_vote": "human_vote_id", "adjudication": "adjudication_id",
    "audit_selection": "selection_id", "run_manifest": "run_id",
    "job_event": "event_id",
}
ZERO_HASH = "0" * 64


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_rows(rows, schema, labels, gold_votes_required, data_root, *, check_media=True):
    errors = []

    def fail(message):
        errors.append(message)

    if not rows:
        fail("No JSON records were supplied.")
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    for line_no, row in rows:
        # Validate against the selected $def to keep failures readable.
        kind = row.get("record_type") if isinstance(row, dict) else None
        if kind not in ID_FIELDS:
            fail(f"line {line_no}: unknown or missing record_type {kind!r}")
            continue
        selected = {"$schema": schema["$schema"], "$defs": schema["$defs"],
                    "$ref": f"#/$defs/{kind}"}
        for err in jsonschema.Draft202012Validator(
                selected, format_checker=validator.format_checker).iter_errors(row):
            location = ".".join(str(x) for x in err.absolute_path) or "<record>"
            fail(f"line {line_no} {kind}.{location}: {err.message}")

    if errors:
        return errors, set()

    by_id: dict[str, dict] = {}
    for _, row in rows:
        record_id = row[ID_FIELDS[row["record_type"]]]
        if record_id in by_id:
            fail(f"Duplicate record identifier: {record_id}")
        by_id[record_id] = row

    def get(record_id: str, expected: str | None, context: str) -> dict | None:
        row = by_id.get(record_id)
        if row is None:
            fail(f"{context}: missing reference {record_id}")
        elif expected is not None and row["record_type"] != expected:
            fail(f"{context}: {record_id} is not a {expected} record")
            return None
        return row

    def evidence_refs(evidence_ids: list, context: str, available=None) -> None:
        for evidence_id in evidence_ids:
            get(evidence_id, "evidence", context)
        if available is not None and not set(evidence_ids).issubset(available):
            fail(f"{context}: cites evidence outside available_evidence_ids")

    task_map = labels["tasks"]
    allowed = {task: {item["code"] for item in info["labels"]}
               for task, info in task_map.items()}
    if set(schema["$defs"]["task_id"]["enum"]) != set(task_map):
        fail("Schema task_id enumeration differs from configs/labels.yaml")
    for payload_name, field in [("annotation_payload", "labels"), ("qa_payload", "target_labels")]:
        branches = schema["$defs"][payload_name]["allOf"]
        for task, codes in allowed.items():
            matching = [b for b in branches
                        if b.get("if", {}).get("properties", {}).get("task_id", {}).get("const") == task]
            declared = matching[0]["then"]["properties"][field]["items"]["enum"] if matching else []
            if set(declared) != codes:
                fail(f"{payload_name}: schema label enumeration out of sync for {task}")

    def label_set(payload: dict, field: str, context: str) -> None:
        task = payload["task_id"]
        values = payload[field]
        if not set(values).issubset(allowed[task]):
            fail(f"{context}: label outside configured task vocabulary")
        if task_map[task]["cardinality"] == "single" and len(values) > 1:
            fail(f"{context}: multiple labels in a single-label task")
        if "UNKNOWN" in values and len(values) != 1:
            fail(f"{context}: UNKNOWN must be exclusive")

    media_by_id: dict[str, dict] = {}
    for _, row in rows:
        if row["record_type"] != "sample":
            continue
        sid = row["sample_id"]
        if hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != row["text_sha256"]:
            fail(f"{sid}: text_sha256 does not match UTF-8 text")
        for claim in row["target_claims"]:
            start, end = claim["text_span"]["start"], claim["text_span"]["end"]
            if end < start or row["text"][start:end] != claim["text"]:
                fail(f"{sid}: target_claim code-point span does not match text")
        for asset in row["media"]:
            previous = media_by_id.get(asset["media_id"])
            if previous is not None and previous != asset:
                fail(f"{sid}: inconsistent reuse of media_id {asset['media_id']}")
            media_by_id[asset["media_id"]] = asset
            if not row["is_mock"] and (asset["hash_status"] != "verified" or asset["sha256"] == ZERO_HASH):
                fail(f"{sid}: real sample contains unverified media or placeholder hash")
        for parent_id in row["parent_sample_ids"]:
            parent = get(parent_id, "sample", sid)
            if parent and (parent["leakage_group_id"] != row["leakage_group_id"] or parent["split"] != row["split"]):
                fail(f"{sid}: parent and derivative cross leakage group or split")
        evidence_refs(row["evidence_ids"], sid)
        if row["operation"]:
            evidence_refs(row["operation"]["verification_evidence_ids"], sid)
    group_splits: dict[str, set] = {}
    for _, row in rows:
        if row["record_type"] == "sample":
            group_splits.setdefault(row["leakage_group_id"], set()).add(row["split"])
    for group, splits in group_splits.items():
        if len(splits) != 1:
            fail(f"{group}: one leakage group occupies multiple splits")

    for _, row in rows:
        kind = row["record_type"]
        rid = row[ID_FIELDS[kind]]
        if row["supersedes_id"]:
            old = get(row["supersedes_id"], kind, rid)
            if old and row["record_version"] <= old["record_version"]:
                fail(f"{rid}: superseding record must advance record_version")
        if kind in {"annotation", "review", "qa"}:
            if canonical_hash(row["payload"]) != row["content_hash"]:
                fail(f"{rid}: content_hash differs from canonical payload JSON")
            evidence_refs(row["available_evidence_ids"], rid)
            evidence_refs(row["payload"]["evidence_ids"], rid, row["available_evidence_ids"])
            run = get(row["producer"]["run_id"], "run_manifest", rid)
            if run and run["is_mock"] != row["is_mock"]:
                fail(f"{rid}: mock and real run identities are mixed")
        if kind == "evidence":
            for sample_id in row["sample_ids"]:
                get(sample_id, "sample", rid)
            asset_id = row["source_locator"]["media_id"]
            if asset_id and asset_id not in media_by_id:
                fail(f"{rid}: unknown source media_id {asset_id}")
        if kind == "annotation":
            sample = get(row["sample_id"], "sample", rid)
            payload = row["payload"]
            label_set(payload, "labels", rid)
            if sample and payload["target_claim"] not in sample["target_claims"]:
                fail(f"{rid}: target_claim differs from the registered sample target")
            if sample and payload["applicability"] == "applicable":
                for task, field in [("text_reasoning", "text_provenance"), ("image_perception", "image_provenance")]:
                    if payload["task_id"] == task and payload["labels"] != [sample[field]]:
                        fail(f"{rid}: provenance label differs from sample manifest")
        if kind == "review":
            target = get(row["target_id"], row["target_record_type"], rid)
            if target and row["target_content_hash"] != target["content_hash"]:
                fail(f"{rid}: review binds the wrong content hash")
            suggested = row["payload"]["suggested_annotation"]
            if suggested:
                label_set(suggested, "labels", rid)
                evidence_refs(suggested["evidence_ids"], rid, row["available_evidence_ids"])
        if kind == "qa":
            from .qa import project_options

            try:
                project_options(row["payload"], seed=0, qa_id=rid)
            except ValueError as exc:
                fail(f"{rid}: {exc}")
            sample = get(row["sample_id"], "sample", rid)
            annotation = get(row["source_annotation_id"], "annotation", rid)
            payload = row["payload"]
            label_set(payload, "target_labels", rid)
            if annotation:
                if annotation["content_hash"] != row["source_annotation_content_hash"]:
                    fail(f"{rid}: upstream annotation content hash mismatch")
                if row["status"] != "STALE" and annotation["status"] not in {"ACCEPTED_SILVER", "ACCEPTED_GOLD"}:
                    fail(f"{rid}: QA depends on unaccepted or stale annotation")
                if annotation["sample_id"] != row["sample_id"] or annotation["payload"]["task_id"] != payload["task_id"]:
                    fail(f"{rid}: upstream sample or task mismatch")
                if set(annotation["payload"]["labels"]) != set(payload["target_labels"]):
                    fail(f"{rid}: QA target_labels differ from upstream labels")
                if row["quality_tier"] == "gold" and annotation["quality_tier"] != "gold":
                    fail(f"{rid}: gold QA requires a gold upstream annotation")
            options = payload["options"]
            option_ids = [option["option_id"] for option in options]
            if len(set(option_ids)) != len(option_ids):
                fail(f"{rid}: duplicate option_id")
            if len({option["text"].strip() for option in options}) != len(options):
                fail(f"{rid}: duplicate option text")
            if not set(payload["answer_option_ids"]).issubset(option_ids):
                fail(f"{rid}: answer_option_ids contains a nonexistent option")
            for option in options:
                if not set(option.get("label_codes", [])).issubset(allowed[payload["task_id"]]):
                    fail(f"{rid}: option has a nonconfigured task label")
            correct = [option for option in options if option["option_id"] in payload["answer_option_ids"]]
            if correct and all("label_codes" in option for option in correct):
                mapped = {code for option in correct for code in option["label_codes"]}
                if mapped != set(payload["target_labels"]):
                    fail(f"{rid}: correct option labels do not cover target_labels")
            solver = row["solver_input"]
            if sample and solver["text"] != sample["text"]:
                fail(f"{rid}: example solver text differs from the current sample")
            if any(mid not in media_by_id for mid in solver["media_ids"]):
                fail(f"{rid}: solver_input references unknown media")
            required = set(payload["required_inputs"])
            if "image" in required and not solver["media_ids"]:
                fail(f"{rid}: required image is absent")
            if "text" in required and not solver["text"]:
                fail(f"{rid}: required text is absent")
            if "evidence" in required and not solver["evidence_ids"]:
                fail(f"{rid}: required solver evidence is absent")
            if required.intersection({"reference_text", "reference_image"}) and not solver["reference_sample_ids"]:
                fail(f"{rid}: required reference material is absent")
            for reference_id in solver["reference_sample_ids"]:
                reference = get(reference_id, "sample", rid)
                if sample and reference and reference["split"] != sample["split"]:
                    fail(f"{rid}: reference sample crosses split")
            for evidence_id in solver["evidence_ids"]:
                ev = get(evidence_id, "evidence", rid)
                if ev and ev["visibility"] == "construction":
                    fail(f"{rid}: construction-only evidence leaked to solver_input")
        if kind in {"annotation", "qa"}:
            for review_id in row["review_ids"]:
                review = get(review_id, "review", rid)
                if review and (review["target_id"] != rid or review["target_content_hash"] != row["content_hash"]):
                    fail(f"{rid}: review reference belongs to another record or content version")
            if row["status"] == "ACCEPTED_SILVER":
                reviewed = [by_id.get(x) for x in row["review_ids"]]
                if not any(x and x["record_type"] == "review" and x["status"] == "COMPLETED" and x["payload"]["decision"] == "PASS" for x in reviewed):
                    fail(f"{rid}: accepted silver has no completed PASS review")
            votes = [get(x, "human_vote", rid) for x in row["human_vote_ids"]]
            adjudication = get(row["adjudication_id"], "adjudication", rid) if row["adjudication_id"] else None
            if row["quality_tier"] == "gold":
                if not adjudication or not adjudication["eligible_for_gold"]:
                    fail(f"{rid}: gold lacks eligible adjudication")
                valid_votes = [x for x in votes if x and x["vote_mode"] == "independent" and x["status"] == "SUBMITTED" and x["blinded_to_model"] and x["blinded_to_other_humans"]]
                if len({x["annotator_id"] for x in valid_votes}) < gold_votes_required:
                    fail(f"{rid}: gold lacks {gold_votes_required} distinct independent human annotators required by the experiment config")
                if adjudication:
                    if adjudication["is_mock"] != row["is_mock"] or adjudication["target_record_type"] != kind:
                        fail(f"{rid}: adjudication has the wrong mock identity or record type")
                    if any(vote["is_mock"] != row["is_mock"] for vote in valid_votes):
                        fail(f"{rid}: gold votes have the wrong mock identity")
                    if not set(row["human_vote_ids"]).issubset(adjudication["human_vote_ids"]):
                        fail(f"{rid}: gold votes are absent from adjudication")
                    final_key = "final_annotation_payload" if kind == "annotation" else "final_qa_payload"
                    if adjudication[final_key] != row["payload"]:
                        fail(f"{rid}: gold payload differs from adjudicated final payload")
                    if adjudication["target_id"] not in {rid, row["supersedes_id"]}:
                        fail(f"{rid}: adjudication targets an unrelated record")
        if kind in {"human_vote", "adjudication"}:
            target = get(row["target_id"], row["target_record_type"], rid)
            if target and target["content_hash"] != row["target_content_hash"]:
                fail(f"{rid}: human record binds the wrong target content hash")
            if kind == "human_vote" and row["audit_selection_id"]:
                get(row["audit_selection_id"], "audit_selection", rid)
            if kind == "adjudication":
                evidence_refs(row["evidence_ids"], rid)
                for vote_id in row["human_vote_ids"]:
                    vote = get(vote_id, "human_vote", rid)
                    if vote and (vote["target_id"] != row["target_id"] or vote["target_content_hash"] != row["target_content_hash"]):
                        fail(f"{rid}: adjudication includes votes for another target")
        if kind == "audit_selection":
            get(row["selected_record_id"], row["selected_record_type"], rid)
            if row["stratum_population_size"] > row["population_size"]:
                fail(f"{rid}: stratum is larger than its population")
        if kind == "job_event":
            get(row["run_id"], "run_manifest", rid)
            get(row["subject_id"], None, rid)
            for output_id in row["output_record_ids"]:
                get(output_id, None, rid)

    if check_media:
        checked_paths: set[Path] = set()
        data_root = data_root.resolve()
        for asset in media_by_id.values():
            path = (data_root / asset["path"]).resolve()
            if not path.is_relative_to(data_root):
                fail(f"Media escapes data root: {asset['path']}")
                continue
            try:
                raw = path.read_bytes()
            except OSError as exc:
                fail(f"Cannot read media {asset['path']}: {exc}")
                continue
            actual = hashlib.sha256(raw).hexdigest()
            if asset["hash_status"] != "verified" or actual != asset["sha256"]:
                fail(f"Media hash is missing, placeholder or incorrect: {asset['path']}")
            if asset["mime_type"] == "image/png":
                if raw[:8] != b"\x89PNG\r\n\x1a\n" or len(raw) < 24:
                    fail(f"Invalid PNG header: {asset['path']}")
                elif struct.unpack(">II", raw[16:24]) != (asset["width"], asset["height"]):
                    fail(f"PNG dimensions disagree with media record: {asset['path']}")
            checked_paths.add(path)
            for _, row in rows:
                if row["record_type"] == "evidence" and row["source_locator"]["media_id"] == asset["media_id"]:
                    if row["snapshot_sha256"] != actual:
                        fail(f"{row['evidence_id']}: evidence snapshot hash differs from media")
    else:
        checked_paths = set()

    return errors, checked_paths
