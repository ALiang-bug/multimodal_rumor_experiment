"""Fail-closed checks for a candidate release snapshot."""

from .grouping import build_groups
from .validation import ID_FIELDS, validate_rows


def check_release(rows, contract, media_root, confirmed_links=()):
    errors, _ = validate_rows(rows, contract.schema, contract.labels, contract.gold_votes, media_root)
    contract_failed = bool(errors)
    if any(not isinstance(row, dict) or row.get("record_type") not in ID_FIELDS
           or any(field not in row for field in ("is_mock", "status", ID_FIELDS.get(row.get("record_type"))))
           for _, row in rows):
        return errors
    records = [row for _, row in rows]
    for row in records:
        kind = row["record_type"]
        rid = row[ID_FIELDS[kind]]
        if row["is_mock"]:
            errors.append(f"{rid}: mock record cannot enter a formal release")
        if row["status"] in {"STALE", "QUARANTINED", "EXCLUDED", "UNAVAILABLE", "WITHDRAWN"}:
            errors.append(f"{rid}: unusable release status {row['status']}")
    if contract_failed:
        return errors
    try:
        build_groups(records, confirmed_links)
    except ValueError as exc:
        errors.append(str(exc))
    samples = {row["sample_id"]: row for row in records if row["record_type"] == "sample"}
    for sid, row in samples.items():
        source = row["source"]
        if not source["license_reference"].strip() or source["redistribution_status"] != "permitted":
            errors.append(f"{sid}: media redistribution permission is unverified or restricted")
        if row["split"] == "unassigned" or row["status"] != "VALIDATED":
            errors.append(f"{sid}: release requires a validated and assigned sample")
    for row in records:
        kind = row["record_type"]
        if kind not in {"annotation", "qa"}:
            continue
        rid = row[ID_FIELDS[kind]]
        tier = row["quality_tier"]
        split = samples[row["sample_id"]]["split"]
        allowed = {"gold"} if split == "test" else {"silver", "gold"}
        if tier not in allowed or row["status"].lower() != f"accepted_{tier}":
            errors.append(f"{rid}: {split} release cannot use {row['status']}/{tier}")
    return errors
