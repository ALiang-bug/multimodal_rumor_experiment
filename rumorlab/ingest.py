"""Normalize generic JSONL source rows and import a validated snapshot."""

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path

from .validation import ID_FIELDS, canonical_hash, validate_rows
from .grouping import build_groups


def normalize(row, source, media_root, is_mock):
    from PIL import Image

    text = row["text"]
    source_id = row["source_id"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonempty string")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must be a nonempty string")
    if not all(source.get(key) for key in ("dataset_name", "dataset_version", "license_reference")):
        raise ValueError("Configure dataset name, version and license before raw ingestion")
    paths = row["images"]
    if not isinstance(paths, list) or not paths:
        raise ValueError("images must be a nonempty list of relative paths")
    media = []
    for relative in paths:
        if not isinstance(relative, str) or not relative or Path(relative).anchor:
            raise ValueError("Image paths must be nonempty and relative to the media root")
        path = (media_root / relative).resolve()
        if not path.is_relative_to(media_root):
            raise ValueError(f"Media escapes data root: {relative}")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        with Image.open(BytesIO(raw)) as image:
            mime = Image.MIME.get(image.format)
            width, height = image.size
            image.load()
        media.append({
            "media_id": "asset_" + canonical_hash([digest, relative]), "path": relative,
            "sha256": digest, "hash_status": "verified", "mime_type": mime,
            "width": width, "height": height, "original_media_id": None,
        })
    source_record = {
        "dataset": source["dataset_name"], "dataset_version": source["dataset_version"],
        "source_id": source_id, "source_url": row.get("source_url"),
        "source_veracity": None, "license_reference": source["license_reference"],
        "redistribution_status": "fixture_only" if is_mock else row.get("redistribution_status", "unverified"),
    }
    raw_label = row.get("source_veracity")
    if raw_label is not None:
        mapping = source["source_veracity_mapping"]
        if str(raw_label) not in mapping:
            raise ValueError(f"No explicit mapping for source_veracity {raw_label!r}")
        source_record["source_veracity"] = raw_label
    family = "family_" + canonical_hash([source_record["dataset"], source_id])
    claims = row["target_claims"]
    semantic = {
        "source": source_record, "text": text, "media": media, "target_claims": claims,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_provenance": "OT", "image_provenance": "OI", "family_id": family,
        "parent_sample_ids": [], "operation": None, "evidence_ids": [],
    }
    digest = canonical_hash(semantic)
    return {
        **semantic, "sample_id": "sample_" + digest, "content_hash": digest,
        "input_hash": canonical_hash(row), "record_type": "sample", "schema_version": "1.0.0",
        "is_mock": is_mock, "mock_note": "Synthetic generic_jsonl input" if is_mock else None,
        "created_at": datetime.now(timezone.utc).isoformat(), "record_version": 1,
        "supersedes_id": None, "status": "VALIDATED", "leakage_group_id": family,
        "split": row.get("split", "unassigned") if source["preserve_source_split"] else "unassigned",
        "pools": [],
    }


def ingest(state, rows, contract, media_root, run_id, *, fixture):
    media_root = media_root.resolve()
    if fixture:
        if any(not isinstance(row, dict) or row.get("is_mock") is not True for _, row in rows):
            raise ValueError("Fixture snapshots must explicitly mark every record is_mock=true")
    else:
        normalized = []
        for line, row in rows:
            try:
                normalized.append((line, normalize(
                    row, contract.experiment["source"], media_root,
                    contract.experiment["mode"] == "mock",
                )))
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise ValueError(f"Source line {line}: {exc}") from exc
        rows = normalized
    return import_snapshot(state, rows, contract, media_root, run_id)


def import_snapshot(state, rows, contract, media_root, run_id, *, sampling=None):
    if not run_id.strip():
        raise ValueError("run-id must be nonempty")
    errors, _ = validate_rows(rows, contract.schema, contract.labels, contract.gold_votes, media_root)
    if errors:
        raise ValueError("\n".join(errors))
    records = [row for _, row in rows]
    build_groups(records)
    inputs = {
        "config": contract.config_hash, "media_root": str(media_root),
        "records": [{key: value for key, value in row.items() if key != "created_at"} for row in records],
    }
    if sampling is not None:
        inputs["sampling"] = sampling
    digest = canonical_hash(inputs)
    with state.db:
        inserted = state.db.execute(
            "INSERT OR IGNORE INTO imports VALUES (?, ?, ?, ?)",
            (run_id, digest, str(media_root), json.dumps(records, ensure_ascii=False)),
        ).rowcount
        previous = state.db.execute("SELECT digest FROM imports WHERE run_id = ?", (run_id,)).fetchone()
        if previous["digest"] != digest:
            raise ValueError(f"Run {run_id!r} already has a different input snapshot")
        if inserted:
            if sampling is not None:
                state.db.execute("INSERT INTO import_manifests VALUES (?, ?)", (run_id, json.dumps(sampling)))
            by_id = {row[ID_FIELDS[row["record_type"]]]: row for row in records}
            for row in records:
                old_id = row["supersedes_id"]
                if old_id:
                    state._invalidate_in_transaction(old_id, by_id[old_id]["content_hash"])
    return len(records), bool(inserted)
