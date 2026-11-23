"""Adapt an official MMFakeBench split and freeze a stratified sample."""

from collections import defaultdict
import hashlib
import json
from pathlib import PurePosixPath

from .ingest import normalize
from .validation import canonical_hash


DATASET = "MMFakeBench"
SOURCE_SPLITS = {"MMFakeBench_test": "test", "MMFakeBench_val": "dev"}
VERACITY = {"true": "real", "real": "real", "fake": "fake"}
USAGE_PROTOCOL = "https://huggingface.co/datasets/liuxuannan/MMFakeBench/blob/main/README.md"


def _group_candidates(raw_rows, name):
    candidates = {}
    for index, raw in enumerate(raw_rows):
        try:
            text, path, label = raw["text"], raw["image_path"], raw["gt_answers"]
            fake_class = raw.get("fake_cls")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be nonempty")
            if not isinstance(label, str) or label.lower() not in VERACITY:
                raise ValueError("gt_answers must be True, Real or Fake")
            if fake_class is not None and not isinstance(fake_class, str):
                raise ValueError("fake_cls must be a string or null")
            if not isinstance(path, str) or "\\" in path or ":" in path or "\0" in path:
                raise ValueError("invalid image_path")
            parts = path.removeprefix("/").split("/")
            if any(part in {"", ".", ".."} for part in parts):
                raise ValueError("image_path must stay inside the official split")
            if parts[0] == name:
                parts = parts[1:]
            if len(parts) < 2 or parts[0] not in {"real", "fake"}:
                raise ValueError("expected image_path under real/ or fake/")
            relative = PurePosixPath(name, *parts).as_posix()
            source_id = name.lower() + "_" + canonical_hash([relative, text])
            stratum = (VERACITY[label.lower()], fake_class or "")
            if source_id in candidates:
                if candidates[source_id][0] != stratum:
                    raise ValueError("conflicting labels for the same caption and image path")
                continue
            candidates[source_id] = (stratum, index, raw, relative)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"MMFakeBench JSON row {index + 1}: {exc}") from exc

    strata = defaultdict(list)
    for source_id, (stratum, index, raw, relative) in candidates.items():
        strata[stratum].append((source_id, index, raw, relative))
    return strata


def prepare_mmfakebench(path, media_root, seed, sample_size=None):
    from PIL import Image

    name = path.stem
    if name not in SOURCE_SPLITS or path.suffix != ".json":
        raise ValueError("Expected official MMFakeBench_test.json or MMFakeBench_val.json")
    if sample_size is not None and sample_size < 1:
        raise ValueError("Sample size must be positive")
    media_root = media_root.resolve()
    if not (media_root / name).is_dir():
        raise ValueError(f"Expected extracted image directory at {media_root / name}")
    raw_bytes = path.read_bytes()
    raw_rows = json.loads(raw_bytes)
    if not isinstance(raw_rows, list):
        raise ValueError("MMFakeBench annotations must be a JSON array")
    strata = _group_candidates(raw_rows, name)
    population = sum(map(len, strata.values()))
    if population == 0:
        raise ValueError("MMFakeBench annotations contain no source rows")
    if sample_size is None:
        sample_size = population
    if population < sample_size:
        raise ValueError(f"Requested {sample_size} MMFakeBench samples but found {population} unique source rows")
    source_split = SOURCE_SPLITS[name]
    source_hash = hashlib.sha256(raw_bytes).hexdigest()
    source = {
        "dataset_name": DATASET, "dataset_version": f"sha256:{source_hash}",
        "license_reference": USAGE_PROTOCOL, "preserve_source_split": True,
        "source_veracity_mapping": {raw["gt_answers"]: VERACITY[raw["gt_answers"].lower()] for raw in raw_rows},
    }

    # Largest remainders keep the source proportions while allocating exactly sample_size rows.
    allocation = {key: divmod(len(candidates) * sample_size, population) for key, candidates in strata.items()}
    quotas = {key: whole for key, (whole, _) in allocation.items()}
    remainder_order = sorted(quotas, key=lambda key: (-allocation[key][1], key))
    for key in remainder_order[:sample_size - sum(quotas.values())]:
        quotas[key] += 1

    rows, selection, quarantine, counts = [], [], [], []
    selected_pairs = {}
    duplicate_media_rows = 0
    for key in sorted(strata):
        count = 0
        ranked = sorted(strata[key], key=lambda row: (canonical_hash([seed, row[0]]), row[0]))
        for source_id, index, raw, relative in ranked:
            if count == quotas[key]:
                break
            text = raw["text"]
            row_hash = canonical_hash(raw)
            row = {
                "source_id": source_id, "text": text, "images": [relative], "split": source_split,
                "source_veracity": raw["gt_answers"], "redistribution_status": "restricted",
                "target_claims": [{"claim_id": "caption", "text": text,
                                   "text_span": {"start": 0, "end": len(text)}}],
                "source_row_sha256": row_hash,
            }
            reference = {"source_id": source_id, "source_row": index + 1, "source_row_sha256": row_hash}
            try:
                record = normalize(row, source, media_root, is_mock=False)
                if record["media"][0]["mime_type"] not in {"image/png", "image/jpeg", "image/webp"}:
                    raise ValueError("unsupported media format")
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                reason = str(exc)
                if isinstance(exc, OSError):
                    reason = exc.strerror or type(exc).__name__
                quarantine.append({**reference, "path": relative, "status": "QUARANTINED",
                                   "reason": reason})
                continue
            pair = (record["text_sha256"], record["media"][0]["sha256"])
            previous = selected_pairs.get(pair)
            if previous is not None:
                if previous != key:
                    raise ValueError(f"Conflicting MMFakeBench labels for identical image bytes and caption at row {index + 1}")
                duplicate_media_rows += 1
                continue
            selected_pairs[pair] = key
            rows.append((index + 1, record))
            selection.append({**reference, "sample_id": record["sample_id"]})
            count += 1
        if count < quotas[key]:
            raise ValueError(f"MMFakeBench stratum {key!r} yielded {count}/{quotas[key]} samples. "
                             f"Check the extracted assets and duplicates. Quarantined rows: {len(quarantine)}. "
                             f"Last media error: {quarantine[-1] if quarantine else 'none'}")
        counts.append({"source_veracity": key[0], "fake_cls": key[1] or None,
                       "population": len(strata[key]), "selected": count})

    manifest = {
        "adapter": "mmfakebench", "method": "proportional-sha256-v1", "seed": seed,
        "annotation_filename": path.name, "annotation_sha256": source_hash, "source_split": name,
        "assigned_split": source_split, "input_rows": len(raw_rows), "unique_source_rows": population,
        "duplicate_source_rows": len(raw_rows) - population, "duplicate_media_rows": duplicate_media_rows,
        "sample_size": sample_size, "strata": counts, "quarantine": quarantine, "selection": selection,
    }
    return rows, manifest
