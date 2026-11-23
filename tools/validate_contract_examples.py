#!/usr/bin/env python3
"""Validate this design package's JSON examples, not an experiment pipeline.

Dependencies: Python standard library, jsonschema and PyYAML.
Default validation is closed-world: referenced records must be in the JSONL.
Use --check-media after the real fixture image and its hashes are installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    import jsonschema
    import yaml
except ImportError as exc:
    print(f"Missing validation dependency: {exc}", file=sys.stderr)
    sys.exit(2)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rumorlab.validation import validate_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "examples/records.jsonl")
    parser.add_argument("--schema", type=Path, default=ROOT / "schemas/records.schema.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "configs/labels.yaml")
    parser.add_argument("--experiment-config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--check-media", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []

    def fail(message: str) -> None:
        errors.append(message)

    try:
        schema = json.loads(args.schema.read_text(encoding="utf-8"))
        labels = yaml.safe_load(args.labels.read_text(encoding="utf-8"))
        experiment = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8")) or {}
        gold_votes_required = experiment.get("human", {}).get("gold_independent_votes", 3)
        if isinstance(gold_votes_required, bool) or not isinstance(gold_votes_required, int) or gold_votes_required < 2:
            raise ValueError("human.gold_independent_votes must be an integer >= 2; default is 3")
        jsonschema.Draft202012Validator.check_schema(schema)
        rows = []
        for line_no, line in enumerate(args.records.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append((line_no, json.loads(line)))
                except json.JSONDecodeError as exc:
                    fail(f"line {line_no}: invalid JSON: {exc}")
    except (OSError, ValueError, jsonschema.SchemaError, yaml.YAMLError) as exc:
        print(f"Cannot load contract inputs: {exc}", file=sys.stderr)
        return 2

    violations, checked_paths = validate_rows(
        rows, schema, labels, gold_votes_required, args.data_root,
        check_media=args.check_media,
    )
    errors.extend(violations)

    if errors:
        for message in errors:
            print(f"ERROR: {message}", file=sys.stderr)
        print(f"FAILED: {len(errors)} contract violation(s).", file=sys.stderr)
        return 1
    print(f"PASS: {len(rows)} records; JSON Schema and example cross-record constraints valid.")
    print(f"Gold independent human votes required by configuration: {gold_votes_required}.")
    print(f"Media bytes checked: {len(checked_paths)} file(s)." if args.check_media
          else "Media byte validation was not requested; use --check-media after filling fixture hashes.")
    print("This validates design examples only; no model inference, human audit or experiment was performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
