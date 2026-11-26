#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

MMFND="$(cd .. && pwd)"
BIN="$MMFND/.envs/rumorlab-m1/bin"
PYTHON="$BIN/python"
CFG=configs/experiment-single-label.yaml
RUN=mmfakebench-test-first500
OUT="runs/$RUN"
WORK="${RUMORLAB_TEST_WORK_DIR:-/var/tmp/rumorlab-${UID}/$RUN}"
VAL_DB="${RUMORLAB_VAL_SOURCE_DB:-runs/state.sqlite3}"
QLOCK=runs/m1-qwen/model_lock.json
GLOCK=runs/m1-gemma/model_lock.json
URL=http://127.0.0.1:8000/v1
export ANNOTATOR_BASE_URL="$URL" REVIEWER_BASE_URL="$URL"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

MEDIA_ROOT="${1:-$MMFND/MAR_KDD/dataset/MMFakeBench}"
SOURCE="${2:-$MEDIA_ROOT/MMFakeBench_test.json}"
if [[ $# -lt 2 && ! -f "$SOURCE" ]]; then
  SOURCE="$MEDIA_ROOT/MMFakeBench_test/source/MMFakeBench_test.json"
fi
test -f "$SOURCE" || { echo "Official test JSON not found: $SOURCE" >&2; exit 2; }
test -f "$VAL_DB" || { echo "Original val import database not found: $VAL_DB" >&2; exit 2; }
mkdir -p -- "$OUT" "$WORK"
case "$(stat -f -c %T "$WORK")" in
  nfs|cifs|smb2) echo "Set RUMORLAB_TEST_WORK_DIR to a local filesystem" >&2; exit 2 ;;
esac
exec 9>"$WORK/run.lock"
flock -n 9 || { echo "An existing run holds $WORK/run.lock" >&2; exit 2; }
DB="$WORK/state.sqlite3"
ATTEMPT="$(mktemp -d "$OUT/attempt-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
printf 'Reports %s\nWorking database %s\n' "$PWD/$ATTEMPT" "$DB"

"$PYTHON" -m rumorlab ingest --adapter mmfakebench --first-n 500 --records "$SOURCE" \
  --data-root "$MEDIA_ROOT" --config "$CFG" --state-db "$DB" --run-id "$RUN" \
  >"$ATTEMPT/ingest.json"

# Freeze the cohort and flag exact val overlap for exclusion from diagnostic scoring.
"$PYTHON" - "$DB" "$VAL_DB" "$RUN" "$ATTEMPT" <<'PY'
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

db_path, val_path, run_id, attempt = sys.argv[1:]

def read_import(path, run):
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute(
            "SELECT records, manifest FROM imports JOIN import_manifests USING(run_id) WHERE run_id = ?", (run,),
        ).fetchone()
    if row is None:
        raise SystemExit(f"Missing frozen import {run} in {path}")
    records, manifest = map(json.loads, row)
    return {record["sample_id"]: record for record in records if record["record_type"] == "sample"}, manifest

test, manifest = read_import(db_path, run_id)
val, val_manifest = read_import(val_path, "mmfakebench-val")
if len(test) != 500 or [row["source_row"] for row in manifest["selection"]] != list(range(1, 501)):
    raise SystemExit("Expected the exact official test rows 1–500")
if len(val) != 1000 or val_manifest["source_split"] != "MMFakeBench_val":
    raise SystemExit("Expected the full official val import for overlap checks")
val_text = {row["text_sha256"] for row in val.values()}
val_media = {media["sha256"] for row in val.values() for media in row["media"]}
overlap = []
for item in manifest["selection"]:
    row = test[item["sample_id"]]
    same_text = row["text_sha256"] in val_text
    same_media = any(media["sha256"] in val_media for media in row["media"])
    if same_text or same_media:
        overlap.append({**item, "same_text": same_text, "same_media": same_media})
for name, data in (
    ("selection.json", manifest),
    ("val-overlap.json", {"val_annotation_sha256": val_manifest["annotation_sha256"],
                          "method": "exact-text-or-image-sha256",
                          "excluded_from_scoring": overlap, "remaining_before_review": 500 - len(overlap)}),
):
    (Path(attempt) / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Frozen test: 500 rows. Exact val overlaps flagged for scoring exclusion: {len(overlap)}")
PY

source scripts/model_servers.sh
offsets=()
for ((i=0; i<500; i+=25)); do offsets+=("$i"); done

run_stage() {
  local stage="$1" workers="$2" rc=0
  shift 2
  local report="$ATTEMPT/$stage.json"
  printf 'Running %s with %s workers and same-model timeout recovery\n' "$stage" "$workers"
  "$PYTHON" -m rumorlab "$stage" --config "$CFG" --state-db "$DB" --run-id "$RUN" \
    --batch-id "$RUN" --batch-offsets "${offsets[@]}" --prefetch-group-size 4 \
    --workers "$workers" --reviewer-lock "$GLOCK" --recover-timeouts "$@" >"$report.tmp" || rc=$?
  "$PYTHON" - "$report.tmp" "$stage" "$rc" <<'PY'
import json
from pathlib import Path
import sys

path, stage, rc = sys.argv[1:]
report = json.loads(Path(path).read_text(encoding="utf-8"))
batches = report["batches"]
if (report["stage"] != stage.replace("-", "_") or len(batches) != 20
        or sum(batch["samples"] for batch in batches) != 500 or int(rc) != int(bool(report["failed"]))):
    raise SystemExit(f"Incomplete {stage} report: {path}")
recovery = report.get("timeout_recovery", {})
print(f"{stage}: {sum(len(batch['results']) for batch in batches)} records, "
      f"{recovery.get('selected', 0)} terminal timeouts retried, {report['failed']} remaining failures")
PY
  mv -- "$report.tmp" "$report"
}

start_server "$QWEN" Qwen/Qwen3.8-27B qwen3 annotator
run_stage annotate 48 --limit 25 --annotator-lock "$QLOCK"
stop_server

start_server "$GEMMA" google/gemma-4-31B-it gemma4 reviewer
run_stage review-labels 16
stop_server

"$PYTHON" -m rumorlab export-snapshot --config "$CFG" --state-db "$DB" --run-id "$RUN" \
  >"$ATTEMPT/snapshot.jsonl.tmp"
mv -- "$ATTEMPT/snapshot.jsonl.tmp" "$ATTEMPT/snapshot.jsonl"
printf 'Done. Test cohort, overlap report, model reports and snapshot in %s\n' "$PWD/$ATTEMPT"
