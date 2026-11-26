#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

BASELINE="${1:?Usage: bash scripts/recover_val_timeouts.sh ORIGINAL_ATTEMPT_DIR}"
MMFND="$(cd .. && pwd)"
BIN="$MMFND/.envs/rumorlab-m1/bin"
PYTHON="$BIN/python"
CFG=configs/experiment-single-label.yaml
OUT=runs/m2-val1000-single-label
DB=${RUMORLAB_STATE_DB:-"$OUT/state.sqlite3"}
RUN=mmfakebench-val
QLOCK=runs/m1-qwen/model_lock.json
GLOCK=runs/m1-gemma/model_lock.json
URL=http://127.0.0.1:8000/v1
export ANNOTATOR_BASE_URL="$URL" REVIEWER_BASE_URL="$URL"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

test -f "$DB"
test -s "$BASELINE/annotate.json"
test -s "$BASELINE/review-labels.json"
printf 'Recovery state database %s\n' "$DB"
ATTEMPT=$(mktemp -d "$OUT/recovery-$(date +%Y%m%d-%H%M%S)-XXXXXX")
printf 'Recovery logs and reports %s\n' "$PWD/$ATTEMPT"

source scripts/model_servers.sh

run_stage() {
  local stage="$1" workers="$2" rc=0
  local report="$ATTEMPT/$stage.json"
  printf 'Recovering %s with %s workers and a 360-second timeout\n' "$stage" "$workers"
  "$PYTHON" -m rumorlab recover-timeouts --stage "$stage" --workers "$workers" \
    --config "$CFG" --run-id "$RUN" --state-db "$DB" --attempt-dir "$BASELINE" \
    --annotator-lock "$QLOCK" --reviewer-lock "$GLOCK" >"$report.tmp" || rc=$?
  "$PYTHON" - "$report.tmp" "$stage" "$rc" <<'PY'
import json
from pathlib import Path
import sys

path, stage, rc = sys.argv[1:]
report = json.loads(Path(path).read_text(encoding="utf-8"))
if report["stage"] != stage.replace("-", "_") or int(rc) != int(bool(report["failed"])):
    raise SystemExit(f"Recovery did not produce a complete report: {path}")
print(f"{stage}: {report['original_timeouts']} original timeouts, {report['new_candidates']} new candidates, "
      f"{report['selected'] - report['failed']} completed, {report['cached']} cached, {report['failed']} failed")
PY
  mv -- "$report.tmp" "$report"
}

start_server "$QWEN" Qwen/Qwen3.8-27B qwen3 annotator
run_stage annotate 48
stop_server

start_server "$GEMMA" google/gemma-4-31B-it gemma4 reviewer
run_stage review-labels 16
stop_server

"$PYTHON" -m rumorlab export-snapshot --config "$CFG" --state-db "$DB" --run-id "$RUN" \
  >"$ATTEMPT/snapshot.jsonl.tmp"
mv -- "$ATTEMPT/snapshot.jsonl.tmp" "$ATTEMPT/snapshot.jsonl"
printf 'Done. Recovery reports and updated snapshot %s\n' "$PWD/$ATTEMPT"
