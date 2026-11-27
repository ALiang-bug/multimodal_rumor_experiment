#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

MMFND="$(cd .. && pwd)"
BIN="$MMFND/.envs/rumorlab-m1/bin"
PYTHON="$BIN/python"
CFG=configs/experiment-single-label-v1.3.yaml
QLOCK=runs/m1-qwen/model_lock.json
GLOCK=runs/m1-gemma/model_lock.json
URL=http://127.0.0.1:8000/v1
export ANNOTATOR_BASE_URL="$URL" REVIEWER_BASE_URL="$URL"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

TEST_SIZE="${1:-3000}"
case "$TEST_SIZE" in
  2000|3000) ;;
  *) echo "Usage: bash scripts/build_mmfakebench_datasets.sh [2000|3000] [media-root]" >&2; exit 2 ;;
esac
MEDIA_ROOT="${2:-$MMFND/MAR_KDD/dataset/MMFakeBench}"
VAL_SOURCE="${MMFAKEBENCH_VAL_JSON:-$MEDIA_ROOT/MMFakeBench_val.json}"
TEST_SOURCE="${MMFAKEBENCH_TEST_JSON:-$MEDIA_ROOT/MMFakeBench_test.json}"
if [[ ! -f "$VAL_SOURCE" && -z "${MMFAKEBENCH_VAL_JSON:-}" ]]; then
  VAL_SOURCE="$MEDIA_ROOT/MMFakeBench_val/source/MMFakeBench_val.json"
fi
if [[ ! -f "$TEST_SOURCE" && -z "${MMFAKEBENCH_TEST_JSON:-}" ]]; then
  TEST_SOURCE="$MEDIA_ROOT/MMFakeBench_test/source/MMFakeBench_test.json"
fi
for input in "$VAL_SOURCE" "$TEST_SOURCE" "$QLOCK" "$GLOCK"; do
  test -f "$input" || { echo "Missing input: $input" >&2; exit 2; }
done
test -x "$PYTHON"

PROJECT="mmfakebench-v1.3-test$TEST_SIZE"
VAL_RUN="$PROJECT-val"
TEST_RUN="$PROJECT-test"
OUT="runs/$PROJECT"
WORK="${RUMORLAB_BUILD_WORK_DIR:-/var/tmp/rumorlab-${UID}/$PROJECT}"
mkdir -p -- "$OUT" "$WORK"
case "$(stat -f -c %T "$WORK")" in
  nfs|cifs|smb2) echo "RUMORLAB_BUILD_WORK_DIR must be on a local filesystem" >&2; exit 2 ;;
esac
exec 9>"$WORK/run.lock"
flock -n 9 || { echo "Another dataset build is using $WORK" >&2; exit 2; }
DB="$WORK/state.sqlite3"
ATTEMPT="$(mktemp -d "$OUT/attempt-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
mkdir -- "$ATTEMPT/val" "$ATTEMPT/test"
printf 'Reports %s\nWorking database %s\n' "$PWD/$ATTEMPT" "$DB"

lab() {
  "$PYTHON" -m rumorlab "$@" --config "$CFG" --state-db "$DB"
}

lab ingest --adapter mmfakebench --run-id "$VAL_RUN" --records "$VAL_SOURCE" \
  --data-root "$MEDIA_ROOT" >"$ATTEMPT/val/ingest.json"
lab ingest --adapter mmfakebench --run-id "$TEST_RUN" --records "$TEST_SOURCE" \
  --data-root "$MEDIA_ROOT" --sample-size "$TEST_SIZE" >"$ATTEMPT/test/ingest.json"
"$PYTHON" - "$ATTEMPT" "$TEST_SIZE" <<'PY'
import json
from pathlib import Path
import sys

attempt, test_size = Path(sys.argv[1]), int(sys.argv[2])
for split, size, official in (("val", 1000, "MMFakeBench_val"), ("test", test_size, "MMFakeBench_test")):
    report = json.loads((attempt / split / "ingest.json").read_text(encoding="utf-8"))
    sampling = report["sampling"]
    if report["records"] != size or sampling["source_split"] != official:
        raise SystemExit(f"Expected {size} rows from {official}")
    print(f"{split}: {size} frozen samples, method={sampling['method']}, seed={sampling['seed']}")
PY

source scripts/model_servers.sh

run_stage() {
  local stage="$1" split="$2" run="$3" size="$4" workers="$5" rc=0 i
  shift 5
  local report="$ATTEMPT/$split/$stage.json"
  local offsets=()
  for ((i=0; i<size; i+=25)); do offsets+=("$i"); done
  printf '%s %s, %s workers with same-model timeout recovery\n' "$split" "$stage" "$workers"
  lab "$stage" --run-id "$run" --batch-id "$run" --batch-offsets "${offsets[@]}" \
    --prefetch-group-size 4 --workers "$workers" --reviewer-lock "$GLOCK" --recover-timeouts \
    "$@" >"$report.tmp" || rc=$?
  "$PYTHON" - "$report.tmp" "$stage" "$size" "$rc" <<'PY'
import json
from pathlib import Path
import sys

path, stage, size, rc = sys.argv[1:]
report = json.loads(Path(path).read_text(encoding="utf-8"))
size = int(size)
if (report["stage"] != stage.replace("-", "_") or len(report["batches"]) != size // 25
        or sum(batch["samples"] for batch in report["batches"]) != size
        or int(rc) != int(bool(report["failed"]))):
    raise SystemExit(f"Incomplete {stage} report: {path}")
print(f"{stage}: {report.get('timeout_recovery', {}).get('selected', 0)} timeouts retried, "
      f"{report['failed']} remaining failures")
PY
  mv -- "$report.tmp" "$report"
}

start_server "$QWEN" Qwen/Qwen3.8-27B qwen3 annotator
run_stage annotate val "$VAL_RUN" 1000 48 --limit 25 --annotator-lock "$QLOCK"
run_stage annotate test "$TEST_RUN" "$TEST_SIZE" 48 --limit 25 --annotator-lock "$QLOCK"
stop_server

start_server "$GEMMA" google/gemma-4-31B-it gemma4 reviewer
run_stage review-labels val "$VAL_RUN" 1000 16
run_stage review-labels test "$TEST_RUN" "$TEST_SIZE" 16
stop_server

for split in val test; do
  run="$VAL_RUN"
  [[ "$split" == val ]] || run="$TEST_RUN"
  lab export-snapshot --run-id "$run" >"$ATTEMPT/$split/snapshot.jsonl.tmp"
  mv -- "$ATTEMPT/$split/snapshot.jsonl.tmp" "$ATTEMPT/$split/snapshot.jsonl"
done
"$PYTHON" -m rumorlab.dataset_bundle "$ATTEMPT" "$DB" "$MEDIA_ROOT" "$VAL_SOURCE" "$TEST_SOURCE" "$VAL_RUN" "$TEST_RUN"
printf 'Datasets and statistics in %s\n' "$PWD/$ATTEMPT"
