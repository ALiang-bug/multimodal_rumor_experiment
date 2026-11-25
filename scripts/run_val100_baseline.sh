#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

stage=${1:-}
case "$stage" in
  annotate) export ANNOTATOR_BASE_URL=${ANNOTATOR_BASE_URL:-http://127.0.0.1:8000/v1} ;;
  review-labels) export REVIEWER_BASE_URL=${REVIEWER_BASE_URL:-http://127.0.0.1:8000/v1} ;;
  *) echo "Usage: bash scripts/run_val100_baseline.sh {annotate|review-labels}" >&2; exit 2 ;;
esac

export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
output=runs/m2-val100-baseline
mkdir -p "$output"
exit_status=0

if [[ "$stage" == annotate ]]; then
  combined="$output/annotate-combined.json"
  echo "Preparing all 100 val samples for one inference queue"
  python -m rumorlab annotate --run-id mmfakebench-val \
    --batch-id m2-val100-baseline --batch-offsets 0 25 50 75 --limit 25 \
    --annotator-lock runs/m1-qwen/model_lock.json --reviewer-lock runs/m1-gemma/model_lock.json \
    > "$combined" || exit_status=$?
  if [[ ! -s "$combined" ]]; then
    exit "$exit_status"
  fi
  python - "$output" <<'PY'
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
combined = json.loads((output / "annotate-combined.json").read_text(encoding="utf-8"))
print(combined["run_timings"])
for report in combined["batches"]:
    offset = report["batch_id"].rsplit("-", 1)[1]
    (output / f"annotate-{offset}.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(report["batch_id"], {key: report[key] for key in ("samples", "workers", "failed")})
PY
  exit "$exit_status"
fi

for offset in 0 25 50 75; do
  batch_id="m2-val100-baseline-$offset"
  report="$output/$stage-$offset.json"
  args=("$stage" --run-id mmfakebench-val --batch-id "$batch_id"
        --reviewer-lock runs/m1-gemma/model_lock.json)
  echo "Running $stage for $batch_id"
  if python -m rumorlab "${args[@]}" > "$report"; then
    :
  else
    status=$?
    if [[ "$status" != 1 || ! -s "$report" ]]; then
      exit "$status"
    fi
    exit_status=1
  fi
  python -c 'import json, sys; r = json.load(open(sys.argv[1], encoding="utf-8")); print({k: r[k] for k in ("samples", "workers", "failed", "elapsed_seconds")})' "$report"
done

exit "$exit_status"
