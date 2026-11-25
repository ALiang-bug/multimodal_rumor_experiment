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

for offset in 0 25 50 75; do
  batch_id="m2-val100-baseline-$offset"
  report="$output/$stage-$offset.json"
  args=("$stage" --run-id mmfakebench-val --batch-id "$batch_id"
        --reviewer-lock runs/m1-gemma/model_lock.json)
  if [[ "$stage" == annotate ]]; then
    args+=(--offset "$offset" --limit 25 --annotator-lock runs/m1-qwen/model_lock.json)
  fi
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
