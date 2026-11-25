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

combined="$output/$stage-combined.json"
pending=$(mktemp "$combined.XXXXXX")
trap 'rm -f -- "$pending"' EXIT
args=("$stage" --run-id mmfakebench-val --batch-id m2-val100-baseline --batch-offsets 0 25 50 75
      --reviewer-lock runs/m1-gemma/model_lock.json)
if [[ "$stage" == annotate ]]; then
  args+=(--limit 25 --annotator-lock runs/m1-qwen/model_lock.json)
fi
echo "Running $stage across four val batches in one inference queue"
python -m rumorlab "${args[@]}" > "$pending" || exit_status=$?
if [[ ! -s "$pending" ]]; then
  exit "$exit_status"
fi
python - "$output" "$stage" "$pending" <<'PY'
import json
from pathlib import Path
import sys

output, stage, pending = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
combined = json.loads(pending.read_text(encoding="utf-8"))
pending.replace(output / f"{stage}-combined.json")
print(combined["run_timings"])
for report in combined["batches"]:
    offset = report["batch_id"].rsplit("-", 1)[1]
    (output / f"{stage}-{offset}.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(report["batch_id"], {key: report[key] for key in ("samples", "workers", "failed", "elapsed_seconds")})
PY

exit "$exit_status"
