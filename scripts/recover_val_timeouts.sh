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

lock_revision() {
  "$PYTHON" -c 'import sys; from pathlib import Path; from rumorlab.config import Contract; from rumorlab.m1 import load_lock; print(load_lock(Contract(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))[0]["revision"])' \
    "$CFG" "$1" "$2"
}
QWEN="$MMFND/models/models--Qwen--Qwen3.8-27B/snapshots/$(lock_revision annotator "$QLOCK")"
GEMMA="$MMFND/models/models--google--gemma-4-31B-it/snapshots/$(lock_revision reviewer "$GLOCK")"
test -f "$QWEN/config.json"
test -f "$GEMMA/config.json"

server_pid=
stop_server() {
  local pid="${server_pid:-}" i
  [[ -n "$pid" ]] || return 0
  if kill -TERM -- "-$pid" 2>/dev/null; then
    for ((i=0; i<60; i++)); do
      kill -0 -- "-$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  server_pid=
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_server() {
  local snapshot="$1" model="$2" parser="$3" tag="$4" i
  local log="$ATTEMPT/$tag-server.log"
  "$PYTHON" -c 'import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(("127.0.0.1",8000)); s.close()'
  printf 'Starting %s, log %s\n' "$model" "$log"
  setsid "$BIN/vllm" serve "$snapshot" --served-model-name "$model" \
    --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 --dtype bfloat16 \
    --max-model-len 12288 --max-num-seqs 64 --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image":2}' --reasoning-parser "$parser" >>"$log" 2>&1 &
  server_pid=$!
  for ((i=0; i<180; i++)); do
    kill -0 "$server_pid" 2>/dev/null || break
    if curl -fsS --max-time 2 "$URL/models" 2>/dev/null |
      "$PYTHON" -c 'import json,sys; sys.exit(not any(x["id"] == sys.argv[1] for x in json.load(sys.stdin)["data"]))' "$model" 2>/dev/null; then
      echo "Ready $model"
      return 0
    fi
    sleep 5
  done
  tail -n 40 "$log" >&2
  echo "vLLM failed to become ready: $model" >&2
  return 1
}

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
