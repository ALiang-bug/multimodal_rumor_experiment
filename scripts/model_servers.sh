# Shared single-GPU lifecycle. Callers set MMFND, BIN, PYTHON, CFG, QLOCK, GLOCK, URL and ATTEMPT.
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
