#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

STATUS="run_logs/switch_to_priority_width_matched_control_20260409.status"
LOG="run_logs/switch_to_priority_width_matched_control_20260409.log"
PID_FILE="run_logs/switch_to_priority_width_matched_control_20260409.pid"

QUEUE_PID_FILE="run_logs/fullscale_width_matched_control_20260409fullfix.pid"
QUEUE_STATUS="run_logs/fullscale_width_matched_control_20260409fullfix.status"
BASELINE_JSON="outputs_step0_general_musique/eval_reports/causal_eval_musique_full_qwen3-8b_qatopk5_baseline_20260409fullfix.json"
PRIORITY_SCRIPT="run_logs/fullscale_width_matched_control_musique_top5_priority_20260409.sh"
PRIORITY_STATUS="run_logs/fullscale_width_matched_control_musique_top5_priority_20260409.status"
POLL_SECONDS="${POLL_SECONDS:-30}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"
}

queue_alive() {
  [[ -f "$QUEUE_PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$QUEUE_PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

validate_json() {
  .venv-hipporag/bin/python - <<'PY'
import json
from pathlib import Path
path = Path("outputs_step0_general_musique/eval_reports/causal_eval_musique_full_qwen3-8b_qatopk5_baseline_20260409fullfix.json")
data = json.loads(path.read_text(encoding="utf-8"))
overall = data.get("overall_recomputed", {}) or {}
num_queries = overall.get("num_queries")
if int(num_queries or 0) <= 20:
    raise SystemExit(f"unexpected num_queries={num_queries}")
print(num_queries)
PY
}

stop_long_queue() {
  [[ -f "$QUEUE_PID_FILE" ]] || return 0
  local pid
  pid="$(cat "$QUEUE_PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  local pgid
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ' || true)"
  if [[ -n "$pgid" ]]; then
    log "Stopping long queue pid=${pid} pgid=${pgid}"
    kill -TERM -- "-$pgid" 2>/dev/null || true
  else
    log "Stopping long queue pid=${pid}"
    kill -TERM "$pid" 2>/dev/null || true
  fi
}

start_priority_queue() {
  if [[ -f "$PRIORITY_STATUS" ]] && grep -q '^DONE$' "$PRIORITY_STATUS"; then
    log "Priority queue already DONE; nothing to launch"
    return 0
  fi
  log "Launching priority queue script=${PRIORITY_SCRIPT}"
  nohup bash "$PRIORITY_SCRIPT" >/dev/null 2>&1 &
  log "Priority queue launched pid=$!"
}

trap 'code=$?; log "FAILED exit=${code}"; exit $code' ERR

: > "$LOG"
printf '%s\n' "$$" > "$PID_FILE"
echo "RUNNING wait_for_baseline" > "$STATUS"
log "Watcher started poll_seconds=${POLL_SECONDS}"

while true; do
  if [[ -f "$BASELINE_JSON" ]]; then
    break
  fi
  if ! queue_alive; then
    log "Long queue is not alive before baseline JSON appeared"
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

echo "RUNNING validate_baseline" > "$STATUS"
num_queries="$(validate_json)"
log "Detected baseline JSON with num_queries=${num_queries}"

echo "RUNNING stop_long_queue" > "$STATUS"
stop_long_queue
sleep 2

echo "RUNNING launch_priority_queue" > "$STATUS"
start_priority_queue

echo "DONE" > "$STATUS"
log "Watcher finished"
