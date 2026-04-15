#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

STATUS="run_logs/switch_after_baseline_top10_to_remaining_20260409.status"
LOG="run_logs/switch_after_baseline_top10_to_remaining_20260409.log"
PID_FILE="run_logs/switch_after_baseline_top10_to_remaining_20260409.pid"

QUEUE_PID_FILE="run_logs/fullscale_width_matched_control_20260409fullfix.pid"
TARGET_JSON="outputs_step0_general_musique/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260409fullfix.json"
REMAINING_SCRIPT="run_logs/fullscale_width_matched_control_musique_top5_remaining_20260409.sh"
REMAINING_STATUS="run_logs/fullscale_width_matched_control_musique_top5_remaining_20260409.status"
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
path = Path("outputs_step0_general_musique/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260409fullfix.json")
data = json.loads(path.read_text(encoding="utf-8"))
method = data.get("expand_assemble_qa", {}) or {}
num_queries = (data.get("overall_recomputed", {}) or {}).get("num_queries")
if int(num_queries or 0) <= 20:
    raise SystemExit(f"unexpected num_queries={num_queries}")
if method.get("method_EM") is None:
    raise SystemExit("missing method_EM")
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

start_remaining_queue() {
  if [[ -f "$REMAINING_STATUS" ]] && grep -q '^DONE$' "$REMAINING_STATUS"; then
    log "Remaining queue already DONE; nothing to launch"
    return 0
  fi
  log "Launching remaining queue script=${REMAINING_SCRIPT}"
  nohup bash "$REMAINING_SCRIPT" >/dev/null 2>&1 &
  log "Remaining queue launched pid=$!"
}

trap 'code=$?; log "FAILED exit=${code}"; exit $code' ERR

: > "$LOG"
printf '%s\n' "$$" > "$PID_FILE"
echo "RUNNING wait_for_top10" > "$STATUS"
log "Watcher started poll_seconds=${POLL_SECONDS}"

while true; do
  if [[ -f "$TARGET_JSON" ]]; then
    break
  fi
  if ! queue_alive; then
    log "Long queue is not alive before target JSON appeared"
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

echo "RUNNING validate_top10" > "$STATUS"
num_queries="$(validate_json)"
log "Detected baseline_top10 JSON with num_queries=${num_queries}"

echo "RUNNING stop_long_queue" > "$STATUS"
stop_long_queue
sleep 2

echo "RUNNING launch_remaining_queue" > "$STATUS"
start_remaining_queue

echo "DONE" > "$STATUS"
log "Watcher finished"
