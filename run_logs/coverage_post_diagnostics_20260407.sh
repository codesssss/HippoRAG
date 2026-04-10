#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

QUEUE_STATUS="run_logs/coverage_assembly_matrix_20260407.status"
QUEUE_PID_FILE="run_logs/coverage_assembly_matrix_20260407.pid"

STATUS="run_logs/coverage_post_diagnostics_20260407.status"
LOG="run_logs/coverage_post_diagnostics_20260407.log"
PID_FILE="run_logs/coverage_post_diagnostics_20260407.pid"
OUTPUT_JSON="run_logs/coverage_post_diagnostics_20260407.json"
OUTPUT_MD="run_logs/coverage_post_diagnostics_20260407.md"

PY=".venv-hipporag/bin/python"
SCRIPT="scripts/analyze_coverage_postrun.py"

BASELINE_DATE_TAG="20260406"
NEW_DATE_TAG="20260407"
POLL_SECONDS="${POLL_SECONDS:-60}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"
}

queue_done() {
  [[ -f "$QUEUE_STATUS" ]] && grep -q '^DONE' "$QUEUE_STATUS"
}

queue_alive() {
  if [[ ! -f "$QUEUE_PID_FILE" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$QUEUE_PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  ps -p "$pid" >/dev/null 2>&1
}

run_diagnostics() {
  echo "RUNNING diagnostics" > "$STATUS"
  log "START diagnostics output_json=${OUTPUT_JSON}"
  "$PY" "$SCRIPT" \
    --baseline_date_tag "$BASELINE_DATE_TAG" \
    --new_date_tag "$NEW_DATE_TAG" \
    --output_json "$OUTPUT_JSON" \
    --output_md "$OUTPUT_MD" >> "$LOG" 2>&1
  log "DONE diagnostics output_json=${OUTPUT_JSON} output_md=${OUTPUT_MD}"
}

trap 'code=$?; log "FAILED exit=${code} while waiting or running diagnostics"; exit $code' ERR

printf '%s\n' "$$" > "$PID_FILE"
: > "$LOG"
echo "RUNNING wait_for_queue" > "$STATUS"
log "Watcher started poll_seconds=${POLL_SECONDS}"

while true; do
  if queue_done; then
    log "Detected queue completion from ${QUEUE_STATUS}"
    break
  fi
  if ! queue_alive; then
    log "Queue PID is not alive but status is not DONE; running diagnostics on partial results"
    break
  fi
  sleep "$POLL_SECONDS"
done

run_diagnostics
echo "DONE" > "$STATUS"
log "Watcher finished"
