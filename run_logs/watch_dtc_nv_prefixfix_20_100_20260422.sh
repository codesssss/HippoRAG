#!/usr/bin/env bash
set -euo pipefail

log_path="run_logs/dtc_nv_prefixfix_20_100_20260422.log"
pid_path="run_logs/dtc_nv_prefixfix_20_100_20260422.pid"

while true; do
  if [[ -f "${pid_path}" ]]; then
    pid="$(cat "${pid_path}")"
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[$(date '+%F %T')] main process ${pid} finished"
      tail -n 80 "${log_path}" 2>/dev/null || true
      exit 0
    fi
  fi

  echo "[$(date '+%F %T')] still running"
  grep -E "START|DONE|ALL_DONE|setwise_selector_qa|output_json|Retrieving:" "${log_path}" 2>/dev/null | tail -n 40 || true
  sleep 300
done
