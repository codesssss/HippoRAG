#!/usr/bin/env bash
set -euo pipefail

main_pid_file="run_logs/dtc_nv_soft1000_20260422.pid"
main_log="run_logs/dtc_nv_soft1000_20260422.log"
watch_log="run_logs/dtc_nv_soft1000_20260422.watch.log"
run_script="run_logs/run_dtc_nv_soft1000_20260422.sh"

expected_outputs=(
  "outputs_step0_general_nvembed_2wikimultihopqa/eval_reports/dtc_embed_nvembed_soft_limit1000_anchor2_fresh8043.json"
  "outputs_step0_general_nvembed_hotpotqa/eval_reports/dtc_embed_nvembed_soft_limit1000_anchor2_fresh8043.json"
  "outputs_step0_general_nvembed_musique/eval_reports/dtc_embed_nvembed_soft_limit1000_anchor2_fresh8043.json"
)

count_done() {
  local count=0
  local path
  for path in "${expected_outputs[@]}"; do
    if [[ -s "${path}" ]]; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

log_status() {
  local done_count="$1"
  local parent_pid="${2:-}"
  {
    echo "[$(date '+%F %T')] WATCH done=${done_count}/${#expected_outputs[@]} parent=${parent_pid:-none}"
    if [[ -n "${parent_pid}" ]] && kill -0 "${parent_pid}" 2>/dev/null; then
      ps -o pid,ppid,stat,etime,cmd -p "${parent_pid}" || true
      ps --ppid "${parent_pid}" -o pid,ppid,stat,etime,cmd || true
    fi
    grep -E 'START|DONE|ALL_DONE|Traceback|ERROR|Error|Exception' "${main_log}" 2>/dev/null | tail -30 || true
    echo
  } >> "${watch_log}"
}

restart_count=0
while true; do
  done_count="$(count_done)"
  parent_pid=""
  if [[ -s "${main_pid_file}" ]]; then
    parent_pid="$(cat "${main_pid_file}")"
  fi

  log_status "${done_count}" "${parent_pid}"

  if [[ "${done_count}" -eq "${#expected_outputs[@]}" ]]; then
    echo "[$(date '+%F %T')] WATCH_ALL_DONE" >> "${watch_log}"
    exit 0
  fi

  if [[ -n "${parent_pid}" ]] && kill -0 "${parent_pid}" 2>/dev/null; then
    sleep 600
    continue
  fi

  if pgrep -f 'scripts/eval_causal_qwen3.py .*outputs_step0_general_nvembed' >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] WATCH parent dead but eval child exists; waiting" >> "${watch_log}"
    sleep 600
    continue
  fi

  if [[ "${restart_count}" -ge 2 ]]; then
    echo "[$(date '+%F %T')] WATCH giving up after ${restart_count} restarts" >> "${watch_log}"
    exit 2
  fi

  restart_count=$((restart_count + 1))
  echo "[$(date '+%F %T')] WATCH restarting missing outputs attempt=${restart_count}" >> "${watch_log}"
  setsid "${run_script}" >> "${main_log}" 2>&1 &
  echo "$!" > "${main_pid_file}"
  sleep 600
done
