#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
METHOD="evidence_transition_graphragv4_fact_witnessed_sto"
GATE_ROOT="${ROOT_DIR}/run_logs/etv4_clean_vs_no_multi_anchor_full1000_20260511"
CLEAN_ROOT="${ROOT_DIR}/run_logs/etv4_clean_mainline_full1000_reuse_gpt4omini_20260511"
ABL_ROOT="${ROOT_DIR}/run_logs/etv4_no_multi_anchor_precision_full1000_reuse_gpt4omini_20260511"

report_path() {
  local out_root="$1"
  local dataset="$2"
  printf '%s/%s/reports/%s_%s_retrieval.json' "${out_root}" "${dataset}" "${dataset}" "${METHOD}"
}

short_policy() {
  case "$1" in
    clean_mainline) printf 'clean' ;;
    no_multi_anchor_precision) printf 'nomulti' ;;
    *) printf '%s' "$1" | tr -cd '[:alnum:]_' ;;
  esac
}

print_row() {
  local policy="$1"
  local dataset="$2"
  local out_root="$3"
  local short
  short="$(short_policy "${policy}")"
  local session="etv4_f1000_${short}_${dataset}_20260511"
  local status_path="${GATE_ROOT}/status/${short}_${dataset}.status"
  local log_path="${GATE_ROOT}/logs/${short}_${dataset}.log"
  local report
  report="$(report_path "${out_root}" "${dataset}")"
  local session_state="missing"
  if tmux has-session -t "${session}" 2>/dev/null; then
    local dead
    dead="$(tmux list-panes -t "${session}" -F '#{pane_dead}' 2>/dev/null | head -n 1 || true)"
    if [[ "${dead}" == "1" ]]; then
      session_state="exited"
    else
      session_state="running"
    fi
  fi
  local report_state="no"
  local report_size="-"
  if [[ -f "${report}" ]]; then
    report_state="yes"
    report_size="$(stat -c '%s' "${report}")"
  fi
  local status="-"
  if [[ -f "${status_path}" ]]; then
    status="$(tail -n 1 "${status_path}" | tr '\t' ' ' | cut -c1-90)"
  fi
  local log_tail="-"
  if [[ -f "${log_path}" ]]; then
    log_tail="$(tail -n 1 "${log_path}" | cut -c1-90)"
  fi
  printf '| %s | %s | %s | %s | %s | %s | %s |\n' "${dataset}" "${policy}" "${session_state}" "${report_state}" "${report_size}" "${status}" "${log_tail}"
}

echo "| Dataset | Policy | Session | Report | Bytes | Status | Last log line |"
echo "| --- | --- | --- | --- | ---: | --- | --- |"
print_row "clean_mainline" "2wikimultihopqa" "${CLEAN_ROOT}"
print_row "clean_mainline" "musique" "${CLEAN_ROOT}"
print_row "clean_mainline" "hotpotqa" "${CLEAN_ROOT}"
print_row "no_multi_anchor_precision" "2wikimultihopqa" "${ABL_ROOT}"
print_row "no_multi_anchor_precision" "musique" "${ABL_ROOT}"
print_row "no_multi_anchor_precision" "hotpotqa" "${ABL_ROOT}"

echo
echo "Logs: ${GATE_ROOT}/logs"
echo "Status: ${GATE_ROOT}/status"
echo "Final report target: ${GATE_ROOT}/reports/no_multi_anchor_precision_full1000_gate.md"
