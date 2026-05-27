#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
METHOD="evidence_transition_graphragv4_fact_witnessed_sto"
GATE_ROOT="${ROOT_DIR}/run_logs/etv4_clean_vs_no_multi_anchor_full1000_20260511"
CLEAN_ROOT="${ROOT_DIR}/run_logs/etv4_clean_mainline_full1000_reuse_gpt4omini_20260511"
ABL_ROOT="${ROOT_DIR}/run_logs/etv4_no_multi_anchor_precision_full1000_reuse_gpt4omini_20260511"
WRITER="${ROOT_DIR}/run_logs/write_etv4_full1000_no_multi_anchor_gate_report_20260511.py"
LOG_PATH="${GATE_ROOT}/logs/gate_reporter.log"
STATUS_PATH="${GATE_ROOT}/status/gate_reporter.status"

mkdir -p "${GATE_ROOT}/logs" "${GATE_ROOT}/status" "${GATE_ROOT}/reports"

report_path() {
  local out_root="$1"
  local dataset="$2"
  printf '%s/%s/reports/%s_%s_retrieval.json' "${out_root}" "${dataset}" "${dataset}" "${METHOD}"
}

expected_reports=(
  "$(report_path "${CLEAN_ROOT}" "2wikimultihopqa")"
  "$(report_path "${CLEAN_ROOT}" "musique")"
  "$(report_path "${CLEAN_ROOT}" "hotpotqa")"
  "$(report_path "${ABL_ROOT}" "2wikimultihopqa")"
  "$(report_path "${ABL_ROOT}" "musique")"
  "$(report_path "${ABL_ROOT}" "hotpotqa")"
)

{
  echo "[START] gate_reporter start=$(date -Is)"
  while true; do
    missing=()
    for path in "${expected_reports[@]}"; do
      if [[ ! -f "${path}" ]]; then
        missing+=("${path}")
      fi
    done
    if [[ "${#missing[@]}" -eq 0 ]]; then
      echo "[READY] all retrieval reports exist at $(date -Is)"
      printf 'ready\t%s\n' "$(date -Is)" > "${STATUS_PATH}"
      "${WRITER}"
      printf 'done\t%s\t%s\n' "$(date -Is)" "${GATE_ROOT}/reports/no_multi_anchor_precision_full1000_gate.md" > "${STATUS_PATH}"
      echo "[DONE] gate report written at $(date -Is)"
      exit 0
    fi
    echo "[WAIT] missing=${#missing[@]} time=$(date -Is)"
    printf 'waiting\t%s\tmissing=%s\n' "$(date -Is)" "${#missing[@]}" > "${STATUS_PATH}"
    printf '  missing: %s\n' "${missing[@]}"
    sleep 60
  done
} 2>&1 | tee "${LOG_PATH}"
