#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
OUT_DIR="${ROOT_DIR}/run_logs/nq_popqa_hippo_bridge_expand_parity_limit100_20260506"
SESSION="nq_popqa_bridge100"
NQ_DAEC="${OUT_DIR}/evals/nq_bridge_daec_qwen8b_legacy_structure_limit100.json"
NQ_CE="${OUT_DIR}/evals/nq_bridge_ce_qwen8b_legacy_structure_limit100.json"
STATUS_PATH="${OUT_DIR}/status/stop_after_nq_watchdog.status"
LOG_PATH="${OUT_DIR}/logs/stop_after_nq_watchdog.log"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/status"

log_msg() {
  printf '[%s] [stop_after_nq] %s\n' "$(date -Is)" "$*" | tee -a "${LOG_PATH}"
}

echo "[START] waiting_for_nq_outputs session=${SESSION} time=$(date -Is)" > "${STATUS_PATH}"
log_msg "Waiting for NQ DAEC+CE outputs before stopping original sequential launcher."

while true; do
  if [[ -s "${NQ_DAEC}" && -s "${NQ_CE}" ]]; then
    echo "[READY] nq_outputs_present time=$(date -Is)" > "${STATUS_PATH}"
    log_msg "NQ outputs are present. Stopping original session ${SESSION} so PopQA is not duplicated."
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      tmux kill-session -t "${SESSION}"
      echo "[DONE] stopped_session=${SESSION} time=$(date -Is)" > "${STATUS_PATH}"
      log_msg "Stopped session ${SESSION}."
    else
      echo "[DONE] session_absent=${SESSION} time=$(date -Is)" > "${STATUS_PATH}"
      log_msg "Session ${SESSION} already absent."
    fi
    exit 0
  fi

  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[DONE] session_absent_before_nq_complete=${SESSION} time=$(date -Is)" > "${STATUS_PATH}"
    log_msg "Session ${SESSION} is absent before both NQ outputs were present; leaving PopQA parallel run untouched."
    exit 0
  fi

  sleep 30
done
