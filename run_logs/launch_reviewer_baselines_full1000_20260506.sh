#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
OUT_DIR="${ROOT_DIR}/run_logs/reviewer_baselines_full1000_20260506"
mkdir -p "${OUT_DIR}"

run_phase() {
  local name="$1"
  local script="$2"
  local log_path="${OUT_DIR}/${name}.log"
  local status_path="${OUT_DIR}/${name}.status"

  echo "[START] phase=${name} script=${script} start=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}"
    /bin/bash "${script}"
  ) > "${log_path}" 2>&1
  echo "[DONE] phase=${name} end=$(date -Is)" > "${status_path}"
}

echo "[START] reviewer baselines full1000 queue $(date -Is)" > "${OUT_DIR}/launcher.status"

run_phase ircot_style_full1000 run_logs/launch_ircot_style_full1000_20260506.sh
run_phase llm_direct_select_full1000 run_logs/launch_llm_direct_select_proprag_full1000_20260506.sh

echo "[DONE] reviewer baselines full1000 queue $(date -Is)" > "${OUT_DIR}/launcher.status"
