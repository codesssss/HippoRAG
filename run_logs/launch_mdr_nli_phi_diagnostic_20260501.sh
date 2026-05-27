#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/mdr_nli_phi_diagnostic_20260501"

mkdir -p "${OUT_DIR}"

run_one() {
  local dataset="$1"
  local limit="${2:-20}"
  local log_path="${OUT_DIR}/${dataset}_nli_phi_diagnostic_limit${limit}.log"
  local status_path="${OUT_DIR}/${dataset}_nli_phi_diagnostic_limit${limit}.status"

  echo "[START] dataset=${dataset} limit=${limit} start=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/mdr_nli_phi_diagnostic.py \
      --dataset "${dataset}" \
      --limit "${limit}" \
      --qa_top_k 5 \
      --tau_percentile 50.0 \
      --nli_model "cross-encoder/nli-deberta-v3-base" \
      --nli_batch_size 32
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    echo "[DONE] dataset=${dataset} limit=${limit} end=$(date -Is)" > "${status_path}"
  else
    echo "[FAILED] dataset=${dataset} limit=${limit} code=${code} end=$(date -Is)" > "${status_path}"
  fi
  return "${code}"
}

main() {
  local limit="${1:-20}"
  echo "[START] mdr_nli_phi_diagnostic limit=${limit} start=$(date -Is)" > "${OUT_DIR}/launcher.status"

  # Run sequentially — NLI model is on GPU, parallel would OOM
  local rc=0
  for dataset in 2wikimultihopqa hotpotqa musique; do
    run_one "${dataset}" "${limit}" || rc=1
  done

  if [[ "${rc}" -eq 0 ]]; then
    echo "[DONE] mdr_nli_phi_diagnostic limit=${limit} end=$(date -Is)" > "${OUT_DIR}/launcher.status"
  else
    echo "[FAILED] mdr_nli_phi_diagnostic limit=${limit} rc=${rc} end=$(date -Is)" > "${OUT_DIR}/launcher.status"
  fi
  return "${rc}"
}

main "$@"
