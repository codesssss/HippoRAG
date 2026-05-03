#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/hipporag_pool_exports_full1000_20260503"
SAVE_DIR="outputs_step0_general_nvembed"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
export HIPPORAG_RERANK_FORCE_NO_THINK=1
mkdir -p "${OUT_DIR}"

dataset_port() {
  case "$1" in
    2wikimultihopqa) echo 8041 ;;
    hotpotqa) echo 8042 ;;
    musique) echo 8043 ;;
    *) return 2 ;;
  esac
}

run_one() {
  local dataset="$1"
  local port
  port="$(dataset_port "${dataset}")" || return 2
  local output_json="${OUT_DIR}/${dataset}_hipporag_pool100.json"
  local log_path="${OUT_DIR}/${dataset}.log"
  local status_path="${OUT_DIR}/${dataset}.status"

  echo "[START] dataset=${dataset} pool_k=100 port=${port} start=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/export_hipporag_pool.py \
      --dataset "${dataset}" \
      --limit 1000 \
      --pool_k 100 \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name qwen3-8b-train \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name VLLM/nvidia/NV-Embed-v2 \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --retrieval_top_k 100 \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]] && [[ -s "${output_json}" ]]; then
    echo "[DONE] dataset=${dataset} pool_k=100 end=$(date -Is)" > "${status_path}"
  else
    echo "[FAILED] dataset=${dataset} pool_k=100 code=${code} end=$(date -Is)" > "${status_path}"
  fi
  return "${code}"
}

main() {
  echo "[START] hipporag_pool_exports start=$(date -Is)" > "${OUT_DIR}/launcher.status"
  # Run sequentially to avoid save_dir race conditions
  local rc=0
  for dataset in 2wikimultihopqa hotpotqa musique; do
    run_one "${dataset}" || rc=1
  done
  if [[ "${rc}" -eq 0 ]]; then
    echo "[DONE] hipporag_pool_exports end=$(date -Is)" > "${OUT_DIR}/launcher.status"
  else
    echo "[FAILED] hipporag_pool_exports rc=${rc} end=$(date -Is)" > "${OUT_DIR}/launcher.status"
  fi
  return "${rc}"
}

main "$@"
