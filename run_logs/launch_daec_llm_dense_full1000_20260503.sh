#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/daec_llm_dense_full1000_20260503"
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

pool_json() {
  echo "${ROOT_DIR}/run_logs/dense_pool_exports_full1000_20260424/${1}_dense_pool100.json"
}

run_one() {
  local dataset="$1"
  local port pool output_json log_path status_path
  port="$(dataset_port "${dataset}")" || return 2
  pool="$(pool_json "${dataset}")"
  output_json="${OUT_DIR}/${dataset}_daec_llm_dense_full1000.json"
  log_path="${OUT_DIR}/${dataset}_daec_llm_dense_full1000.log"
  status_path="${OUT_DIR}/${dataset}_daec_llm_dense_full1000.status"

  echo "[START] dataset=${dataset} selector=daec_noisyor_llm pool=dense limit=1000 port=${port} start=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit 1000 \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name qwen3-8b-train \
      --max_retry_attempts 20 \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name VLLM/nvidia/NV-Embed-v2 \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --external_pool_json "${pool}" \
      --external_pool_source_name dense_pool100 \
      --external_pool_strict_questions true \
      --setwise_selector daec_noisyor_llm \
      --setwise_pool_k 100 \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --dtc_decomposition_mode llm \
      --dtc_binding_max_candidates 5 \
      --llm_binding_url "http://localhost:${port}/v1" \
      --llm_binding_model "qwen3-8b-train" \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]] && [[ -s "${output_json}" ]]; then
    echo "[DONE] dataset=${dataset} selector=daec_noisyor_llm pool=dense limit=1000 end=$(date -Is)" > "${status_path}"
  else
    echo "[FAILED] dataset=${dataset} selector=daec_noisyor_llm pool=dense limit=1000 code=${code} end=$(date -Is)" > "${status_path}"
  fi
  return "${code}"
}

main() {
  echo "[START] daec_llm_dense_full1000 start=$(date -Is)" > "${OUT_DIR}/launcher.status"
  local pids=()
  for dataset in 2wikimultihopqa hotpotqa musique; do
    run_one "${dataset}" &
    pids+=("$!")
  done
  local rc=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  if [[ "${rc}" -eq 0 ]]; then
    echo "[DONE] daec_llm_dense_full1000 end=$(date -Is)" > "${OUT_DIR}/launcher.status"
  else
    echo "[FAILED] daec_llm_dense_full1000 rc=${rc} end=$(date -Is)" > "${OUT_DIR}/launcher.status"
  fi
  return "${rc}"
}

main "$@"
