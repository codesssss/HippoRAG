#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/agsto_daec_llm_20260504"
SAVE_DIR="outputs_step0_general_nvembed"
EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
LLM_MODEL="qwen3-8b-train"
LIMIT="${LIMIT:-100}"
POOL_K="${POOL_K:-100}"
MATCH_MODE="${MATCH_MODE:-wiki_title}"

mkdir -p "${OUT_DIR}/pools" "${OUT_DIR}/evals" "${OUT_DIR}/logs" "${OUT_DIR}/status"

dataset_port() {
  case "$1" in
    musique) echo "8041" ;;
    hotpotqa) echo "8043" ;;
    2wikimultihopqa) echo "8041" ;;
    *) return 2 ;;
  esac
}

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_DIR}/logs/launcher.log"
}

export_pool() {
  local dataset="$1"
  local pool_json="${OUT_DIR}/pools/${dataset}_agsto_pool${POOL_K}_limit${LIMIT}.json"
  local status_path="${OUT_DIR}/status/${dataset}_export.status"
  local log_path="${OUT_DIR}/logs/${dataset}_export.log"

  if [[ -s "${pool_json}" ]]; then
    log_msg "SKIP export dataset=${dataset} existing=${pool_json}"
    return 0
  fi

  log_msg "START export dataset=${dataset} limit=${LIMIT} pool_k=${POOL_K}"
  echo "[START] dataset=${dataset} stage=export time=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" scripts/export_agsto_pool.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --pool_k "${POOL_K}" \
      --candidate_limit 180 \
      --proposal_candidate_depth 24 \
      --support_proposal_depth 12 \
      --beam_size 12 \
      --output_json "${pool_json}"
  ) > "${log_path}" 2>&1
  echo "[DONE] dataset=${dataset} stage=export time=$(date -Is)" > "${status_path}"
  log_msg "DONE export dataset=${dataset} output=${pool_json}"
}

eval_daec_llm() {
  local dataset="$1"
  local port
  port="$(dataset_port "${dataset}")"
  local pool_json="${OUT_DIR}/pools/${dataset}_agsto_pool${POOL_K}_limit${LIMIT}.json"
  local output_json="${OUT_DIR}/evals/${dataset}_agsto_pool${POOL_K}_daec_llm_limit${LIMIT}.json"
  local cache_path="${OUT_DIR}/evals/${dataset}_agsto_pool${POOL_K}_daec_llm.binding_cache.json"
  local status_path="${OUT_DIR}/status/${dataset}_eval.status"
  local log_path="${OUT_DIR}/logs/${dataset}_eval.log"

  log_msg "START eval dataset=${dataset} selector=daec_noisyor_llm limit=${LIMIT} port=${port}"
  echo "[START] dataset=${dataset} stage=eval selector=daec_noisyor_llm time=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name "${LLM_MODEL}" \
      --max_retry_attempts 20 \
      --qwen_disable_thinking \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name "${EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --external_pool_json "${pool_json}" \
      --external_pool_source_name "agsto_v12_pool${POOL_K}" \
      --external_pool_strict_questions true \
      --setwise_selector daec_noisyor_llm \
      --setwise_pool_k "${POOL_K}" \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --dtc_decomposition_mode llm \
      --dtc_binding_max_candidates 5 \
      --llm_binding_url "http://localhost:${port}/v1" \
      --llm_binding_model "${LLM_MODEL}" \
      --llm_binding_cache_path "${cache_path}" \
      --llm_binding_title_match_mode "${MATCH_MODE}" \
      --causal_enabled false \
      --causal_engine_version v2 \
      --causal_v2_base_retrieval_mode dense \
      --structure_rerank_enabled false \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1
  echo "[DONE] dataset=${dataset} stage=eval selector=daec_noisyor_llm time=$(date -Is)" > "${status_path}"
  log_msg "DONE eval dataset=${dataset} output=${output_json}"
}

log_msg "AG-STO + DAEC-LLM limit${LIMIT} run begin"
for dataset in musique hotpotqa 2wikimultihopqa; do
  export_pool "${dataset}"
  eval_daec_llm "${dataset}"
done
log_msg "AG-STO + DAEC-LLM limit${LIMIT} run complete"
