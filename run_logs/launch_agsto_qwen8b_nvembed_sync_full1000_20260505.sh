#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/agsto_qwen8b_nvembed_sync_full1000_20260505"
SAVE_DIR="outputs_step0_general_nvembed"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
EMBEDDING_API_MODEL="nvidia/NV-Embed-v2"
EVAL_EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"
LLM_MODEL="qwen3-8b-train"
LIMIT=1000
POOL_K=100
MATCH_MODE="wiki_title"

export HIPPORAG_RERANK_FORCE_NO_THINK=1
mkdir -p "${OUT_DIR}/pools" "${OUT_DIR}/evals" "${OUT_DIR}/logs" "${OUT_DIR}/status"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_DIR}/logs/launcher.log"
}

dataset_port() {
  case "$1" in
    musique) echo "8041" ;;
    hotpotqa) echo "8042" ;;
    2wikimultihopqa) echo "8043" ;;
    *) return 2 ;;
  esac
}

export_pool() {
  local dataset="$1"
  local pool_json="${OUT_DIR}/pools/${dataset}_qwen8b_nvembed_sync_pool${POOL_K}_full1000.json"
  local status_path="${OUT_DIR}/status/${dataset}_export.status"
  local log_path="${OUT_DIR}/logs/${dataset}_export.log"

  if [[ -s "${pool_json}" ]]; then
    log_msg "SKIP export dataset=${dataset} existing=${pool_json}"
    echo "[SKIP] dataset=${dataset} stage=export existing=${pool_json} time=$(date -Is)" > "${status_path}"
    return 0
  fi

  log_msg "START export dataset=${dataset} limit=${LIMIT}"
  echo "[START] dataset=${dataset} stage=export limit=${LIMIT} time=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/export_agsto_pool.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --pool_k "${POOL_K}" \
      --openie_template "outputs_step0_general_nvembed_{dataset}/openie_results_ner_qwen3-8b.json" \
      --retrieval_top_k 20 \
      --candidate_limit 120 \
      --proposal_candidate_depth 10 \
      --support_proposal_depth 6 \
      --beam_size 12 \
      --native_dense_anchor true \
      --native_dense_anchor_top_k 20 \
      --chunk_embedding_template "outputs_step0_general_nvembed_{dataset}/qwen3-8b_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --embedding_model "${EMBEDDING_API_MODEL}" \
      --embedding_batch_size 8 \
      --dense_query_instruction_mode raw \
      --semantic_residual_weight 12.0 \
      --output_json "${pool_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]] && [[ -s "${pool_json}" ]]; then
    echo "[DONE] dataset=${dataset} stage=export time=$(date -Is)" > "${status_path}"
    log_msg "DONE export dataset=${dataset} output=${pool_json}"
  else
    echo "[FAILED] dataset=${dataset} stage=export code=${code} time=$(date -Is)" > "${status_path}"
    log_msg "FAILED export dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

eval_daec() {
  local dataset="$1"
  local port="$2"
  local pool_json="${OUT_DIR}/pools/${dataset}_qwen8b_nvembed_sync_pool${POOL_K}_full1000.json"
  local output_json="${OUT_DIR}/evals/${dataset}_qwen8b_nvembed_sync_daec_llm_full1000.json"
  local cache_path="${OUT_DIR}/evals/${dataset}_qwen8b_nvembed_sync_daec_llm_full1000.binding_cache.json"
  local status_path="${OUT_DIR}/status/${dataset}_daec_eval.status"
  local log_path="${OUT_DIR}/logs/${dataset}_daec_eval.log"

  if [[ -s "${output_json}" ]]; then
    log_msg "SKIP daec eval dataset=${dataset} existing=${output_json}"
    echo "[SKIP] dataset=${dataset} stage=daec_eval existing=${output_json} time=$(date -Is)" > "${status_path}"
    return 0
  fi

  if [[ ! -s "${pool_json}" ]]; then
    log_msg "FAILED daec eval dataset=${dataset} missing_pool=${pool_json}"
    echo "[FAILED] dataset=${dataset} stage=daec_eval missing_pool=${pool_json} time=$(date -Is)" > "${status_path}"
    return 2
  fi

  log_msg "START daec eval dataset=${dataset} limit=${LIMIT} port=${port}"
  echo "[START] dataset=${dataset} stage=daec_eval limit=${LIMIT} port=${port} time=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name "${LLM_MODEL}" \
      --max_retry_attempts 20 \
      --qwen_disable_thinking \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name "${EVAL_EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --external_pool_json "${pool_json}" \
      --external_pool_source_name "agsto_qwen8b_nvembed_sync_pool${POOL_K}" \
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
  local code=$?
  if [[ "${code}" -eq 0 ]] && [[ -s "${output_json}" ]]; then
    echo "[DONE] dataset=${dataset} stage=daec_eval time=$(date -Is)" > "${status_path}"
    log_msg "DONE daec eval dataset=${dataset} output=${output_json}"
  else
    echo "[FAILED] dataset=${dataset} stage=daec_eval code=${code} time=$(date -Is)" > "${status_path}"
    log_msg "FAILED daec eval dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_dataset() {
  local dataset="$1"
  local port
  port="$(dataset_port "${dataset}")" || return 2
  export_pool "${dataset}" || return 1
  eval_daec "${dataset}" "${port}" || return 1
}

main() {
  echo "[START] agsto_qwen8b_nvembed_sync_full1000 time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
  log_msg "AG-STO Qwen3-8B + NV-Embed-v2 sync full1000 begin"
  log_msg "lanes: musique->8041 hotpotqa->8042 2wikimultihopqa->8043"

  (run_dataset musique) &
  local pid_musique=$!
  (run_dataset hotpotqa) &
  local pid_hotpotqa=$!
  (run_dataset 2wikimultihopqa) &
  local pid_2wiki=$!

  local status=0
  if ! wait "${pid_musique}"; then
    log_msg "FAILED dataset=musique"
    status=1
  fi
  if ! wait "${pid_hotpotqa}"; then
    log_msg "FAILED dataset=hotpotqa"
    status=1
  fi
  if ! wait "${pid_2wiki}"; then
    log_msg "FAILED dataset=2wikimultihopqa"
    status=1
  fi

  if [[ "${status}" -eq 0 ]]; then
    echo "[DONE] agsto_qwen8b_nvembed_sync_full1000 time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
    log_msg "AG-STO Qwen3-8B + NV-Embed-v2 sync full1000 complete"
  else
    echo "[FAILED] agsto_qwen8b_nvembed_sync_full1000 status=${status} time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
    log_msg "AG-STO Qwen3-8B + NV-Embed-v2 sync full1000 finished with failures"
  fi
  return "${status}"
}

main "$@"
