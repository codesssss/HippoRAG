#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/nq_popqa_hippo_bridge_expand_parity_limit100_20260506"
SAVE_DIR="outputs_step0_general_nvembed"
POOL_DIR="${ROOT_DIR}/run_logs/nq_popqa_full_pools_daec_20260504/pools"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EVAL_EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"
LLM_MODEL="qwen3-8b-train"
LIMIT=100
POOL_K=100
MATCH_MODE="${MATCH_MODE:-wiki_title}"
CE_MODEL="/mnt/nvme/bge-reranker-v2-m3"
CE_DEVICE="${CE_DEVICE:-cuda:4}"
PORT="${POPQA_PORT:-8042}"
DATASET="popqa"

export HIPPORAG_RERANK_FORCE_NO_THINK=1
mkdir -p "${OUT_DIR}/evals" "${OUT_DIR}/logs" "${OUT_DIR}/status"

log_msg() {
  printf '[%s] [popqa_parallel] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_DIR}/logs/launcher_popqa_parallel.log"
}

variant_selector_args() {
  local variant="$1"
  case "${variant}" in
    bridge_daec)
      printf '%s\n' \
        --setwise_selector bridge_append \
        --setwise_score_mode bridge \
        --setwise_pool_k "${POOL_K}" \
        --setwise_non_anchor_title_dedup true \
        --expand_base_k 10 \
        --append_max_docs 3 \
        --append_policy bridge \
        --append_random_seed 0 \
        --expand_min_structure_score 0.35 \
        --assemble_mode daec_noisyor_llm \
        --structure_relation_probe_mode general_factual \
        --dtc_decomposition_mode llm \
        --dtc_binding_max_candidates 5
      ;;
    bridge_ce)
      printf '%s\n' \
        --setwise_selector bridge_append \
        --setwise_score_mode bridge \
        --setwise_pool_k "${POOL_K}" \
        --setwise_non_anchor_title_dedup true \
        --expand_base_k 10 \
        --append_max_docs 3 \
        --expand_min_structure_score 0.35 \
        --append_policy bridge \
        --append_random_seed 0 \
        --assemble_mode cross_encoder \
        --structure_relation_probe_mode general_factual \
        --ce_model "${CE_MODEL}" \
        --ce_device "${CE_DEVICE}"
      ;;
    *)
      return 2
      ;;
  esac
}

eval_variant() {
  local variant="$1"
  local pool_json output_json log_path status_path cache_path
  local selector_args=()

  pool_json="${POOL_DIR}/${DATASET}_hipporag_pool100.json"
  output_json="${OUT_DIR}/evals/${DATASET}_${variant}_qwen8b_legacy_structure_limit100.json"
  log_path="${OUT_DIR}/logs/${DATASET}_${variant}.log"
  status_path="${OUT_DIR}/status/${DATASET}_${variant}.status"
  cache_path="${OUT_DIR}/evals/${DATASET}_shared.binding_cache.json"
  mapfile -t selector_args < <(variant_selector_args "${variant}") || return 2

  if [[ ! -s "${pool_json}" ]]; then
    echo "[FAILED] dataset=${DATASET} variant=${variant} missing_pool=${pool_json} time=$(date -Is)" > "${status_path}"
    log_msg "FAILED eval dataset=${DATASET} variant=${variant} missing_pool=${pool_json}"
    return 1
  fi
  if [[ -s "${output_json}" ]]; then
    echo "[SKIP] dataset=${DATASET} variant=${variant} output_exists=${output_json} time=$(date -Is)" > "${status_path}"
    log_msg "SKIP eval dataset=${DATASET} variant=${variant} output=${output_json}"
    return 0
  fi

  log_msg "START eval dataset=${DATASET} variant=${variant} limit=${LIMIT} qa_top_k=5 port=${PORT}"
  echo "[START] dataset=${DATASET} variant=${variant} limit=${LIMIT} qa_top_k=5 port=${PORT} parallel=true time=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset "${DATASET}" \
      --limit "${LIMIT}" \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name "${LLM_MODEL}" \
      --max_retry_attempts 20 \
      --qwen_disable_thinking \
      --llm_base_url "http://localhost:${PORT}/v1" \
      --embedding_name "${EVAL_EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --external_pool_json "${pool_json}" \
      --external_pool_source_name hipporag_pool100 \
      --external_pool_strict_questions true \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --llm_binding_url "http://localhost:${PORT}/v1" \
      --llm_binding_model "${LLM_MODEL}" \
      --llm_binding_cache_path "${cache_path}" \
      --llm_binding_title_match_mode "${MATCH_MODE}" \
      --causal_enabled true \
      --causal_engine_version legacy \
      --structure_rerank_enabled true \
      "${selector_args[@]}" \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]] && [[ -s "${output_json}" ]]; then
    echo "[DONE] dataset=${DATASET} variant=${variant} parallel=true time=$(date -Is)" > "${status_path}"
    log_msg "DONE eval dataset=${DATASET} variant=${variant} output=${output_json}"
  else
    echo "[FAILED] dataset=${DATASET} variant=${variant} code=${code} parallel=true time=$(date -Is)" > "${status_path}"
    log_msg "FAILED eval dataset=${DATASET} variant=${variant} code=${code}"
  fi
  return "${code}"
}

main() {
  echo "[START] popqa_parallel_bridge_expand_parity time=$(date -Is)" > "${OUT_DIR}/status/popqa_parallel_launcher.status"
  log_msg "PopQA bridge expand parity parallel begin; shared output dir=${OUT_DIR}"

  local status=0
  eval_variant bridge_daec || status=1
  eval_variant bridge_ce || status=1

  if [[ "${status}" -eq 0 ]]; then
    echo "[DONE] popqa_parallel_bridge_expand_parity time=$(date -Is)" > "${OUT_DIR}/status/popqa_parallel_launcher.status"
    log_msg "PopQA bridge expand parity parallel complete"
  else
    echo "[FAILED] popqa_parallel_bridge_expand_parity status=${status} time=$(date -Is)" > "${OUT_DIR}/status/popqa_parallel_launcher.status"
    log_msg "PopQA bridge expand parity parallel finished with failures"
  fi
  return "${status}"
}

main "$@"
