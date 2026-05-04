#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/nq_popqa_full_pools_daec_20260504"
SAVE_DIR="outputs_step0_general_nvembed"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
MATCH_MODE="${MATCH_MODE:-wiki_title}"
LIMIT="${LIMIT:-1000}"

export HIPPORAG_RERANK_FORCE_NO_THINK=1
mkdir -p "${OUT_DIR}/pools" "${OUT_DIR}/evals"

dataset_port() {
  case "$1" in
    nq) echo "${NQ_PORT:-8041}" ;;
    popqa) echo "${POPQA_PORT:-8042}" ;;
    *) return 2 ;;
  esac
}

pool_path() {
  local dataset="$1"
  local pool="$2"
  echo "${OUT_DIR}/pools/${dataset}_${pool}_pool100.json"
}

run_logged() {
  local label="$1"
  local status_path="$2"
  local log_path="$3"
  shift 3

  echo "[START] ${label} start=$(date -Is)" > "${status_path}"
  echo "[START] ${label} start=$(date -Is)" > "${log_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "$@"
  ) >> "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    echo "[DONE] ${label} end=$(date -Is)" > "${status_path}"
    echo "[DONE] ${label} end=$(date -Is)" >> "${log_path}"
  else
    echo "[FAILED] ${label} code=${code} end=$(date -Is)" > "${status_path}"
    echo "[FAILED] ${label} code=${code} end=$(date -Is)" >> "${log_path}"
  fi
  return "${code}"
}

export_pool() {
  local dataset="$1"
  local pool="$2"
  local port output_json status_path log_path

  port="$(dataset_port "${dataset}")" || return 2
  output_json="$(pool_path "${dataset}" "${pool}")"
  status_path="${OUT_DIR}/pools/${dataset}_${pool}.status"
  log_path="${OUT_DIR}/pools/${dataset}_${pool}.log"
  if [[ -s "${output_json}" ]]; then
    echo "[SKIP] dataset=${dataset} pool=${pool} existing=${output_json} time=$(date -Is)" > "${status_path}"
    return 0
  fi

  case "${pool}" in
    dense)
      run_logged "export dataset=${dataset} pool=dense limit=${LIMIT}" "${status_path}" "${log_path}" \
        "${PYTHON_BIN}" scripts/export_dense_pool.py \
          --dataset "${dataset}" \
          --limit "${LIMIT}" \
          --pool_k 100 \
          --save_dir "${SAVE_DIR}" \
          --llm_name qwen3-8b \
          --embedding_name VLLM/nvidia/NV-Embed-v2 \
          --embedding_base_url "${EMBEDDING_BASE_URL}" \
          --embedding_batch_size 32 \
          --output_json "${output_json}"
      ;;
    hipporag)
      run_logged "export dataset=${dataset} pool=hipporag limit=${LIMIT}" "${status_path}" "${log_path}" \
        "${PYTHON_BIN}" scripts/export_hipporag_pool.py \
          --dataset "${dataset}" \
          --limit "${LIMIT}" \
          --pool_k 100 \
          --save_dir "${SAVE_DIR}" \
          --llm_name qwen3-8b \
          --llm_request_name qwen3-8b-train \
          --llm_base_url "http://localhost:${port}/v1" \
          --embedding_name VLLM/nvidia/NV-Embed-v2 \
          --embedding_base_url "${EMBEDDING_BASE_URL}" \
          --retrieval_top_k 100 \
          --output_json "${output_json}"
      ;;
    proprag)
      run_logged "export dataset=${dataset} pool=proprag limit=${LIMIT}" "${status_path}" "${log_path}" \
        "${PYTHON_BIN}" scripts/export_proprag_pool.py \
          --dataset "${dataset}" \
          --limit "${LIMIT}" \
          --pool_k 100 \
          --llm_name qwen3-8b-train \
          --llm_base_url "http://localhost:${port}/v1" \
          --embedding_name VLLM/nvidia/NV-Embed-v2 \
          --embedding_base_url "${EMBEDDING_BASE_URL}" \
          --save_dir /mnt/nvme/code/PropRAG/outputs_aligned_clean_nothink_top100_nvembed_rebuild_20260423 \
          --retrieval_top_k 100 \
          --qa_top_k 5 \
          --embedding_batch_size 32 \
          --max_new_tokens 2048 \
          --reuse_preextracted_openie true \
          --openie_llm_name qwen3-8b-train \
          --force_index_from_scratch false \
          --force_openie_from_scratch false \
          --openie_mode online \
          --use_propositions true \
          --use_beam_search true \
          --beam_width 4 \
          --max_path_length 3 \
          --second_stage_filter_k 40 \
          --sim_threshold 0.75 \
          --output_json "${output_json}"
      ;;
    *)
      echo "[FAILED] unknown pool=${pool}" > "${status_path}"
      return 2
      ;;
  esac
}

eval_pool() {
  local dataset="$1"
  local pool="$2"
  local variant="$3"
  local selector port input_pool output_json status_path log_path cache_path

  port="$(dataset_port "${dataset}")" || return 2
  input_pool="$(pool_path "${dataset}" "${pool}")"
  if [[ ! -s "${input_pool}" ]]; then
    echo "[SKIP] dataset=${dataset} pool=${pool} variant=${variant} missing_pool=${input_pool} time=$(date -Is)" \
      > "${OUT_DIR}/evals/${dataset}_${pool}_${variant}.status"
    return 0
  fi

  case "${variant}" in
    daec_hc) selector="daec_noisyor" ;;
    daec_llm_ctl) selector="daec_noisyor_llm" ;;
    *) return 2 ;;
  esac

  output_json="${OUT_DIR}/evals/${dataset}_${pool}_${variant}_full${LIMIT}.json"
  status_path="${OUT_DIR}/evals/${dataset}_${pool}_${variant}.status"
  log_path="${OUT_DIR}/evals/${dataset}_${pool}_${variant}.log"
  cache_path="${OUT_DIR}/evals/${dataset}_${pool}_${variant}.binding_cache.json"
  if [[ -s "${output_json}" ]]; then
    echo "[SKIP] dataset=${dataset} pool=${pool} variant=${variant} existing=${output_json} time=$(date -Is)" > "${status_path}"
    return 0
  fi

  local cmd=(
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py
    --dataset "${dataset}"
    --limit "${LIMIT}"
    --save_dir "${SAVE_DIR}"
    --llm_name qwen3-8b
    --llm_request_name qwen3-8b-train
    --max_retry_attempts 20
    --llm_base_url "http://localhost:${port}/v1"
    --embedding_name VLLM/nvidia/NV-Embed-v2
    --embedding_base_url "${EMBEDDING_BASE_URL}"
    --external_pool_json "${input_pool}"
    --external_pool_source_name "${pool}_pool100"
    --external_pool_strict_questions true
    --setwise_selector "${selector}"
    --setwise_pool_k 100
    --qa_top_k 5
    --qa_doc_max_chars 2048
    --dtc_decomposition_mode llm
    --dtc_binding_max_candidates 5
    --causal_enabled false
    --causal_engine_version v2
    --causal_v2_base_retrieval_mode dense
    --structure_rerank_enabled false
    --output_json "${output_json}"
  )
  if [[ "${variant}" == "daec_llm_ctl" ]]; then
    cmd+=(
      --llm_binding_url "http://localhost:${port}/v1"
      --llm_binding_model qwen3-8b-train
      --llm_binding_cache_path "${cache_path}"
      --llm_binding_title_match_mode "${MATCH_MODE}"
    )
  fi

  run_logged "eval dataset=${dataset} pool=${pool} variant=${variant} limit=${LIMIT}" \
    "${status_path}" "${log_path}" "${cmd[@]}"
}

main() {
  echo "[START] nq_popqa_full_pools_daec limit=${LIMIT} match=${MATCH_MODE} start=$(date -Is)" > "${OUT_DIR}/launcher.status"
  echo "[START] nq_popqa_full_pools_daec limit=${LIMIT} match=${MATCH_MODE} start=$(date -Is)" > "${OUT_DIR}/launcher.log"

  local dataset pool variant
  for dataset in nq popqa; do
    for pool in dense hipporag proprag; do
      echo "[JOB_START] export dataset=${dataset} pool=${pool} start=$(date -Is)" | tee -a "${OUT_DIR}/launcher.log"
      export_pool "${dataset}" "${pool}" || true
      echo "[JOB_DONE] export dataset=${dataset} pool=${pool} end=$(date -Is)" | tee -a "${OUT_DIR}/launcher.log"
      for variant in daec_hc daec_llm_ctl; do
        echo "[JOB_START] eval dataset=${dataset} pool=${pool} variant=${variant} start=$(date -Is)" | tee -a "${OUT_DIR}/launcher.log"
        eval_pool "${dataset}" "${pool}" "${variant}" || true
        echo "[JOB_DONE] eval dataset=${dataset} pool=${pool} variant=${variant} end=$(date -Is)" | tee -a "${OUT_DIR}/launcher.log"
      done
    done
  done

  echo "[DONE] nq_popqa_full_pools_daec limit=${LIMIT} end=$(date -Is)" > "${OUT_DIR}/launcher.status"
}

main "$@"
