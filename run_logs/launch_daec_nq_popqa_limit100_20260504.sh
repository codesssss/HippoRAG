#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/daec_nq_popqa_limit100_20260504"
SAVE_DIR="outputs_step0_general_nvembed"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
MATCH_MODE="${MATCH_MODE:-wiki_title}"
LIMIT="${LIMIT:-100}"

export HIPPORAG_RERANK_FORCE_NO_THINK=1
mkdir -p "${OUT_DIR}"

dataset_port() {
  case "$1" in
    nq) echo "${NQ_PORT:-8041}" ;;
    popqa) echo "${POPQA_PORT:-8042}" ;;
    *) return 2 ;;
  esac
}

run_one() {
  local dataset="$1"
  local variant="$2"
  local port stem output_json log_path status_path cache_path selector

  port="$(dataset_port "${dataset}")" || return 2
  stem="${dataset}_${variant}_limit${LIMIT}"
  output_json="${OUT_DIR}/${stem}.json"
  log_path="${OUT_DIR}/${stem}.log"
  status_path="${OUT_DIR}/${stem}.status"
  cache_path="${OUT_DIR}/${stem}.binding_cache.json"

  case "${variant}" in
    top5)
      selector="none"
      ;;
    daec_hc)
      selector="daec_noisyor"
      ;;
    daec_llm_ctl)
      selector="daec_noisyor_llm"
      ;;
    *)
      echo "[FAILED] unknown variant=${variant}" > "${status_path}"
      return 2
      ;;
  esac

  echo "[START] dataset=${dataset} variant=${variant} selector=${selector} limit=${LIMIT} start=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    cmd=(
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
      --retrieval_top_k 100
      --causal_enabled false
      --causal_engine_version v2
      --causal_v2_base_retrieval_mode dense
      --structure_rerank_enabled false
      --setwise_selector "${selector}"
      --setwise_pool_k 100
      --qa_top_k 5
      --qa_doc_max_chars 2048
      --dtc_decomposition_mode llm
      --dtc_binding_max_candidates 5
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
    "${cmd[@]}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]] && [[ -s "${output_json}" ]]; then
    echo "[DONE] dataset=${dataset} variant=${variant} limit=${LIMIT} end=$(date -Is)" > "${status_path}"
  else
    echo "[FAILED] dataset=${dataset} variant=${variant} limit=${LIMIT} code=${code} end=$(date -Is)" > "${status_path}"
  fi
  return "${code}"
}

run_dataset_worker() {
  local dataset="$1"
  local rc=0
  local variant
  for variant in top5 daec_hc daec_llm_ctl; do
    echo "[JOB_START] dataset=${dataset} variant=${variant} limit=${LIMIT} start=$(date -Is)" | tee -a "${OUT_DIR}/launcher.log"
    run_one "${dataset}" "${variant}" || rc=1
    echo "[JOB_DONE] dataset=${dataset} variant=${variant} rc=${rc} end=$(date -Is)" | tee -a "${OUT_DIR}/launcher.log"
  done
  return "${rc}"
}

main() {
  echo "[START] daec_nq_popqa_limit${LIMIT} start=$(date -Is)" > "${OUT_DIR}/launcher.status"
  echo "[START] daec_nq_popqa_limit${LIMIT} start=$(date -Is)" > "${OUT_DIR}/launcher.log"
  local pids=()
  local dataset
  for dataset in nq popqa; do
    run_dataset_worker "${dataset}" &
    pids+=("$!")
  done
  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  if [[ "${rc}" -eq 0 ]]; then
    echo "[DONE] daec_nq_popqa_limit${LIMIT} end=$(date -Is)" > "${OUT_DIR}/launcher.status"
  else
    echo "[FAILED] daec_nq_popqa_limit${LIMIT} rc=${rc} end=$(date -Is)" > "${OUT_DIR}/launcher.status"
  fi
  return "${rc}"
}

main "$@"
