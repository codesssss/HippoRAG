#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
SOURCE_SETR_DIR="${ROOT_DIR}/run_logs/setr_full1000_20260503"
OUT_DIR="${ROOT_DIR}/run_logs/setr_fill10_proprag_full1000_20260507"
REPORT_DIR="${ROOT_DIR}/reports/setr_fill10_proprag_full1000_20260507"
SAVE_DIR="outputs_step0_general_nvembed"
LLM_MODEL="qwen3-8b-train"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"
LIMIT=1000
VARIANT="setr_k20_doc768"
FILL_TO_K=10

mkdir -p "${OUT_DIR}" "${REPORT_DIR}"

dataset_port() {
  case "$1" in
    2wikimultihopqa) echo 8041 ;;
    hotpotqa) echo 8042 ;;
    musique) echo 8043 ;;
    *) return 2 ;;
  esac
}

pool_path() {
  local dataset="$1"
  echo "${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424/${dataset}_pool100.json"
}

log_msg() {
  echo "[$(date -Is)] $*" | tee -a "${OUT_DIR}/launcher.log"
}

run_cmd() {
  local label="$1"
  shift
  local log_path="${OUT_DIR}/${label}.log"
  local status_path="${OUT_DIR}/${label}.status"

  if [[ -f "${status_path}" ]] && grep -q '^DONE ' "${status_path}"; then
    log_msg "SKIP ${label}"
    return 0
  fi

  echo "START ${label} $(date -Is)" > "${status_path}"
  log_msg "START ${label}"
  (
    cd "${ROOT_DIR}" || exit 2
    "$@"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    echo "DONE ${label} $(date -Is)" > "${status_path}"
    log_msg "DONE ${label}"
  else
    echo "FAILED ${label} code=${code} $(date -Is)" > "${status_path}"
    log_msg "FAILED ${label} code=${code}"
    return "${code}"
  fi
}

run_dataset() {
  local dataset="$1"
  local port source_pool selection_jsonl selected_pool_json output_json
  port="$(dataset_port "${dataset}")" || return 2
  source_pool="$(pool_path "${dataset}")"
  selection_jsonl="${SOURCE_SETR_DIR}/${dataset}_proprag_${VARIANT}.selection.jsonl"
  selected_pool_json="${OUT_DIR}/${dataset}_proprag_${VARIANT}_fill10.selected_pool.json"
  output_json="${REPORT_DIR}/${dataset}_proprag_${VARIANT}_fill10.eval.json"

  run_cmd "apply_${dataset}_proprag_${VARIANT}_fill10" \
    "${PYTHON_BIN}" scripts/apply_setr_selection_to_pool.py \
      --pool_json "${source_pool}" \
      --selection_jsonl "${selection_jsonl}" \
      --output_pool_json "${selected_pool_json}" \
      --limit "${LIMIT}" \
      --fallback_mode rank_order_fill_to_k \
      --fill_to_k "${FILL_TO_K}"

  run_cmd "eval_${dataset}_proprag_${VARIANT}_fill10" \
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name "${LLM_MODEL}" \
      --max_retry_attempts 20 \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name "${EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --qa_top_k "${FILL_TO_K}" \
      --qa_doc_max_chars 2048 \
      --external_pool_json "${selected_pool_json}" \
      --external_pool_source_name "setr_fill10_${VARIANT}_proprag_full1000" \
      --external_pool_strict_questions true \
      --setwise_selector none \
      --output_json "${output_json}"
}

main() {
  echo "[START] setr_fill10_proprag_full1000 variant=${VARIANT} fill_to_k=${FILL_TO_K} start=$(date -Is)" > "${OUT_DIR}/launcher.status"
  echo "[START] setr_fill10_proprag_full1000 variant=${VARIANT} fill_to_k=${FILL_TO_K} start=$(date -Is)" > "${OUT_DIR}/launcher.log"

  local pids=()
  local dataset
  for dataset in 2wikimultihopqa hotpotqa musique; do
    (
      run_dataset "${dataset}"
      code=$?
      if [[ "${code}" -eq 0 ]]; then
        echo "DONE dataset=${dataset} $(date -Is)" > "${OUT_DIR}/${dataset}.status"
      else
        echo "FAILED dataset=${dataset} code=${code} $(date -Is)" > "${OUT_DIR}/${dataset}.status"
      fi
      exit "${code}"
    ) &
    pids+=("$!")
  done

  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done

  if [[ "${rc}" -eq 0 ]]; then
    echo "[DONE] setr_fill10_proprag_full1000 variant=${VARIANT} fill_to_k=${FILL_TO_K} end=$(date -Is)" > "${OUT_DIR}/launcher.status"
  else
    echo "[FAILED] setr_fill10_proprag_full1000 variant=${VARIANT} fill_to_k=${FILL_TO_K} rc=${rc} end=$(date -Is)" > "${OUT_DIR}/launcher.status"
  fi
  return "${rc}"
}

main "$@"
