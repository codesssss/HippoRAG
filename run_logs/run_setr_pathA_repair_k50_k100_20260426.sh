#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
OUT_DIR="${ROOT_DIR}/run_logs/setr_pathA_pilot_20260426"
REPAIR_DIR="${ROOT_DIR}/run_logs/setr_pathA_repair_k50_k100_20260426"
REPORT_DIR="${ROOT_DIR}/reports/setr_pathA"
SAVE_DIR="outputs_step0_general_nvembed"
LLM_BASE_URL="http://localhost:8041/v1"
LLM_MODEL="qwen3-8b-train"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"
DATASET="2wikimultihopqa"
LIMIT=100

mkdir -p "${REPAIR_DIR}" "${REPORT_DIR}"

log_msg() {
  echo "[$(date -Is)] $*" | tee -a "${REPAIR_DIR}/launcher.status"
}

run_cmd() {
  local label="$1"
  shift
  local log_path="${REPAIR_DIR}/${label}.log"
  local status_path="${REPAIR_DIR}/${label}.status"
  log_msg "START ${label}"
  echo "START ${label} $(date -Is)" > "${status_path}"
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
    log_msg "FAILED ${label} code=${code}; continuing"
  fi
  return 0
}

pool_path() {
  local pool="$1"
  case "${pool}" in
    dense) echo "${ROOT_DIR}/run_logs/dense_pool_exports_full1000_20260424/${DATASET}_dense_pool100.json" ;;
    proprag) echo "${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424/${DATASET}_pool100.json" ;;
    *) echo ""; return 2 ;;
  esac
}

doc_chars_for_k() {
  local k="$1"
  case "${k}" in
    50) echo 320 ;;
    100) echo 160 ;;
    *) echo 768 ;;
  esac
}

run_repair_pipeline() {
  local pool="$1"
  local k="$2"
  local doc_chars
  doc_chars="$(doc_chars_for_k "${k}")"
  local pool_json
  pool_json="$(pool_path "${pool}")"
  local prefix="${REPAIR_DIR}/setr_${DATASET}_${pool}_k${k}_limit${LIMIT}_doc${doc_chars}"
  local report_prefix="${REPORT_DIR}/setr_${DATASET}_${pool}_k${k}_limit${LIMIT}_doc${doc_chars}"
  local input_jsonl="${prefix}.inputs.jsonl"
  local selection_jsonl="${prefix}.selection.jsonl"
  local selected_pool_json="${prefix}.selected_pool.json"

  run_cmd "repair_export_${pool}_k${k}" \
    .venv-hipporag/bin/python scripts/export_setr_inputs.py \
      --pool_json "${pool_json}" \
      --output_jsonl "${input_jsonl}" \
      --pool_k "${k}" \
      --doc_max_chars "${doc_chars}" \
      --limit "${LIMIT}"

  run_cmd "repair_select_${pool}_k${k}" \
    .venv-hipporag/bin/python scripts/run_setr_style_selector.py \
      --input_jsonl "${input_jsonl}" \
      --output_jsonl "${selection_jsonl}" \
      --llm_base_url "${LLM_BASE_URL}" \
      --model "${LLM_MODEL}" \
      --max_tokens 512 \
      --temperature 0 \
      --num_workers 1 \
      --resume

  run_cmd "repair_apply_${pool}_k${k}" \
    .venv-hipporag/bin/python scripts/apply_setr_selection_to_pool.py \
      --pool_json "${pool_json}" \
      --selection_jsonl "${selection_jsonl}" \
      --output_pool_json "${selected_pool_json}" \
      --limit "${LIMIT}"

  run_cmd "repair_eval_${pool}_k${k}" \
    .venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
      --dataset "${DATASET}" --limit "${LIMIT}" --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b --llm_request_name "${LLM_MODEL}" \
      --max_retry_attempts 20 \
      --llm_base_url "${LLM_BASE_URL}" \
      --embedding_name "${EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --external_pool_json "${selected_pool_json}" \
      --external_pool_source_name "setr_style_${pool}_k${k}_doc${doc_chars}" \
      --setwise_selector none \
      --output_json "${report_prefix}.eval.json"
}

log_msg "SetR Path-A repair queue begin"
for pool in dense proprag; do
  for k in 50 100; do
    run_repair_pipeline "${pool}" "${k}"
  done
done
log_msg "SetR Path-A repair queue finished"
