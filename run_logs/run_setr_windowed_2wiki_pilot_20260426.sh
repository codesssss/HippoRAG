#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
OUT_DIR="${ROOT_DIR}/run_logs/setr_windowed_2wiki_pilot_20260426"
REPORT_DIR="${ROOT_DIR}/reports/setr_pathA"
SAVE_DIR="outputs_step0_general_nvembed"
LLM_BASE_URL="http://localhost:8041/v1"
LLM_MODEL="qwen3-8b-train"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"
DATASET="2wikimultihopqa"
LIMIT=100
WINDOW_SIZE=20
DOC_MAX_CHARS=768
NUM_WORKERS=2

mkdir -p "${OUT_DIR}" "${REPORT_DIR}"

log_msg() {
  echo "[$(date -Is)] $*" | tee -a "${OUT_DIR}/launcher.status"
}

run_cmd() {
  local label="$1"
  shift
  local log_path="${OUT_DIR}/${label}.log"
  local status_path="${OUT_DIR}/${label}.status"
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

run_windowed_pipeline() {
  local pool="$1"
  local k="$2"
  local pool_json
  pool_json="$(pool_path "${pool}")"
  local prefix="${OUT_DIR}/setr_windowed_${DATASET}_${pool}_k${k}_limit${LIMIT}"
  local report_prefix="${REPORT_DIR}/setr_windowed_${DATASET}_${pool}_k${k}_limit${LIMIT}"
  local selection_jsonl="${prefix}.selection.jsonl"
  local selected_pool_json="${prefix}.selected_pool.json"

  run_cmd "windowed_select_${pool}_k${k}" \
    .venv-hipporag/bin/python scripts/run_setr_windowed_selector.py \
      --pool_json "${pool_json}" \
      --output_selection_jsonl "${selection_jsonl}" \
      --pool_k "${k}" \
      --window_size "${WINDOW_SIZE}" \
      --stage1_max_per_window 5 \
      --stage2_max_candidates 25 \
      --final_max_count 5 \
      --doc_max_chars "${DOC_MAX_CHARS}" \
      --llm_base_url "${LLM_BASE_URL}" \
      --model "${LLM_MODEL}" \
      --max_tokens 512 \
      --temperature 0 \
      --limit "${LIMIT}" \
      --num_workers "${NUM_WORKERS}" \
      --resume

  run_cmd "windowed_apply_${pool}_k${k}" \
    .venv-hipporag/bin/python scripts/apply_setr_selection_to_pool.py \
      --pool_json "${pool_json}" \
      --selection_jsonl "${selection_jsonl}" \
      --output_pool_json "${selected_pool_json}" \
      --limit "${LIMIT}"

  run_cmd "windowed_eval_${pool}_k${k}" \
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
      --external_pool_source_name "setr_windowed_iri_2stage_${pool}_k${k}" \
      --setwise_selector none \
      --output_json "${report_prefix}.eval.json"
}

log_msg "SetR-windowed 2Wiki pilot queue begin"
for pool in dense proprag; do
  for k in 50 100; do
    run_windowed_pipeline "${pool}" "${k}"
  done
done
log_msg "SetR-windowed 2Wiki pilot queue finished"
