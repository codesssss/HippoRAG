#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
RUN_TAG="baseline_qwen32b_graph_top200_gpt4omini_full_20260512"
OUT_DIR="${ROOT_DIR}/run_logs/${RUN_TAG}"

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
POOL_K="${POOL_K:-200}"
QA_TOP_K="${QA_TOP_K:-5}"

READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

mkdir -p \
  "${OUT_DIR}/logs" \
  "${OUT_DIR}/status" \
  "${OUT_DIR}/reader_qa/hipporag" \
  "${OUT_DIR}/reader_runtime/hipporag"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_DIR}/logs/hipporag_reader_parallel.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${OUT_DIR}/status/${name}.status"
}

reader_input_path() {
  local dataset="$1"
  echo "${OUT_DIR}/reader_inputs/hipporag/${dataset}_hipporag_qwen32b_nothink_pool${POOL_K}_reader_input.json"
}

reader_report_path() {
  local dataset="$1"
  echo "${OUT_DIR}/reader_qa/hipporag/${dataset}_hipporag_qwen32b_nothink_top${POOL_K}_gpt4omini_reader_top${QA_TOP_K}.json"
}

reader_report_md_path() {
  local dataset="$1"
  echo "${OUT_DIR}/reader_qa/hipporag/${dataset}_hipporag_qwen32b_nothink_top${POOL_K}_gpt4omini_reader_top${QA_TOP_K}.md"
}

run_reader_dataset() {
  local dataset="$1"
  local input_json output_json output_md log_path
  input_json="$(reader_input_path "${dataset}")"
  output_json="$(reader_report_path "${dataset}")"
  output_md="$(reader_report_md_path "${dataset}")"
  log_path="${OUT_DIR}/logs/reader_hipporag_${dataset}.log"

  if [[ -s "${output_json}" ]]; then
    write_status "reader_hipporag_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP HippoRAG reader dataset=${dataset}"
    return 0
  fi
  if [[ ! -s "${input_json}" ]]; then
    write_status "reader_hipporag_${dataset}" "FAILED missing input=${input_json}"
    log_msg "FAILED HippoRAG reader dataset=${dataset}: missing input"
    return 1
  fi

  write_status "reader_hipporag_${dataset}" "START input=${input_json} output=${output_json}"
  log_msg "START HippoRAG reader dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py \
      --input-files "${input_json}" \
      --method-name "hipporag_qwen32b_top${POOL_K}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --doc-source external_pool_topk_docs \
      --save-dir "${OUT_DIR}/reader_runtime/hipporag/${dataset}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens none \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${output_json}" \
      --output-md "${output_md}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "reader_hipporag_${dataset}" "DONE output=${output_json}"
    log_msg "DONE HippoRAG reader dataset=${dataset}"
  else
    write_status "reader_hipporag_${dataset}" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED HippoRAG reader dataset=${dataset} code=${code}"
    return "${code}"
  fi
}

main() {
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    log_msg "FAILED missing OPENAI_API_KEY"
    return 1
  fi

  log_msg "BEGIN HippoRAG parallel reader: datasets=${DATASETS}, reader=${READER_LLM_NAME}, top=${QA_TOP_K}, max_new_tokens=none"
  local dataset
  local pids=()
  for dataset in ${DATASETS}; do
    run_reader_dataset "${dataset}" &
    pids+=("$!")
  done

  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  if [[ "${rc}" -eq 0 ]]; then
    log_msg "DONE HippoRAG parallel reader"
  else
    log_msg "FAILED HippoRAG parallel reader"
  fi
  return "${rc}"
}

main "$@"
