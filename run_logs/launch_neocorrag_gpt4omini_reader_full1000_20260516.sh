#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
RUN_TAG="neocorrag_reader_gpt4omini_full1000_20260516"
OUT_DIR="${ROOT_DIR}/run_logs/${RUN_TAG}"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
QA_TOP_K="${QA_TOP_K:-5}"
READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-none}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

mkdir -p \
  "${OUT_DIR}/logs" \
  "${OUT_DIR}/status" \
  "${OUT_DIR}/reader_inputs" \
  "${OUT_DIR}/reader_runtime/neocorrag" \
  "${OUT_DIR}/reader_qa/neocorrag"

if [[ -s "${SECRET_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRET_ENV}"
  set +a
fi

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_DIR}/logs/launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${OUT_DIR}/status/${name}.status"
}

reader_input_path() {
  local dataset="$1"
  printf '%s/reader_inputs/%s_neocorrag_qwen32b_no_think_top5_reader_input.json' \
    "${OUT_DIR}" "${dataset}"
}

reader_report_path() {
  local dataset="$1"
  printf '%s/reader_qa/neocorrag/%s_neocorrag_qwen32b_no_think_gpt4omini_reader_top%s_full1000.json' \
    "${OUT_DIR}" "${dataset}" "${QA_TOP_K}"
}

reader_report_md_path() {
  local dataset="$1"
  printf '%s/reader_qa/neocorrag/%s_neocorrag_qwen32b_no_think_gpt4omini_reader_top%s_full1000.md' \
    "${OUT_DIR}" "${dataset}" "${QA_TOP_K}"
}

require_api_key() {
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    write_status "launcher" "FAILED missing OPENAI_API_KEY expected_secret_env=${SECRET_ENV}"
    log_msg "FAILED missing OPENAI_API_KEY"
    return 1
  fi
}

run_reader_dataset() {
  local dataset="$1"
  local input_json output_json output_md log_path
  input_json="$(reader_input_path "${dataset}")"
  output_json="$(reader_report_path "${dataset}")"
  output_md="$(reader_report_md_path "${dataset}")"
  log_path="${OUT_DIR}/logs/reader_neocorrag_${dataset}.log"

  if [[ -s "${output_json}" ]]; then
    write_status "reader_neocorrag_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP NeocorRAG reader dataset=${dataset}"
    return 0
  fi
  if [[ ! -s "${input_json}" ]]; then
    write_status "reader_neocorrag_${dataset}" "FAILED missing input=${input_json}"
    log_msg "FAILED NeocorRAG reader dataset=${dataset}: missing input=${input_json}"
    return 2
  fi

  write_status "reader_neocorrag_${dataset}" "START input=${input_json} output=${output_json}"
  log_msg "START NeocorRAG GPT-4o-mini reader dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py \
      --input-files "${input_json}" \
      --method-name "neocorrag_qwen32b_no_think_top${QA_TOP_K}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --doc-source saved_docs \
      --save-dir "${OUT_DIR}/reader_runtime/neocorrag/${dataset}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${output_json}" \
      --output-md "${output_md}"
  ) > "${log_path}" 2>&1
  local code=$?

  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "reader_neocorrag_${dataset}" "DONE output=${output_json}"
    log_msg "DONE NeocorRAG GPT-4o-mini reader dataset=${dataset}"
  else
    write_status "reader_neocorrag_${dataset}" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED NeocorRAG GPT-4o-mini reader dataset=${dataset} code=${code}"
    return "${code}"
  fi
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG} datasets=${DATASETS}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Reader: ${READER_LLM_NAME} at ${READER_LLM_BASE_URL}; top=${QA_TOP_K}; max_new_tokens=${MAX_NEW_TOKENS}"

  require_api_key || return 1

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
    write_status "launcher" "DONE run_tag=${RUN_TAG}"
    log_msg "DONE ${RUN_TAG}"
  else
    write_status "launcher" "FAILED run_tag=${RUN_TAG} rc=${rc}"
    log_msg "FAILED ${RUN_TAG} rc=${rc}"
  fi
  return "${rc}"
}

main "$@"
