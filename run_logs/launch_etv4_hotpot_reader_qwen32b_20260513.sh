#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"

GRAPH_DIR="${ROOT_DIR}/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512"
READER_DIR="${ROOT_DIR}/run_logs/etv4_clean_mainline_qwen32b_reader_gpt4omini_20260512"
RUN_DIR="${ROOT_DIR}/run_logs/etv4_hotpot_reader_qwen32b_gpt4omini_20260513"

DATASET="hotpotqa"
RETRIEVAL_JSON="${GRAPH_DIR}/${DATASET}/reports/${DATASET}_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json"
OUTPUT_JSON="${READER_DIR}/reports/${DATASET}_etv4_qwen32b_gpt4omini_reader_qa.json"
OUTPUT_MD="${READER_DIR}/reports/${DATASET}_etv4_qwen32b_gpt4omini_reader_qa.md"
LOG_PATH="${READER_DIR}/logs/${DATASET}.log"

MAX_QUERIES="${MAX_QUERIES:-1000}"
QA_TOP_K="${QA_TOP_K:-5}"
READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
KEY_SOURCE_SESSION="${KEY_SOURCE_SESSION:-baseline_qwen32b_top200_20260512}"

mkdir -p \
  "${RUN_DIR}/logs" \
  "${RUN_DIR}/status" \
  "${READER_DIR}/logs" \
  "${READER_DIR}/reports" \
  "${READER_DIR}/reader_runtime/${DATASET}"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${RUN_DIR}/logs/launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${RUN_DIR}/status/${name}.status"
}

load_api_key_from_tmux() {
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    return 0
  fi
  if ! command -v tmux >/dev/null 2>&1; then
    return 1
  fi
  local line
  line="$(tmux show-environment -t "${KEY_SOURCE_SESSION}" OPENAI_API_KEY 2>/dev/null || true)"
  if [[ "${line}" == OPENAI_API_KEY=* ]]; then
    export OPENAI_API_KEY="${line#OPENAI_API_KEY=}"
  fi
  [[ -n "${OPENAI_API_KEY:-}" ]]
}

main() {
  write_status "launcher" "START dataset=${DATASET}"
  log_msg "BEGIN ETV4 ${DATASET} Qwen32B graph GPT-4o-mini reader"

  if [[ ! -s "${RETRIEVAL_JSON}" ]]; then
    write_status "launcher" "FAILED missing retrieval=${RETRIEVAL_JSON}"
    log_msg "FAILED missing retrieval=${RETRIEVAL_JSON}"
    return 1
  fi
  if [[ -s "${OUTPUT_JSON}" ]]; then
    write_status "launcher" "SKIP existing output=${OUTPUT_JSON}"
    log_msg "SKIP existing output=${OUTPUT_JSON}"
    return 0
  fi
  if ! load_api_key_from_tmux; then
    write_status "launcher" "FAILED missing OPENAI_API_KEY"
    log_msg "FAILED missing OPENAI_API_KEY"
    return 1
  fi

  write_status "reader" "START retrieval=${RETRIEVAL_JSON} output=${OUTPUT_JSON}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_reader_qa.py \
      --retrieval-reports "${RETRIEVAL_JSON}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${READER_DIR}/reader_runtime/${DATASET}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens none \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${OUTPUT_JSON}" \
      --output-md "${OUTPUT_MD}"
  ) > "${LOG_PATH}" 2>&1
  local code=$?

  if [[ "${code}" -eq 0 && -s "${OUTPUT_JSON}" ]]; then
    write_status "reader" "DONE output=${OUTPUT_JSON}"
    write_status "launcher" "DONE output=${OUTPUT_JSON}"
    log_msg "DONE ETV4 ${DATASET} reader output=${OUTPUT_JSON}"
  else
    write_status "reader" "FAILED code=${code} log=${LOG_PATH}"
    write_status "launcher" "FAILED reader code=${code}"
    log_msg "FAILED ETV4 ${DATASET} reader code=${code} log=${LOG_PATH}"
    return "${code}"
  fi
}

main "$@"
