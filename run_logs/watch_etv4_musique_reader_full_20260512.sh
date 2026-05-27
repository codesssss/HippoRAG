#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"

GRAPH_DIR="${ROOT_DIR}/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512"
READER_DIR="${ROOT_DIR}/run_logs/etv4_clean_mainline_qwen32b_reader_gpt4omini_20260512"
RUN_DIR="${ROOT_DIR}/run_logs/watch_etv4_musique_reader_full_20260512"

DATASET="musique"
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
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p \
  "${RUN_DIR}/logs" \
  "${RUN_DIR}/status" \
  "${READER_DIR}/logs" \
  "${READER_DIR}/reports" \
  "${READER_DIR}/reader_runtime/${DATASET}"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${RUN_DIR}/logs/watcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${RUN_DIR}/status/${name}.status"
}

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  log_msg "FAILED missing OPENAI_API_KEY for ${READER_LLM_NAME}"
  write_status "watcher" "FAILED missing OPENAI_API_KEY"
  exit 1
fi

log_msg "BEGIN watch ETV4 MuSiQue retrieval=${RETRIEVAL_JSON}"
write_status "watcher" "WAIT retrieval=${RETRIEVAL_JSON}"

while [[ ! -s "${RETRIEVAL_JSON}" ]]; do
  log_msg "WAIT missing retrieval=${RETRIEVAL_JSON}"
  sleep "${POLL_SECONDS}"
done

write_status "watcher" "FOUND retrieval=${RETRIEVAL_JSON}"
log_msg "FOUND retrieval=${RETRIEVAL_JSON}"

if [[ -s "${OUTPUT_JSON}" ]]; then
  write_status "reader" "SKIP existing output=${OUTPUT_JSON}"
  log_msg "SKIP reader existing output=${OUTPUT_JSON}"
  write_status "watcher" "DONE existing output=${OUTPUT_JSON}"
  exit 0
fi

write_status "reader" "START retrieval=${RETRIEVAL_JSON} output=${OUTPUT_JSON}"
log_msg "START reader output=${OUTPUT_JSON}"
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
code=$?

if [[ "${code}" -eq 0 && -s "${OUTPUT_JSON}" ]]; then
  write_status "reader" "DONE output=${OUTPUT_JSON}"
  write_status "watcher" "DONE output=${OUTPUT_JSON}"
  log_msg "DONE reader output=${OUTPUT_JSON}"
else
  write_status "reader" "FAILED code=${code} log=${LOG_PATH}"
  write_status "watcher" "FAILED reader code=${code}"
  log_msg "FAILED reader code=${code} log=${LOG_PATH}"
  exit "${code}"
fi
