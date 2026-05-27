#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/pcec_fresh_e2e_reader_qa_qwen32b_20260511"
RETRIEVAL_ROOT="${ROOT_DIR}/run_logs/pcec_fresh_e2e_reader_qa/retrieval_reports_reader_docids"
MAX_QUERIES="${MAX_QUERIES:-1000}"
QA_TOP_K="${QA_TOP_K:-5}"
LLM_NAME="${LLM_NAME:-qwen3-32b-judge}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8045/v1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
EMBEDDING_NAME="${EMBEDDING_NAME:-nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/status" "${OUT_DIR}/reader_runtime" "${OUT_DIR}/reports"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_DIR}/logs/launcher.log"
}

retrieval_report() {
  local dataset="$1"
  echo "${RETRIEVAL_ROOT}/${dataset}_pcec_fresh_e2e_prefix4_residual1_pool100_limit1000.json"
}

check_endpoint() {
  local name="$1"
  local url="$2"
  if curl -fsS "${url}" >/dev/null 2>&1; then
    log_msg "OK endpoint ${name}: ${url}"
    return 0
  fi
  log_msg "FAILED endpoint ${name}: ${url}"
  return 1
}

run_dataset() {
  local dataset="$1"
  local retrieval_json
  retrieval_json="$(retrieval_report "${dataset}")"
  local output_json="${OUT_DIR}/reports/${dataset}_pcec_fresh_e2e_reader_qa_qwen32b_full1000.json"
  local output_md="${OUT_DIR}/reports/${dataset}_pcec_fresh_e2e_reader_qa_qwen32b_full1000.md"
  local status_path="${OUT_DIR}/status/${dataset}.status"
  local log_path="${OUT_DIR}/logs/${dataset}.log"

  if [[ ! -s "${retrieval_json}" ]]; then
    echo "[FAILED] dataset=${dataset} missing_retrieval=${retrieval_json} time=$(date -Is)" > "${status_path}"
    log_msg "FAILED dataset=${dataset} missing retrieval report: ${retrieval_json}"
    return 2
  fi

  echo "[START] dataset=${dataset} model=${LLM_NAME} max_queries=${MAX_QUERIES} time=$(date -Is)" > "${status_path}"
  log_msg "START reader QA dataset=${dataset} model=${LLM_NAME} base_url=${LLM_BASE_URL}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" \
      evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_reader_qa.py \
      --retrieval-reports "${retrieval_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${OUT_DIR}/reader_runtime/${dataset}" \
      --llm-name "${LLM_NAME}" \
      --llm-base-url "${LLM_BASE_URL}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --qwen-disable-thinking \
      --output-json "${output_json}" \
      --output-md "${output_md}"
  ) > "${log_path}" 2>&1
  local code=$?

  if [[ "${code}" -eq 0 ]] && [[ -s "${output_json}" ]]; then
    echo "[DONE] dataset=${dataset} output=${output_json} time=$(date -Is)" > "${status_path}"
    log_msg "DONE reader QA dataset=${dataset} output=${output_json}"
  else
    echo "[FAILED] dataset=${dataset} code=${code} output=${output_json} time=$(date -Is)" > "${status_path}"
    log_msg "FAILED reader QA dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

main() {
  echo "[START] pcec_qwen32b_reader_full1000 time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
  log_msg "PCEC Qwen3-32B reader QA full1000 begin"
  log_msg "outputs: ${OUT_DIR}"
  log_msg "datasets run concurrently against one 32B vLLM endpoint"

  local preflight=0
  check_endpoint "qwen32b" "${LLM_BASE_URL}/models" || preflight=1
  check_endpoint "nv_embed" "http://localhost:8019/v1/models" || preflight=1
  if [[ "${preflight}" -ne 0 ]]; then
    echo "[FAILED] pcec_qwen32b_reader_full1000 preflight time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
    return 1
  fi

  run_dataset 2wikimultihopqa &
  local pid_2wiki=$!
  run_dataset hotpotqa &
  local pid_hotpot=$!
  run_dataset musique &
  local pid_musique=$!

  local status=0
  wait "${pid_2wiki}" || status=1
  wait "${pid_hotpot}" || status=1
  wait "${pid_musique}" || status=1

  if [[ "${status}" -eq 0 ]]; then
    echo "[DONE] pcec_qwen32b_reader_full1000 time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
    log_msg "PCEC Qwen3-32B reader QA full1000 complete"
  else
    echo "[FAILED] pcec_qwen32b_reader_full1000 status=${status} time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
    log_msg "PCEC Qwen3-32B reader QA full1000 finished with failures"
  fi
  return "${status}"
}

main "$@"
