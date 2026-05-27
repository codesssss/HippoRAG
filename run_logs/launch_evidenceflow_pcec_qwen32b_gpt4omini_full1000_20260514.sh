#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"

RUN_TAG="evidenceflow_pcec_qwen32b_gpt4omini_full1000_20260514"
OUT_DIR="${ROOT_DIR}/run_logs/${RUN_TAG}"
RETRIEVAL_ROOT="${ROOT_DIR}/run_logs/pcec_qwen32b_graph_e2e_full1000_20260511/retrieval_reports_reader_docids"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

if [[ -s "${SECRET_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRET_ENV}"
  set +a
fi

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
QA_TOP_K="${QA_TOP_K:-5}"
READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

mkdir -p \
  "${OUT_DIR}/logs" \
  "${OUT_DIR}/status" \
  "${OUT_DIR}/reader_runtime" \
  "${OUT_DIR}/reports"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_DIR}/logs/launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${OUT_DIR}/status/${name}.status"
}

require_api_key() {
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    write_status "launcher" "FAILED missing OPENAI_API_KEY"
    log_msg "FAILED missing OPENAI_API_KEY; expected SECRET_ENV=${SECRET_ENV}"
    return 1
  fi
}

retrieval_report() {
  local dataset="$1"
  printf '%s/%s_pcec_fresh_e2e_prefix4_residual1_pool100_limit1000.json' \
    "${RETRIEVAL_ROOT}" "${dataset}"
}

output_json() {
  local dataset="$1"
  printf '%s/reports/%s_evidenceflow_qwen32b_gpt4omini_reader_qa_full1000.json' \
    "${OUT_DIR}" "${dataset}"
}

output_md() {
  local dataset="$1"
  printf '%s/reports/%s_evidenceflow_qwen32b_gpt4omini_reader_qa_full1000.md' \
    "${OUT_DIR}" "${dataset}"
}

run_dataset() {
  local dataset="$1"
  local retrieval_json output_json_path output_md_path status_path log_path
  retrieval_json="$(retrieval_report "${dataset}")"
  output_json_path="$(output_json "${dataset}")"
  output_md_path="$(output_md "${dataset}")"
  status_path="${OUT_DIR}/status/${dataset}.status"
  log_path="${OUT_DIR}/logs/${dataset}.log"

  if [[ ! -s "${retrieval_json}" ]]; then
    write_status "${dataset}" "FAILED missing_retrieval=${retrieval_json}"
    log_msg "FAILED dataset=${dataset} missing retrieval=${retrieval_json}"
    return 2
  fi
  if [[ -s "${output_json_path}" ]]; then
    write_status "${dataset}" "SKIP existing output=${output_json_path}"
    log_msg "SKIP dataset=${dataset} existing output=${output_json_path}"
    return 0
  fi

  write_status "${dataset}" "START retrieval=${retrieval_json} output=${output_json_path}"
  log_msg "START EvidenceFlow/PCEC GPT-4o-mini reader dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_reader_qa.py \
      --retrieval-reports "${retrieval_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${OUT_DIR}/reader_runtime/${dataset}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${output_json_path}" \
      --output-md "${output_md_path}"
  ) > "${log_path}" 2>&1
  local code=$?

  if [[ "${code}" -eq 0 && -s "${output_json_path}" ]]; then
    write_status "${dataset}" "DONE output=${output_json_path}"
    log_msg "DONE EvidenceFlow/PCEC GPT-4o-mini reader dataset=${dataset}"
  else
    write_status "${dataset}" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED EvidenceFlow/PCEC GPT-4o-mini reader dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG} datasets=${DATASETS}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Retrieval source: ${RETRIEVAL_ROOT}"
  log_msg "Reader: ${READER_LLM_NAME} at ${READER_LLM_BASE_URL}"

  require_api_key || return 1

  local pids=()
  local dataset
  for dataset in ${DATASETS}; do
    run_dataset "${dataset}" &
    pids+=("$!")
  done

  local status=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done

  if [[ "${status}" -eq 0 ]]; then
    write_status "launcher" "DONE run_tag=${RUN_TAG}"
    log_msg "DONE ${RUN_TAG}"
  else
    write_status "launcher" "FAILED status=${status}"
    log_msg "FAILED ${RUN_TAG} status=${status}"
  fi
  return "${status}"
}

main "$@"
