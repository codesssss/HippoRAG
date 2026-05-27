#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"

RUN_TAG="dense_entry_gpt4omini_reader_full1000_20260516"
OUT_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
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

if [[ -s "${SECRET_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRET_ENV}"
  set +a
fi

mkdir -p \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/status" \
  "${OUT_ROOT}/reader_qa/reports" \
  "${OUT_ROOT}/reader_qa/runtime"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/logs/launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${OUT_ROOT}/status/${name}.status"
}

retrieval_report_path() {
  local dataset="$1"
  printf '%s/retrieval_reports/%s_dense_entry_only_top5_limit1000.json' "${OUT_ROOT}" "${dataset}"
}

reader_output_json() {
  local dataset="$1"
  printf '%s/reader_qa/reports/%s_dense_entry_gpt4omini_reader_qa_full1000.json' "${OUT_ROOT}" "${dataset}"
}

reader_output_md() {
  local dataset="$1"
  printf '%s/reader_qa/reports/%s_dense_entry_gpt4omini_reader_qa_full1000.md' "${OUT_ROOT}" "${dataset}"
}

run_reader_dataset() {
  local dataset="$1"
  local retrieval_json output_json output_md log_path status_name
  retrieval_json="$(retrieval_report_path "${dataset}")"
  output_json="$(reader_output_json "${dataset}")"
  output_md="$(reader_output_md "${dataset}")"
  log_path="${OUT_ROOT}/logs/reader_dense_entry_${dataset}.log"
  status_name="reader_dense_entry_${dataset}"

  if [[ ! -s "${retrieval_json}" ]]; then
    write_status "${status_name}" "FAILED missing_retrieval=${retrieval_json}"
    log_msg "FAILED Dense-entry reader dataset=${dataset}: missing retrieval=${retrieval_json}"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "${status_name}" "SKIP existing output=${output_json}"
    log_msg "SKIP Dense-entry reader dataset=${dataset}"
    return 0
  fi

  write_status "${status_name}" "START retrieval=${retrieval_json} output=${output_json}"
  log_msg "START Dense-entry GPT-4o-mini reader dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_reader_qa.py \
      --retrieval-reports "${retrieval_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${OUT_ROOT}/reader_qa/runtime/${dataset}" \
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
    write_status "${status_name}" "DONE output=${output_json}"
    log_msg "DONE Dense-entry GPT-4o-mini reader dataset=${dataset}"
  else
    write_status "${status_name}" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED Dense-entry GPT-4o-mini reader dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG} datasets=${DATASETS}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: dense entry top-5 from same EvidenceFlow source run; GPT-4o-mini reader; top-5 reader context"

  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    write_status "launcher" "FAILED missing OPENAI_API_KEY"
    log_msg "FAILED missing OPENAI_API_KEY; expected SECRET_ENV=${SECRET_ENV}"
    return 1
  fi

  local pids=()
  local dataset pid rc=0
  for dataset in ${DATASETS}; do
    run_reader_dataset "${dataset}" &
    pids+=("$!")
  done
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
