#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/mnt/nvme/code/HippoRAG}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/run_logs/pcec_residual_budget_sweep_retrieval_only_20260519}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/evals}"
READER_ROOT="${READER_ROOT:-${RUN_ROOT}/reader_qa}"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
PREFIXES="${PREFIXES:-3 0}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
QA_TOP_K="${QA_TOP_K:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-none}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

mkdir -p "${READER_ROOT}/logs" "${READER_ROOT}/status" "${READER_ROOT}/reports" "${READER_ROOT}/reader_runtime"

if [[ -f "${SECRET_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${SECRET_ENV}"
fi

READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${READER_ROOT}/logs/launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${READER_ROOT}/status/${name}.status"
}

variant_name() {
  case "$1" in
    3) printf 'prefix3_residual2' ;;
    0) printf 'prefix0_residual5' ;;
    *) printf 'prefix%s_residual%s' "$1" "$((QA_TOP_K - $1))" ;;
  esac
}

retrieval_report_path() {
  local dataset="$1"
  local prefix="$2"
  local variant
  variant="$(variant_name "${prefix}")"
  printf '%s/%s_pcec_native_pool_%s_pool100_limit1000.json' "${EVAL_ROOT}" "${dataset}" "${variant}"
}

reader_output_json() {
  local dataset="$1"
  local prefix="$2"
  local variant
  variant="$(variant_name "${prefix}")"
  printf '%s/reports/%s_pcec_native_pool_%s_gpt4omini_reader_qa_full1000.json' "${READER_ROOT}" "${dataset}" "${variant}"
}

reader_output_md() {
  local dataset="$1"
  local prefix="$2"
  local variant
  variant="$(variant_name "${prefix}")"
  printf '%s/reports/%s_pcec_native_pool_%s_gpt4omini_reader_qa_full1000.md' "${READER_ROOT}" "${dataset}" "${variant}"
}

run_reader() {
  local dataset="$1"
  local prefix="$2"
  local variant retrieval_json output_json output_md status_name log_path runtime_dir
  variant="$(variant_name "${prefix}")"
  retrieval_json="$(retrieval_report_path "${dataset}" "${prefix}")"
  output_json="$(reader_output_json "${dataset}" "${prefix}")"
  output_md="$(reader_output_md "${dataset}" "${prefix}")"
  status_name="reader_${variant}_${dataset}"
  log_path="${READER_ROOT}/logs/${status_name}.log"
  runtime_dir="${READER_ROOT}/reader_runtime/${dataset}_${variant}"

  if [[ ! -s "${retrieval_json}" ]]; then
    write_status "${status_name}" "FAILED missing_retrieval=${retrieval_json}"
    log_msg "FAILED ${status_name}: missing retrieval ${retrieval_json}"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "${status_name}" "SKIP existing output=${output_json}"
    log_msg "SKIP ${status_name}: existing output"
    return 0
  fi
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    write_status "${status_name}" "FAILED missing_OPENAI_API_KEY secret=${SECRET_ENV}"
    log_msg "FAILED ${status_name}: missing OPENAI_API_KEY"
    return 3
  fi

  write_status "${status_name}" "START dataset=${dataset} variant=${variant} model=${READER_LLM_NAME}"
  log_msg "START ${status_name}: ${READER_LLM_NAME}, max_new_tokens=${MAX_NEW_TOKENS}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    export OPENAI_API_KEY="${OPENAI_API_KEY}"
    "${PYTHON_BIN}" evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_reader_qa.py \
      --retrieval-reports "${retrieval_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${runtime_dir}" \
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
    log_msg "DONE ${status_name}"
  else
    write_status "${status_name}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED ${status_name}: code=${code}"
  fi
  return "${code}"
}

main() {
  : > "${READER_ROOT}/logs/launcher.log"
  write_status "launcher" "START prefixes=${PREFIXES} datasets=${DATASETS}"
  log_msg "BEGIN residual-budget GPT-4o-mini reader sweep"
  log_msg "Reader=${READER_LLM_NAME} base=${READER_LLM_BASE_URL}; datasets=${DATASETS}; prefixes=${PREFIXES}"

  local preflight=0
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    log_msg "FAILED preflight: missing OPENAI_API_KEY from ${SECRET_ENV}"
    preflight=1
  fi
  if ! curl -fsS "${EMBEDDING_BASE_URL%/embeddings}/models" >/dev/null 2>&1; then
    log_msg "FAILED preflight: embedding endpoint unavailable ${EMBEDDING_BASE_URL}"
    preflight=1
  fi
  if [[ "${preflight}" -ne 0 ]]; then
    write_status "launcher" "FAILED preflight"
    return 1
  fi

  local pids=()
  local dataset prefix
  for prefix in ${PREFIXES}; do
    for dataset in ${DATASETS}; do
      run_reader "${dataset}" "${prefix}" &
      pids+=("$!")
    done
  done

  local status=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  if [[ "${status}" -eq 0 ]]; then
    write_status "launcher" "DONE"
    log_msg "DONE residual-budget GPT-4o-mini reader sweep"
  else
    write_status "launcher" "FAILED status=${status}"
    log_msg "FAILED residual-budget GPT-4o-mini reader sweep status=${status}"
  fi
  return "${status}"
}

main "$@"
