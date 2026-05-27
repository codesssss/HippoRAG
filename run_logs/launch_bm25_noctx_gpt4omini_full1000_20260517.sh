#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
RUN_TAG="${RUN_TAG:-bm25_noctx_gpt4omini_full1000_20260517}"
OUT_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa musique nq popqa}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
POOL_K="${POOL_K:-200}"
QA_TOP_K="${QA_TOP_K:-5}"
READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
READER_MAX_NEW_TOKENS="${READER_MAX_NEW_TOKENS:-none}"
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
  "${OUT_ROOT}/pools/bm25" \
  "${OUT_ROOT}/reader_inputs/bm25" \
  "${OUT_ROOT}/reader_inputs/no_context" \
  "${OUT_ROOT}/reader_qa/bm25" \
  "${OUT_ROOT}/reader_qa/no_context" \
  "${OUT_ROOT}/reader_runtime/bm25" \
  "${OUT_ROOT}/reader_runtime/no_context"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/logs/launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${OUT_ROOT}/status/${name}.status"
}

dataset_file_stem() {
  case "$1" in
    nq|natural_questions) printf 'nq_rear\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

display_dataset_name() {
  case "$1" in
    nq_rear) printf 'nq\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

run_logged() {
  local label="$1"
  local status_name="$2"
  local log_path="$3"
  shift 3

  write_status "${status_name}" "START ${label} log=${log_path}"
  (
    printf '[START] %s time=%s\n' "${label}" "$(date -Is)"
    cd "${ROOT_DIR}" || exit 1
    "$@"
    rc=$?
    printf '[EXIT] %s rc=%s time=%s\n' "${label}" "${rc}" "$(date -Is)"
    exit "${rc}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    write_status "${status_name}" "DONE ${label}"
  else
    write_status "${status_name}" "FAILED code=${code} ${label} log=${log_path}"
  fi
  return "${code}"
}

bm25_pool_path() {
  local dataset="$1"
  printf '%s/pools/bm25/%s_bm25_pool%s_limit%s.json' "${OUT_ROOT}" "${dataset}" "${POOL_K}" "${MAX_QUERIES}"
}

bm25_reader_input_path() {
  local dataset="$1"
  printf '%s/reader_inputs/bm25/%s_bm25_pool%s_reader_input.json' "${OUT_ROOT}" "${dataset}" "${POOL_K}"
}

bm25_reader_report_path() {
  local display="$1"
  printf '%s/reader_qa/bm25/%s_bm25_gpt4omini_reader_top%s_full1000.json' "${OUT_ROOT}" "${display}" "${QA_TOP_K}"
}

bm25_reader_report_md_path() {
  local display="$1"
  printf '%s/reader_qa/bm25/%s_bm25_gpt4omini_reader_top%s_full1000.md' "${OUT_ROOT}" "${display}" "${QA_TOP_K}"
}

noctx_input_path() {
  local dataset="$1"
  printf '%s/reader_inputs/no_context/%s_gpt4omini_no_context_reader_input.json' "${OUT_ROOT}" "${dataset}"
}

noctx_reader_report_path() {
  local display="$1"
  printf '%s/reader_qa/no_context/%s_gpt4omini_no_context_reader_full1000.json' "${OUT_ROOT}" "${display}"
}

noctx_reader_report_md_path() {
  local display="$1"
  printf '%s/reader_qa/no_context/%s_gpt4omini_no_context_reader_full1000.md' "${OUT_ROOT}" "${display}"
}

export_bm25_dataset() {
  local dataset="$1"
  local output_json log_path
  output_json="$(bm25_pool_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/export_bm25_${dataset}.log"
  if [[ -s "${output_json}" ]]; then
    write_status "export_bm25_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  run_logged "export_bm25 dataset=${dataset}" "export_bm25_${dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" scripts/export_bm25_pool.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --pool_k "${POOL_K}" \
      --output_json "${output_json}"
}

wrap_bm25_dataset() {
  local dataset="$1"
  local input_json log_path
  input_json="$(bm25_reader_input_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/wrap_bm25_${dataset}.log"
  if [[ -s "${input_json}" ]]; then
    write_status "wrap_bm25_${dataset}" "SKIP existing input=${input_json}"
    return 0
  fi
  run_logged "wrap_bm25 dataset=${dataset}" "wrap_bm25_${dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" scripts/make_external_pool_reader_inputs.py \
      --pool-json "$(bm25_pool_path "${dataset}")" \
      --method-name "bm25" \
      --output-json "${input_json}"
}

run_bm25_reader_dataset() {
  local dataset="$1"
  local display input_json output_json output_md log_path
  display="$(display_dataset_name "${dataset}")"
  input_json="$(bm25_reader_input_path "${dataset}")"
  output_json="$(bm25_reader_report_path "${display}")"
  output_md="$(bm25_reader_report_md_path "${display}")"
  log_path="${OUT_ROOT}/logs/reader_bm25_${dataset}.log"
  if [[ -s "${output_json}" ]]; then
    write_status "reader_bm25_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  run_logged "reader_bm25 dataset=${dataset}" "reader_bm25_${dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py \
      --input-files "${input_json}" \
      --method-name bm25 \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --doc-source external_pool_topk_docs \
      --save-dir "${OUT_ROOT}/reader_runtime/bm25/${display}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens "${READER_MAX_NEW_TOKENS}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${output_json}" \
      --output-md "${output_md}"
}

make_noctx_dataset() {
  local dataset="$1"
  local input_json log_path
  input_json="$(noctx_input_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/make_no_context_${dataset}.log"
  if [[ -s "${input_json}" ]]; then
    write_status "make_no_context_${dataset}" "SKIP existing input=${input_json}"
    return 0
  fi
  run_logged "make_no_context dataset=${dataset}" "make_no_context_${dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" scripts/make_no_context_reader_inputs.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --method-name gpt4omini_no_context \
      --output-json "${input_json}"
}

run_noctx_reader_dataset() {
  local dataset="$1"
  local display input_json output_json output_md log_path
  display="$(display_dataset_name "${dataset}")"
  input_json="$(noctx_input_path "${dataset}")"
  output_json="$(noctx_reader_report_path "${display}")"
  output_md="$(noctx_reader_report_md_path "${display}")"
  log_path="${OUT_ROOT}/logs/reader_no_context_${dataset}.log"
  if [[ -s "${output_json}" ]]; then
    write_status "reader_no_context_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  run_logged "reader_no_context dataset=${dataset}" "reader_no_context_${dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py \
      --input-files "${input_json}" \
      --method-name gpt4omini_no_context \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --doc-source saved_docs \
      --save-dir "${OUT_ROOT}/reader_runtime/no_context/${display}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens "${READER_MAX_NEW_TOKENS}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${output_json}" \
      --output-md "${output_md}"
}

run_dataset() {
  local requested="$1"
  local dataset
  dataset="$(dataset_file_stem "${requested}")"
  write_status "lane_${dataset}" "START requested=${requested}"
  log_msg "START dataset=${dataset} requested=${requested}"
  export_bm25_dataset "${dataset}" || return 1
  wrap_bm25_dataset "${dataset}" || return 1
  make_noctx_dataset "${dataset}" || return 1
  run_bm25_reader_dataset "${dataset}" || return 1
  run_noctx_reader_dataset "${dataset}" || return 1
  write_status "lane_${dataset}" "DONE requested=${requested}"
  log_msg "DONE dataset=${dataset}"
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG} datasets=${DATASETS}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: BM25 retrieval has no LLM; pure GPT-4o-mini has no retrieved context; all QA uses GPT-4o-mini reader"
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    write_status "launcher" "FAILED missing OPENAI_API_KEY expected_secret_env=${SECRET_ENV}"
    log_msg "FAILED missing OPENAI_API_KEY"
    return 1
  fi
  for dataset in ${DATASETS}; do
    run_dataset "${dataset}" || {
      write_status "launcher" "FAILED dataset=${dataset}"
      return 1
    }
  done
  write_status "launcher" "DONE run_tag=${RUN_TAG}"
  log_msg "DONE ${RUN_TAG}"
}

main "$@"
