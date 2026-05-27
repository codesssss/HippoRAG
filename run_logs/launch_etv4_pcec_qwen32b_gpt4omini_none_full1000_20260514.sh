#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"

RUN_TAG="etv4_pcec_qwen32b_gpt4omini_none_full1000_20260514"
ETV4_ROOT="${ROOT_DIR}/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512"
PCEC_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
POOL_ROOT="${PCEC_ROOT}/pools"
READER_QA_ROOT="${PCEC_ROOT}/reader_qa"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

if [[ -s "${SECRET_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRET_ENV}"
  set +a
fi

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
PCEC_POOL_K="${PCEC_POOL_K:-100}"
READER_BUDGET_K="${READER_BUDGET_K:-5}"
PREFIX_BUDGET_M="${PREFIX_BUDGET_M:-4}"

GRAPH_LLM_NAME="${GRAPH_LLM_NAME:-qwen3-32b-judge}"
GRAPH_LLM_BASE_URL="${GRAPH_LLM_BASE_URL:-http://localhost:8045/v1}"
UTILITY_LLM_NAME="${UTILITY_LLM_NAME:-qwen3-8b}"
BINDING_MODEL="${BINDING_MODEL:-qwen3-8b-train}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-none}"
QA_TOP_K="${QA_TOP_K:-5}"

REQUIREMENT_ROOT="${REQUIREMENT_ROOT:-${ROOT_DIR}/run_logs/etv3_dbec_latest_full1000_20260510/evals}"

mkdir -p \
  "${PCEC_ROOT}/logs" \
  "${PCEC_ROOT}/status" \
  "${PCEC_ROOT}/evals" \
  "${PCEC_ROOT}/runtime" \
  "${POOL_ROOT}" \
  "${READER_QA_ROOT}/reader_runtime" \
  "${READER_QA_ROOT}/reports"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${PCEC_ROOT}/logs/launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${PCEC_ROOT}/status/${name}.status"
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

require_api_key() {
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    write_status "launcher" "FAILED missing OPENAI_API_KEY"
    log_msg "FAILED missing OPENAI_API_KEY; expected SECRET_ENV=${SECRET_ENV}"
    return 1
  fi
}

etv4_retrieval_report() {
  local dataset="$1"
  printf '%s/%s/reports/%s_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json' \
    "${ETV4_ROOT}" "${dataset}" "${dataset}"
}

etv4_openie_results() {
  local dataset="$1"
  printf '%s/%s/index/openie_results_ner_qwen3-32b-judge.json' \
    "${ETV4_ROOT}" "${dataset}"
}

pool_json_path() {
  local dataset="$1"
  printf '%s/%s_etv4_pool%s_limit%s.json' \
    "${POOL_ROOT}" "${dataset}" "${PCEC_POOL_K}" "${MAX_QUERIES}"
}

requirement_report_path() {
  local dataset="$1"
  printf '%s/%s_etv3_pool%s_dbec_stable_limit%s.json' \
    "${REQUIREMENT_ROOT}" "${dataset}" "${PCEC_POOL_K}" "${MAX_QUERIES}"
}

binding_cache_path() {
  local dataset="$1"
  printf '%s/%s_etv3_pool%s_dbec_latest.binding_cache.json' \
    "${REQUIREMENT_ROOT}" "${dataset}" "${PCEC_POOL_K}"
}

pcec_report_path() {
  local dataset="$1"
  printf '%s/evals/%s_pcec_native_pool_prefix%s_residual%s_pool%s_limit%s.json' \
    "${PCEC_ROOT}" "${dataset}" "${PREFIX_BUDGET_M}" \
    "$((READER_BUDGET_K - PREFIX_BUDGET_M))" "${PCEC_POOL_K}" "${MAX_QUERIES}"
}

reader_output_json() {
  local dataset="$1"
  printf '%s/reports/%s_etv4_pcec_gpt4omini_none_reader_qa_full1000.json' \
    "${READER_QA_ROOT}" "${dataset}"
}

reader_output_md() {
  local dataset="$1"
  printf '%s/reports/%s_etv4_pcec_gpt4omini_none_reader_qa_full1000.md' \
    "${READER_QA_ROOT}" "${dataset}"
}

export_pool_dataset() {
  local dataset="$1"
  local retrieval_json openie_json output_json log_path
  retrieval_json="$(etv4_retrieval_report "${dataset}")"
  openie_json="$(etv4_openie_results "${dataset}")"
  output_json="$(pool_json_path "${dataset}")"
  log_path="${PCEC_ROOT}/logs/01_export_etv4_pool_${dataset}.log"

  if [[ ! -s "${retrieval_json}" ]]; then
    write_status "01_export_pool_${dataset}" "FAILED missing_retrieval=${retrieval_json}"
    log_msg "FAILED export ETV4 pool dataset=${dataset} missing retrieval=${retrieval_json}"
    return 2
  fi
  if [[ ! -s "${openie_json}" ]]; then
    write_status "01_export_pool_${dataset}" "FAILED missing_openie=${openie_json}"
    log_msg "FAILED export ETV4 pool dataset=${dataset} missing openie=${openie_json}"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "01_export_pool_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP export ETV4 pool dataset=${dataset} existing output=${output_json}"
    return 0
  fi

  write_status "01_export_pool_${dataset}" "START dataset=${dataset}"
  log_msg "START export ETV4 pool100 dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    "${PYTHON_BIN}" scripts/export_evidence_transition_pool.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --pool_k "${PCEC_POOL_K}" \
      --data_root reproduce/dataset \
      --retrieval_report "${retrieval_json}" \
      --openie_results "${openie_json}" \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "01_export_pool_${dataset}" "DONE output=${output_json}"
    log_msg "DONE export ETV4 pool100 dataset=${dataset}"
  else
    write_status "01_export_pool_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED export ETV4 pool dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_pcec_dataset() {
  local dataset="$1"
  local pool_json requirement_json binding_json output_json log_path
  pool_json="$(pool_json_path "${dataset}")"
  requirement_json="$(requirement_report_path "${dataset}")"
  binding_json="$(binding_cache_path "${dataset}")"
  output_json="$(pcec_report_path "${dataset}")"
  log_path="${PCEC_ROOT}/logs/02_pcec_native_pool_${dataset}.log"

  if [[ ! -s "${pool_json}" ]]; then
    write_status "02_pcec_${dataset}" "FAILED missing_pool=${pool_json}"
    log_msg "FAILED ETV4+PCEC dataset=${dataset} missing pool=${pool_json}"
    return 2
  fi
  if [[ ! -s "${requirement_json}" ]]; then
    write_status "02_pcec_${dataset}" "FAILED missing_requirement=${requirement_json}"
    log_msg "FAILED ETV4+PCEC dataset=${dataset} missing requirement=${requirement_json}"
    return 2
  fi
  if [[ ! -s "${binding_json}" ]]; then
    write_status "02_pcec_${dataset}" "FAILED missing_binding_cache=${binding_json}"
    log_msg "FAILED ETV4+PCEC dataset=${dataset} missing binding cache=${binding_json}"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "02_pcec_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP ETV4+PCEC dataset=${dataset} existing output=${output_json}"
    return 0
  fi

  write_status "02_pcec_${dataset}" "START dataset=${dataset} pool=${pool_json}"
  log_msg "START native PCEC over ETV4 pool dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" evidence_transition_graphragv4_composition/run_native_pool.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --max-queries "${MAX_QUERIES}" \
      --pool-k "${PCEC_POOL_K}" \
      --reader-budget-k "${READER_BUDGET_K}" \
      --prefix-budget-m "${PREFIX_BUDGET_M}" \
      --pool-json "${pool_json}" \
      --requirement-report "${requirement_json}" \
      --binding-cache-path "${binding_json}" \
      --output-json "${output_json}" \
      --output-root "${PCEC_ROOT}" \
      --data-root reproduce/dataset \
      --save-dir "${PCEC_ROOT}/runtime/${dataset}" \
      --llm-name "${UTILITY_LLM_NAME}" \
      --llm-request-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${GRAPH_LLM_BASE_URL}" \
      --llm-binding-model "${BINDING_MODEL}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --qwen-disable-thinking
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "02_pcec_${dataset}" "DONE output=${output_json}"
    log_msg "DONE native PCEC over ETV4 pool dataset=${dataset}"
  else
    write_status "02_pcec_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED native PCEC over ETV4 pool dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_reader_qa_dataset() {
  local dataset="$1"
  local retrieval_json output_json output_md log_path
  retrieval_json="$(pcec_report_path "${dataset}")"
  output_json="$(reader_output_json "${dataset}")"
  output_md="$(reader_output_md "${dataset}")"
  log_path="${PCEC_ROOT}/logs/03_reader_qa_${dataset}.log"

  if [[ ! -s "${retrieval_json}" ]]; then
    write_status "03_reader_qa_${dataset}" "FAILED missing_retrieval=${retrieval_json}"
    log_msg "FAILED ETV4+PCEC reader QA dataset=${dataset} missing retrieval=${retrieval_json}"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "03_reader_qa_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP ETV4+PCEC reader QA dataset=${dataset} existing output=${output_json}"
    return 0
  fi

  write_status "03_reader_qa_${dataset}" "START dataset=${dataset} model=${READER_LLM_NAME}"
  log_msg "START ETV4+PCEC GPT-4o-mini reader dataset=${dataset} max_new_tokens=${MAX_NEW_TOKENS}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_reader_qa.py \
      --retrieval-reports "${retrieval_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${READER_QA_ROOT}/reader_runtime/${dataset}" \
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
    write_status "03_reader_qa_${dataset}" "DONE output=${output_json}"
    log_msg "DONE ETV4+PCEC GPT-4o-mini reader dataset=${dataset}"
  else
    write_status "03_reader_qa_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED ETV4+PCEC GPT-4o-mini reader dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

export_all_pools() {
  local status=0
  local dataset
  for dataset in ${DATASETS}; do
    export_pool_dataset "${dataset}" || status=1
  done
  return "${status}"
}

run_pcec_all() {
  local status=0
  local dataset
  for dataset in ${DATASETS}; do
    run_pcec_dataset "${dataset}" || status=1
  done
  return "${status}"
}

run_reader_qa_all() {
  local status=0
  local pids=()
  local dataset
  for dataset in ${DATASETS}; do
    run_reader_qa_dataset "${dataset}" &
    pids+=("$!")
  done
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  return "${status}"
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG} datasets=${DATASETS}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "ETV4 source: ${ETV4_ROOT}"
  log_msg "Protocol: ETV4 retrieval report -> pool100 export -> native PCEC -> GPT-4o-mini reader"
  log_msg "Reader: ${READER_LLM_NAME} at ${READER_LLM_BASE_URL}; max_new_tokens=${MAX_NEW_TOKENS}"

  local preflight=0
  check_endpoint "qwen32b" "${GRAPH_LLM_BASE_URL}/models" || preflight=1
  check_endpoint "nv_embed" "http://localhost:8019/v1/models" || preflight=1
  require_api_key || preflight=1
  if [[ "${preflight}" -ne 0 ]]; then
    write_status "launcher" "FAILED preflight"
    return 1
  fi

  export_all_pools || return 1
  run_pcec_all || return 1
  run_reader_qa_all || return 1

  write_status "launcher" "DONE run_tag=${RUN_TAG}"
  log_msg "DONE ${RUN_TAG}"
  return 0
}

main "$@"
