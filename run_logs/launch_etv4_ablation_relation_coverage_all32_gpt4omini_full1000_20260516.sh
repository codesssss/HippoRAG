#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"

RUN_TAG="etv4_ablation_relation_coverage_all32_gpt4omini_full1000_20260516"
OUT_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
BASE_RUN="${ROOT_DIR}/run_logs/all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514"
POOL_ROOT="${ROOT_DIR}/run_logs/etv4_pcec_qwen32b_gpt4omini_none_full1000_20260514/pools"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
POOL_K="${POOL_K:-100}"
READER_BUDGET_K="${READER_BUDGET_K:-5}"
PREFIX_BUDGET_M="${PREFIX_BUDGET_M:-4}"
NO_COVERAGE_PREFIX_BUDGET_M="${NO_COVERAGE_PREFIX_BUDGET_M:-5}"

GRAPH_LLM_NAME="${GRAPH_LLM_NAME:-qwen3-32b-judge}"
GRAPH_LLM_BASE_URL="${GRAPH_LLM_BASE_URL:-http://localhost:8045/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-none}"
QA_TOP_K="${QA_TOP_K:-5}"

if [[ -s "${SECRET_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRET_ENV}"
  set +a
fi

mkdir -p \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/status" \
  "${OUT_ROOT}/no_relation/evals" \
  "${OUT_ROOT}/no_coverage/evals" \
  "${OUT_ROOT}/reader_qa/no_relation/reports" \
  "${OUT_ROOT}/reader_qa/no_coverage/reports" \
  "${OUT_ROOT}/reader_qa/no_relation/runtime" \
  "${OUT_ROOT}/reader_qa/no_coverage/runtime"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/logs/launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${OUT_ROOT}/status/${name}.status"
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

pool_json_path() {
  local dataset="$1"
  printf '%s/%s_etv4_pool%s_limit%s.json' "${POOL_ROOT}" "${dataset}" "${POOL_K}" "${MAX_QUERIES}"
}

requirement_report_path() {
  local dataset="$1"
  printf '%s/dbec_assets/etv4/evals/%s_etv4_pool%s_dbec_qwen32b_stable_limit%s.json' \
    "${BASE_RUN}" "${dataset}" "${POOL_K}" "${MAX_QUERIES}"
}

binding_cache_path() {
  local dataset="$1"
  printf '%s/dbec_assets/etv4/evals/%s_etv4_pool%s_dbec_qwen32b.binding_cache.json' \
    "${BASE_RUN}" "${dataset}" "${POOL_K}"
}

retrieval_output_path() {
  local variant="$1"
  local dataset="$2"
  case "${variant}" in
    no_relation)
      printf '%s/no_relation/evals/%s_pcec_no_relation_prefix%s_residual%s_pool%s_limit%s.json' \
        "${OUT_ROOT}" "${dataset}" "${PREFIX_BUDGET_M}" \
        "$((READER_BUDGET_K - PREFIX_BUDGET_M))" "${POOL_K}" "${MAX_QUERIES}"
      ;;
    no_coverage)
      printf '%s/no_coverage/evals/%s_pcec_no_coverage_prefix%s_residual%s_pool%s_limit%s.json' \
        "${OUT_ROOT}" "${dataset}" "${NO_COVERAGE_PREFIX_BUDGET_M}" \
        "$((READER_BUDGET_K - NO_COVERAGE_PREFIX_BUDGET_M))" "${POOL_K}" "${MAX_QUERIES}"
      ;;
    *)
      return 2
      ;;
  esac
}

reader_output_json() {
  local variant="$1"
  local dataset="$2"
  printf '%s/reader_qa/%s/reports/%s_etv4_%s_gpt4omini_reader_qa_full1000.json' \
    "${OUT_ROOT}" "${variant}" "${dataset}" "${variant}"
}

reader_output_md() {
  local variant="$1"
  local dataset="$2"
  printf '%s/reader_qa/%s/reports/%s_etv4_%s_gpt4omini_reader_qa_full1000.md' \
    "${OUT_ROOT}" "${variant}" "${dataset}" "${variant}"
}

run_readout_dataset() {
  local variant="$1"
  local dataset="$2"
  local pool_json requirement_json binding_json output_json log_path prefix_budget binding_top_m status_name
  pool_json="$(pool_json_path "${dataset}")"
  requirement_json="$(requirement_report_path "${dataset}")"
  binding_json="$(binding_cache_path "${dataset}")"
  output_json="$(retrieval_output_path "${variant}" "${dataset}")"
  log_path="${OUT_ROOT}/logs/01_readout_${variant}_${dataset}.log"
  status_name="01_readout_${variant}_${dataset}"

  if [[ ! -s "${pool_json}" || ! -s "${requirement_json}" || ! -s "${binding_json}" ]]; then
    write_status "${status_name}" "FAILED missing_input pool=${pool_json} report=${requirement_json} binding=${binding_json}"
    log_msg "FAILED readout variant=${variant} dataset=${dataset} missing input"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "${status_name}" "SKIP existing output=${output_json}"
    log_msg "SKIP readout variant=${variant} dataset=${dataset}"
    return 0
  fi

  prefix_budget="${PREFIX_BUDGET_M}"
  binding_top_m="5"
  if [[ "${variant}" == "no_relation" ]]; then
    binding_top_m="0"
  elif [[ "${variant}" == "no_coverage" ]]; then
    prefix_budget="${NO_COVERAGE_PREFIX_BUDGET_M}"
  fi

  write_status "${status_name}" "START output=${output_json}"
  log_msg "START readout variant=${variant} dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    export HIPPORAG_RERANK_FORCE_NO_THINK=1
    "${PYTHON_BIN}" evidence_transition_graphragv4_composition/run_native_pool.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --max-queries "${MAX_QUERIES}" \
      --pool-k "${POOL_K}" \
      --reader-budget-k "${READER_BUDGET_K}" \
      --prefix-budget-m "${prefix_budget}" \
      --pool-json "${pool_json}" \
      --requirement-report "${requirement_json}" \
      --binding-cache-path "${binding_json}" \
      --output-json "${output_json}" \
      --output-root "${OUT_ROOT}/${variant}" \
      --data-root reproduce/dataset \
      --save-dir "${BASE_RUN}/pcec/etv4/runtime/${dataset}" \
      --llm-name "${GRAPH_LLM_NAME}" \
      --llm-request-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${GRAPH_LLM_BASE_URL}" \
      --llm-binding-model "${GRAPH_LLM_NAME}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --dtc-binding-max-candidates "${binding_top_m}" \
      --qwen-disable-thinking
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "${status_name}" "DONE output=${output_json}"
    log_msg "DONE readout variant=${variant} dataset=${dataset}"
  else
    write_status "${status_name}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED readout variant=${variant} dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_reader_dataset() {
  local variant="$1"
  local dataset="$2"
  local retrieval_json output_json output_md log_path status_name
  retrieval_json="$(retrieval_output_path "${variant}" "${dataset}")"
  output_json="$(reader_output_json "${variant}" "${dataset}")"
  output_md="$(reader_output_md "${variant}" "${dataset}")"
  log_path="${OUT_ROOT}/logs/02_reader_${variant}_${dataset}.log"
  status_name="02_reader_${variant}_${dataset}"

  if [[ ! -s "${retrieval_json}" ]]; then
    write_status "${status_name}" "FAILED missing_retrieval=${retrieval_json}"
    log_msg "FAILED reader variant=${variant} dataset=${dataset} missing retrieval"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "${status_name}" "SKIP existing output=${output_json}"
    log_msg "SKIP reader variant=${variant} dataset=${dataset}"
    return 0
  fi

  write_status "${status_name}" "START retrieval=${retrieval_json}"
  log_msg "START reader variant=${variant} dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_reader_qa.py \
      --retrieval-reports "${retrieval_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${OUT_ROOT}/reader_qa/${variant}/runtime/${dataset}" \
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
    log_msg "DONE reader variant=${variant} dataset=${dataset}"
  else
    write_status "${status_name}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED reader variant=${variant} dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_parallel_readouts() {
  local pids=()
  local variant dataset pid status=0
  for variant in no_relation no_coverage; do
    for dataset in ${DATASETS}; do
      run_readout_dataset "${variant}" "${dataset}" &
      pids+=("$!")
    done
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  return "${status}"
}

run_parallel_readers() {
  local pids=()
  local variant dataset pid status=0
  for variant in no_relation no_coverage; do
    for dataset in ${DATASETS}; do
      run_reader_dataset "${variant}" "${dataset}" &
      pids+=("$!")
    done
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  return "${status}"
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG} datasets=${DATASETS}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: Qwen3-32B no_think for non-reader; GPT-4o-mini reader; max top-5 reader docs"

  local preflight=0
  check_endpoint "qwen32b" "${GRAPH_LLM_BASE_URL}/models" || preflight=1
  check_endpoint "nv_embed" "http://localhost:8019/v1/models" || preflight=1
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    log_msg "FAILED missing OPENAI_API_KEY; expected SECRET_ENV=${SECRET_ENV}"
    write_status "launcher" "FAILED missing OPENAI_API_KEY"
    return 1
  fi
  if [[ "${preflight}" -ne 0 ]]; then
    write_status "launcher" "FAILED preflight"
    return 1
  fi

  run_parallel_readouts || log_msg "WARN one or more readout jobs failed"
  run_parallel_readers || log_msg "WARN one or more reader jobs failed"

  write_status "launcher" "DONE run_tag=${RUN_TAG}"
  log_msg "DONE ${RUN_TAG}"
}

main "$@"
