#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"

RUN_TAG="evidencelink_simple_requirements_full1000_20260525"
OUT_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
BASE_RUN="${ROOT_DIR}/run_logs/all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514"
POOL_ROOT="${ROOT_DIR}/run_logs/etv4_pcec_qwen32b_gpt4omini_none_full1000_20260514/pools"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
REQUIREMENT_MODE="${REQUIREMENT_MODE:-simple_question_anchors}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
ARTIFACT_LIMIT="${ARTIFACT_LIMIT:-1000}"
POOL_K="${POOL_K:-100}"
READER_BUDGET_K="${READER_BUDGET_K:-5}"
PREFIX_BUDGET_M="${PREFIX_BUDGET_M:-4}"
RUN_READERS="${RUN_READERS:-0}"

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

case "${REQUIREMENT_MODE}" in
  whole_question|simple_question_anchors) ;;
  *)
    printf 'Unsupported REQUIREMENT_MODE=%s; expected whole_question or simple_question_anchors\n' "${REQUIREMENT_MODE}" >&2
    exit 2
    ;;
esac

mkdir -p \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/status" \
  "${OUT_ROOT}/requirements/${REQUIREMENT_MODE}/evals" \
  "${OUT_ROOT}/readout/${REQUIREMENT_MODE}/evals" \
  "${OUT_ROOT}/reader_qa/${REQUIREMENT_MODE}/reports" \
  "${OUT_ROOT}/reader_qa/${REQUIREMENT_MODE}/runtime"

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
  printf '%s/%s_etv4_pool%s_limit%s.json' "${POOL_ROOT}" "${dataset}" "${POOL_K}" "${ARTIFACT_LIMIT}"
}

simple_requirement_report_path() {
  local dataset="$1"
  printf '%s/requirements/%s/evals/%s_%s_pool%s_limit%s.json' \
    "${OUT_ROOT}" "${REQUIREMENT_MODE}" "${dataset}" "${REQUIREMENT_MODE}" "${POOL_K}" "${ARTIFACT_LIMIT}"
}

simple_binding_cache_path() {
  local dataset="$1"
  printf '%s/requirements/%s/evals/%s_%s_pool%s.binding_cache.json' \
    "${OUT_ROOT}" "${REQUIREMENT_MODE}" "${dataset}" "${REQUIREMENT_MODE}" "${POOL_K}"
}

retrieval_output_path() {
  local dataset="$1"
  printf '%s/readout/%s/evals/%s_%s_prefix%s_residual%s_pool%s_limit%s.json' \
    "${OUT_ROOT}" "${REQUIREMENT_MODE}" "${dataset}" "${REQUIREMENT_MODE}" "${PREFIX_BUDGET_M}" \
    "$((READER_BUDGET_K - PREFIX_BUDGET_M))" "${POOL_K}" "${MAX_QUERIES}"
}

reader_output_json() {
  local dataset="$1"
  printf '%s/reader_qa/%s/reports/%s_%s_gpt4omini_reader_qa_limit%s.json' \
    "${OUT_ROOT}" "${REQUIREMENT_MODE}" "${dataset}" "${REQUIREMENT_MODE}" "${MAX_QUERIES}"
}

reader_output_md() {
  local dataset="$1"
  printf '%s/reader_qa/%s/reports/%s_%s_gpt4omini_reader_qa_limit%s.md' \
    "${OUT_ROOT}" "${REQUIREMENT_MODE}" "${dataset}" "${REQUIREMENT_MODE}" "${MAX_QUERIES}"
}

make_simple_requirements_dataset() {
  local dataset="$1"
  local pool_json requirement_json binding_json log_path status_name
  pool_json="$(pool_json_path "${dataset}")"
  requirement_json="$(simple_requirement_report_path "${dataset}")"
  binding_json="$(simple_binding_cache_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/00_requirements_${REQUIREMENT_MODE}_${dataset}.log"
  status_name="00_requirements_${REQUIREMENT_MODE}_${dataset}"
  if [[ ! -s "${pool_json}" ]]; then
    write_status "${status_name}" "FAILED missing_pool=${pool_json}"
    log_msg "FAILED simple requirements dataset=${dataset} missing pool"
    return 2
  fi
  if [[ -s "${requirement_json}" && -s "${binding_json}" ]]; then
    write_status "${status_name}" "SKIP existing report=${requirement_json}"
    log_msg "SKIP simple requirements dataset=${dataset}"
    return 0
  fi
  write_status "${status_name}" "START pool=${pool_json}"
  log_msg "START simple requirements dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/make_simple_requirement_report.py \
      --dataset "${dataset}" \
      --mode "${REQUIREMENT_MODE}" \
      --pool-json "${pool_json}" \
      --output-json "${requirement_json}" \
      --binding-cache-json "${binding_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${requirement_json}" && -s "${binding_json}" ]]; then
    write_status "${status_name}" "DONE report=${requirement_json} binding=${binding_json}"
    log_msg "DONE simple requirements dataset=${dataset}"
  else
    write_status "${status_name}" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED simple requirements dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_readout_dataset() {
  local dataset="$1"
  local pool_json requirement_json binding_json output_json log_path status_name
  pool_json="$(pool_json_path "${dataset}")"
  requirement_json="$(simple_requirement_report_path "${dataset}")"
  binding_json="$(simple_binding_cache_path "${dataset}")"
  output_json="$(retrieval_output_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/01_readout_${REQUIREMENT_MODE}_${dataset}.log"
  status_name="01_readout_${REQUIREMENT_MODE}_${dataset}"
  if [[ ! -s "${pool_json}" || ! -s "${requirement_json}" || ! -s "${binding_json}" ]]; then
    write_status "${status_name}" "FAILED missing_input pool=${pool_json} report=${requirement_json} binding=${binding_json}"
    log_msg "FAILED readout dataset=${dataset} missing input"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "${status_name}" "SKIP existing output=${output_json}"
    log_msg "SKIP readout dataset=${dataset}"
    return 0
  fi
  write_status "${status_name}" "START output=${output_json}"
  log_msg "START readout dataset=${dataset}"
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
      --prefix-budget-m "${PREFIX_BUDGET_M}" \
      --pool-json "${pool_json}" \
      --requirement-report "${requirement_json}" \
      --binding-cache-path "${binding_json}" \
      --output-json "${output_json}" \
      --output-root "${OUT_ROOT}/readout/${REQUIREMENT_MODE}" \
      --data-root reproduce/dataset \
      --save-dir "${BASE_RUN}/pcec/etv4/runtime/${dataset}" \
      --llm-name "${GRAPH_LLM_NAME}" \
      --llm-request-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${GRAPH_LLM_BASE_URL}" \
      --llm-binding-model "${GRAPH_LLM_NAME}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --dtc-binding-max-candidates 0 \
      --qwen-disable-thinking
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "${status_name}" "DONE output=${output_json}"
    log_msg "DONE readout dataset=${dataset}"
  else
    write_status "${status_name}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED readout dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_reader_dataset() {
  local dataset="$1"
  local retrieval_json output_json output_md log_path status_name
  retrieval_json="$(retrieval_output_path "${dataset}")"
  output_json="$(reader_output_json "${dataset}")"
  output_md="$(reader_output_md "${dataset}")"
  log_path="${OUT_ROOT}/logs/02_reader_${REQUIREMENT_MODE}_${dataset}.log"
  status_name="02_reader_${REQUIREMENT_MODE}_${dataset}"
  if [[ ! -s "${retrieval_json}" ]]; then
    write_status "${status_name}" "FAILED missing_retrieval=${retrieval_json}"
    log_msg "FAILED reader dataset=${dataset} missing retrieval"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "${status_name}" "SKIP existing output=${output_json}"
    log_msg "SKIP reader dataset=${dataset}"
    return 0
  fi
  write_status "${status_name}" "START retrieval=${retrieval_json}"
  log_msg "START reader dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_reader_qa.py \
      --retrieval-reports "${retrieval_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${OUT_ROOT}/reader_qa/${REQUIREMENT_MODE}/runtime/${dataset}" \
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
    log_msg "DONE reader dataset=${dataset}"
  else
    write_status "${status_name}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED reader dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_parallel() {
  local stage="$1"
  local pids=()
  local dataset pid status=0
  for dataset in ${DATASETS}; do
    "${stage}" "${dataset}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  return "${status}"
}

main() {
  write_status "launcher_${REQUIREMENT_MODE}" "START run_tag=${RUN_TAG} mode=${REQUIREMENT_MODE} datasets=${DATASETS} run_readers=${RUN_READERS}"
  log_msg "BEGIN ${RUN_TAG} mode=${REQUIREMENT_MODE}"
  log_msg "Protocol: same ETv4 source-grounded pool and PCEC readout, ${REQUIREMENT_MODE} requirements, no dependency binding"

  local preflight=0
  check_endpoint "nv_embed" "http://localhost:8019/v1/models" || preflight=1
  if [[ "${RUN_READERS}" == "1" && -z "${OPENAI_API_KEY:-}" ]]; then
    log_msg "FAILED missing OPENAI_API_KEY; expected SECRET_ENV=${SECRET_ENV}"
    write_status "launcher_${REQUIREMENT_MODE}" "FAILED missing OPENAI_API_KEY"
    return 1
  fi
  if [[ "${preflight}" -ne 0 ]]; then
    write_status "launcher_${REQUIREMENT_MODE}" "FAILED preflight"
    return 1
  fi

  run_parallel make_simple_requirements_dataset || log_msg "WARN one or more simple requirement jobs failed"
  run_parallel run_readout_dataset || log_msg "WARN one or more readout jobs failed"
  if [[ "${RUN_READERS}" == "1" ]]; then
    run_parallel run_reader_dataset || log_msg "WARN one or more reader jobs failed"
  fi

  write_status "launcher_${REQUIREMENT_MODE}" "DONE run_tag=${RUN_TAG} mode=${REQUIREMENT_MODE}"
  log_msg "DONE ${RUN_TAG} mode=${REQUIREMENT_MODE}"
}

main "$@"
