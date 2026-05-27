#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"

RUN_TAG="all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514"
OUT_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

ETV3_FROZEN_ROOT="${ROOT_DIR}/run_logs/evidence_transition_graphragv3_variable_flow_qwen32b_nv2_full1000_20260511"
ETV3_POOL_ROOT="${ROOT_DIR}/run_logs/pcec_qwen32b_graph_e2e_full1000_20260511/fresh_pools"
ETV4_POOL_ROOT="${ROOT_DIR}/run_logs/etv4_pcec_qwen32b_gpt4omini_none_full1000_20260514/pools"

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
POOL_K="${POOL_K:-100}"
SETWISE_POOL_K="${SETWISE_POOL_K:-100}"
READER_BUDGET_K="${READER_BUDGET_K:-5}"
PREFIX_BUDGET_M="${PREFIX_BUDGET_M:-4}"
ET_CANDIDATE_POOL_K="${ET_CANDIDATE_POOL_K:-200}"

GRAPH_LLM_NAME="${GRAPH_LLM_NAME:-qwen3-32b-judge}"
GRAPH_LLM_BASE_URL="${GRAPH_LLM_BASE_URL:-http://localhost:8045/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EXPANDER_EMBEDDING_NAME="${EXPANDER_EMBEDDING_NAME:-nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-none}"
QA_TOP_K="${QA_TOP_K:-5}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"

if [[ -s "${SECRET_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRET_ENV}"
  set +a
fi

mkdir -p \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/status" \
  "${OUT_ROOT}/dbec_assets/etv3/evals" \
  "${OUT_ROOT}/dbec_assets/etv4/evals" \
  "${OUT_ROOT}/pcec/etv3/evals" \
  "${OUT_ROOT}/pcec/etv4/evals" \
  "${OUT_ROOT}/pcec/etv3/runtime" \
  "${OUT_ROOT}/pcec/etv4/runtime" \
  "${OUT_ROOT}/reader_qa/etv3/reports" \
  "${OUT_ROOT}/reader_qa/etv4/reports" \
  "${OUT_ROOT}/reader_qa/etv3/runtime" \
  "${OUT_ROOT}/reader_qa/etv4/runtime"

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

require_api_key() {
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    write_status "launcher" "FAILED missing OPENAI_API_KEY"
    log_msg "FAILED missing OPENAI_API_KEY; expected SECRET_ENV=${SECRET_ENV}"
    return 1
  fi
}

pool_json_path() {
  local method="$1"
  local dataset="$2"
  case "${method}" in
    etv3)
      printf '%s/%s_fresh_frozen_etv3_pool%s_limit%s.json' \
        "${ETV3_POOL_ROOT}" "${dataset}" "${POOL_K}" "${MAX_QUERIES}"
      ;;
    etv4)
      printf '%s/%s_etv4_pool%s_limit%s.json' \
        "${ETV4_POOL_ROOT}" "${dataset}" "${POOL_K}" "${MAX_QUERIES}"
      ;;
    *)
      return 2
      ;;
  esac
}

dbec_report_path() {
  local method="$1"
  local dataset="$2"
  printf '%s/dbec_assets/%s/evals/%s_%s_pool%s_dbec_qwen32b_stable_limit%s.json' \
    "${OUT_ROOT}" "${method}" "${dataset}" "${method}" "${POOL_K}" "${MAX_QUERIES}"
}

dbec_binding_cache_path() {
  local method="$1"
  local dataset="$2"
  printf '%s/dbec_assets/%s/evals/%s_%s_pool%s_dbec_qwen32b.binding_cache.json' \
    "${OUT_ROOT}" "${method}" "${dataset}" "${method}" "${POOL_K}"
}

pcec_report_path() {
  local method="$1"
  local dataset="$2"
  local mode
  if [[ "${method}" == "etv3" ]]; then
    mode="pcec_fresh_e2e"
  else
    mode="pcec_native_pool"
  fi
  printf '%s/pcec/%s/evals/%s_%s_prefix%s_residual%s_pool%s_limit%s.json' \
    "${OUT_ROOT}" "${method}" "${dataset}" "${mode}" "${PREFIX_BUDGET_M}" \
    "$((READER_BUDGET_K - PREFIX_BUDGET_M))" "${POOL_K}" "${MAX_QUERIES}"
}

reader_output_json() {
  local method="$1"
  local dataset="$2"
  printf '%s/reader_qa/%s/reports/%s_%s_all32_pcec_gpt4omini_none_reader_qa_full1000.json' \
    "${OUT_ROOT}" "${method}" "${dataset}" "${method}"
}

reader_output_md() {
  local method="$1"
  local dataset="$2"
  printf '%s/reader_qa/%s/reports/%s_%s_all32_pcec_gpt4omini_none_reader_qa_full1000.md' \
    "${OUT_ROOT}" "${method}" "${dataset}" "${method}"
}

run_dbec_asset_dataset() {
  local method="$1"
  local dataset="$2"
  local pool_json output_json binding_json log_path
  pool_json="$(pool_json_path "${method}" "${dataset}")"
  output_json="$(dbec_report_path "${method}" "${dataset}")"
  binding_json="$(dbec_binding_cache_path "${method}" "${dataset}")"
  log_path="${OUT_ROOT}/logs/01_dbec32_${method}_${dataset}.log"

  if [[ ! -s "${pool_json}" ]]; then
    write_status "01_dbec32_${method}_${dataset}" "FAILED missing_pool=${pool_json}"
    log_msg "FAILED DBEC32 asset method=${method} dataset=${dataset} missing pool=${pool_json}"
    return 2
  fi
  if [[ -s "${output_json}" && -s "${binding_json}" ]]; then
    write_status "01_dbec32_${method}_${dataset}" "SKIP existing report=${output_json} binding=${binding_json}"
    log_msg "SKIP DBEC32 asset method=${method} dataset=${dataset}"
    return 0
  fi

  write_status "01_dbec32_${method}_${dataset}" "START pool=${pool_json}"
  log_msg "START DBEC32 requirement/binding asset method=${method} dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    export HIPPORAG_RERANK_FORCE_NO_THINK=1
    "${PYTHON_BIN}" evidence_transition_graphragv3_dbec_latest/run_eval.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --pool-k "${POOL_K}" \
      --setwise-pool-k "${SETWISE_POOL_K}" \
      --qa-top-k "${QA_TOP_K}" \
      --pool-json "${pool_json}" \
      --output-root "${OUT_ROOT}/dbec_assets/${method}" \
      --output-json "${output_json}" \
      --binding-cache-path "${binding_json}" \
      --save-dir outputs_step0_general_nvembed \
      --llm-name "${GRAPH_LLM_NAME}" \
      --llm-request-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${GRAPH_LLM_BASE_URL}" \
      --llm-binding-url "${GRAPH_LLM_BASE_URL}" \
      --llm-binding-model "${GRAPH_LLM_NAME}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --qwen-disable-thinking
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" && -s "${binding_json}" ]]; then
    write_status "01_dbec32_${method}_${dataset}" "DONE report=${output_json} binding=${binding_json}"
    log_msg "DONE DBEC32 requirement/binding asset method=${method} dataset=${dataset}"
  else
    write_status "01_dbec32_${method}_${dataset}" "FAILED code=${code} report=${output_json} binding=${binding_json} log=${log_path}"
    log_msg "FAILED DBEC32 requirement/binding asset method=${method} dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_dbec_assets_all() {
  local method dataset
  for method in etv3 etv4; do
    for dataset in ${DATASETS}; do
      run_dbec_asset_dataset "${method}" "${dataset}" || return 1
    done
  done
}

run_pcec_etv3_dataset() {
  local dataset="$1"
  local requirement_json binding_json output_json log_path
  requirement_json="$(dbec_report_path etv3 "${dataset}")"
  binding_json="$(dbec_binding_cache_path etv3 "${dataset}")"
  output_json="$(pcec_report_path etv3 "${dataset}")"
  log_path="${OUT_ROOT}/logs/02_pcec32_etv3_${dataset}.log"

  if [[ ! -s "${requirement_json}" || ! -s "${binding_json}" ]]; then
    write_status "02_pcec32_etv3_${dataset}" "FAILED missing_dbec_asset report=${requirement_json} binding=${binding_json}"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "02_pcec32_etv3_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP all32 ETv3+PCEC dataset=${dataset}"
    return 0
  fi

  write_status "02_pcec32_etv3_${dataset}" "START report=${requirement_json}"
  log_msg "START all32 ETv3+PCEC dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" evidence_transition_graphragv4_composition/run_fresh_e2e.py \
      --dataset "${dataset}" \
      --max-queries "${MAX_QUERIES}" \
      --artifact-limit "${MAX_QUERIES}" \
      --candidate-pool-k "${POOL_K}" \
      --et-candidate-pool-k "${ET_CANDIDATE_POOL_K}" \
      --reader-budget-k "${READER_BUDGET_K}" \
      --prefix-budget-m "${PREFIX_BUDGET_M}" \
      --frozen-runs-root "${ETV3_FROZEN_ROOT}/runs" \
      --requirement-report-path "${requirement_json}" \
      --binding-cache-path "${binding_json}" \
      --output-json "${output_json}" \
      --output-root "${OUT_ROOT}/pcec/etv3" \
      --llm-name "${GRAPH_LLM_NAME}" \
      --llm-request-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${GRAPH_LLM_BASE_URL}" \
      --llm-binding-model "${GRAPH_LLM_NAME}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --expander-embedding-name "${EXPANDER_EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --qwen-disable-thinking \
      --progress-every "${PROGRESS_EVERY}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "02_pcec32_etv3_${dataset}" "DONE output=${output_json}"
    log_msg "DONE all32 ETv3+PCEC dataset=${dataset}"
  else
    write_status "02_pcec32_etv3_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED all32 ETv3+PCEC dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_pcec_etv4_dataset() {
  local dataset="$1"
  local pool_json requirement_json binding_json output_json log_path
  pool_json="$(pool_json_path etv4 "${dataset}")"
  requirement_json="$(dbec_report_path etv4 "${dataset}")"
  binding_json="$(dbec_binding_cache_path etv4 "${dataset}")"
  output_json="$(pcec_report_path etv4 "${dataset}")"
  log_path="${OUT_ROOT}/logs/02_pcec32_etv4_${dataset}.log"

  if [[ ! -s "${pool_json}" || ! -s "${requirement_json}" || ! -s "${binding_json}" ]]; then
    write_status "02_pcec32_etv4_${dataset}" "FAILED missing_input pool=${pool_json} report=${requirement_json} binding=${binding_json}"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "02_pcec32_etv4_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP all32 ETV4+PCEC dataset=${dataset}"
    return 0
  fi

  write_status "02_pcec32_etv4_${dataset}" "START pool=${pool_json}"
  log_msg "START all32 ETV4+PCEC dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
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
      --output-root "${OUT_ROOT}/pcec/etv4" \
      --data-root reproduce/dataset \
      --save-dir "${OUT_ROOT}/pcec/etv4/runtime/${dataset}" \
      --llm-name "${GRAPH_LLM_NAME}" \
      --llm-request-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${GRAPH_LLM_BASE_URL}" \
      --llm-binding-model "${GRAPH_LLM_NAME}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --qwen-disable-thinking
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "02_pcec32_etv4_${dataset}" "DONE output=${output_json}"
    log_msg "DONE all32 ETV4+PCEC dataset=${dataset}"
  else
    write_status "02_pcec32_etv4_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED all32 ETV4+PCEC dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_pcec_all() {
  local dataset
  for dataset in ${DATASETS}; do
    run_pcec_etv3_dataset "${dataset}" || return 1
  done
  for dataset in ${DATASETS}; do
    run_pcec_etv4_dataset "${dataset}" || return 1
  done
}

run_reader_qa_dataset() {
  local method="$1"
  local dataset="$2"
  local retrieval_json output_json output_md log_path
  retrieval_json="$(pcec_report_path "${method}" "${dataset}")"
  output_json="$(reader_output_json "${method}" "${dataset}")"
  output_md="$(reader_output_md "${method}" "${dataset}")"
  log_path="${OUT_ROOT}/logs/03_reader_${method}_${dataset}.log"

  if [[ ! -s "${retrieval_json}" ]]; then
    write_status "03_reader_${method}_${dataset}" "FAILED missing_retrieval=${retrieval_json}"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "03_reader_${method}_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP reader method=${method} dataset=${dataset}"
    return 0
  fi

  write_status "03_reader_${method}_${dataset}" "START retrieval=${retrieval_json}"
  log_msg "START all32 ${method}+PCEC GPT-4o-mini reader dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_reader_qa.py \
      --retrieval-reports "${retrieval_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${OUT_ROOT}/reader_qa/${method}/runtime/${dataset}" \
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
    write_status "03_reader_${method}_${dataset}" "DONE output=${output_json}"
    log_msg "DONE all32 ${method}+PCEC GPT-4o-mini reader dataset=${dataset}"
  else
    write_status "03_reader_${method}_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED all32 ${method}+PCEC GPT-4o-mini reader dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_reader_method_parallel() {
  local method="$1"
  local status=0
  local pids=()
  local dataset pid
  for dataset in ${DATASETS}; do
    run_reader_qa_dataset "${method}" "${dataset}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  return "${status}"
}

run_readers_all() {
  run_reader_method_parallel etv3 || return 1
  run_reader_method_parallel etv4 || return 1
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG} datasets=${DATASETS}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: Qwen32B graph/OpenIE + Qwen32B PCEC requirement/binding + GPT-4o-mini reader"
  log_msg "DBEC/PCEC utility model: ${GRAPH_LLM_NAME} at ${GRAPH_LLM_BASE_URL}"
  log_msg "Reader: ${READER_LLM_NAME} at ${READER_LLM_BASE_URL}; max_new_tokens=${MAX_NEW_TOKENS}"

  local preflight=0
  check_endpoint "qwen32b" "${GRAPH_LLM_BASE_URL}/models" || preflight=1
  check_endpoint "nv_embed" "http://localhost:8019/v1/models" || preflight=1
  require_api_key || preflight=1
  if [[ "${preflight}" -ne 0 ]]; then
    write_status "launcher" "FAILED preflight"
    return 1
  fi

  run_dbec_assets_all || return 1
  run_pcec_all || return 1
  run_readers_all || return 1

  write_status "launcher" "DONE run_tag=${RUN_TAG}"
  log_msg "DONE ${RUN_TAG}"
  return 0
}

main "$@"
