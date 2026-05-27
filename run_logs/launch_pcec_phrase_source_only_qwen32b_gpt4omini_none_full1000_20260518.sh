#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"

RUN_TAG="pcec_phrase_source_only_qwen32b_gpt4omini_none_full1000_20260518"
ET32_ROOT="${ROOT_DIR}/run_logs/evidence_transition_graphragv3_variable_flow_qwen32b_nv2_full1000_20260511"
PCEC_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
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
GRAPH_LLM_NAME="${GRAPH_LLM_NAME:-qwen3-32b-judge}"
GRAPH_LLM_BASE_URL="${GRAPH_LLM_BASE_URL:-http://localhost:8045/v1}"
UTILITY_LLM_NAME="${UTILITY_LLM_NAME:-qwen3-8b}"
BINDING_MODEL="${BINDING_MODEL:-qwen3-8b-train}"
EMBEDDING_NAME="${EMBEDDING_NAME:-nvidia/NV-Embed-v2}"
PCEC_EMBEDDING_NAME="${PCEC_EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
ET_CANDIDATE_POOL_K="${ET_CANDIDATE_POOL_K:-200}"
PCEC_POOL_K="${PCEC_POOL_K:-100}"
READER_BUDGET_K="${READER_BUDGET_K:-5}"
PREFIX_BUDGET_M="${PREFIX_BUDGET_M:-4}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"

READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-none}"
QA_TOP_K="${QA_TOP_K:-5}"

mkdir -p \
  "${PCEC_ROOT}/logs" \
  "${PCEC_ROOT}/status" \
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

pcec_report_path() {
  local dataset="$1"
  printf '%s/evals/%s_pcec_fresh_e2e_prefix%s_residual%s_pool%s_limit%s.json' \
    "${PCEC_ROOT}" "${dataset}" "${PREFIX_BUDGET_M}" \
    "$((READER_BUDGET_K - PREFIX_BUDGET_M))" "${PCEC_POOL_K}" "${MAX_QUERIES}"
}

reader_output_json() {
  local dataset="$1"
  printf '%s/reports/%s_pcec_phrase_source_only_gpt4omini_none_reader_qa_full1000.json' \
    "${READER_QA_ROOT}" "${dataset}"
}

reader_output_md() {
  local dataset="$1"
  printf '%s/reports/%s_pcec_phrase_source_only_gpt4omini_none_reader_qa_full1000.md' \
    "${READER_QA_ROOT}" "${dataset}"
}

run_pcec_retrieval_dataset() {
  local dataset="$1"
  local log_path="${PCEC_ROOT}/logs/01_pcec_phrase_source_only_retrieval_${dataset}.log"
  local output_json
  output_json="$(pcec_report_path "${dataset}")"
  if [[ -s "${output_json}" ]]; then
    write_status "01_pcec_retrieval_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP PCEC phrase-source retrieval dataset=${dataset} existing output=${output_json}"
    return 0
  fi

  write_status "01_pcec_retrieval_${dataset}" "START dataset=${dataset} frozen_runs_root=${ET32_ROOT}/runs"
  log_msg "START PCEC phrase-source-only retrieval dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" evidence_transition_graphragv4_composition/run_fresh_e2e.py \
      --dataset "${dataset}" \
      --max-queries "${MAX_QUERIES}" \
      --artifact-limit "${MAX_QUERIES}" \
      --candidate-pool-k "${PCEC_POOL_K}" \
      --et-candidate-pool-k "${ET_CANDIDATE_POOL_K}" \
      --reader-budget-k "${READER_BUDGET_K}" \
      --prefix-budget-m "${PREFIX_BUDGET_M}" \
      --frozen-runs-root "${ET32_ROOT}/runs" \
      --output-root "${PCEC_ROOT}" \
      --llm-name "${UTILITY_LLM_NAME}" \
      --llm-request-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${GRAPH_LLM_BASE_URL}" \
      --llm-binding-model "${BINDING_MODEL}" \
      --embedding-name "${PCEC_EMBEDDING_NAME}" \
      --expander-embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --role-graph-edge-policy phrase_source_only \
      --qwen-disable-thinking \
      --progress-every "${PROGRESS_EVERY}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "01_pcec_retrieval_${dataset}" "DONE output=${output_json}"
    log_msg "DONE PCEC phrase-source-only retrieval dataset=${dataset}"
  else
    write_status "01_pcec_retrieval_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED PCEC phrase-source-only retrieval dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_reader_qa_dataset() {
  local dataset="$1"
  local retrieval_json output_json output_md log_path
  retrieval_json="$(pcec_report_path "${dataset}")"
  output_json="$(reader_output_json "${dataset}")"
  output_md="$(reader_output_md "${dataset}")"
  log_path="${PCEC_ROOT}/logs/02_reader_qa_${dataset}.log"

  if [[ ! -s "${retrieval_json}" ]]; then
    write_status "02_reader_qa_${dataset}" "FAILED missing_retrieval=${retrieval_json}"
    log_msg "FAILED reader QA dataset=${dataset} missing retrieval=${retrieval_json}"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "02_reader_qa_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP reader QA dataset=${dataset} existing output=${output_json}"
    return 0
  fi

  write_status "02_reader_qa_${dataset}" "START dataset=${dataset} model=${READER_LLM_NAME}"
  log_msg "START phrase-source-only GPT-4o-mini reader dataset=${dataset} max_new_tokens=${MAX_NEW_TOKENS}"
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
      --embedding-name "${PCEC_EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${output_json}" \
      --output-md "${output_md}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "02_reader_qa_${dataset}" "DONE output=${output_json}"
    log_msg "DONE phrase-source-only GPT-4o-mini reader dataset=${dataset}"
  else
    write_status "02_reader_qa_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED phrase-source-only GPT-4o-mini reader dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_pcec_retrieval_all() {
  local status=0
  local dataset
  for dataset in ${DATASETS}; do
    run_pcec_retrieval_dataset "${dataset}" || status=1
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
  log_msg "Ablation: role_graph_edge_policy=phrase_source_only"
  log_msg "Definition: keep passage--phrase incidence/title-grounding edges; remove cross-fact transition edges"
  log_msg "Frozen ETv3 root: ${ET32_ROOT}/runs"
  log_msg "Reader: ${READER_LLM_NAME} at ${READER_LLM_BASE_URL}; max_new_tokens=${MAX_NEW_TOKENS}"

  local preflight=0
  check_endpoint "qwen32b" "${GRAPH_LLM_BASE_URL}/models" || preflight=1
  check_endpoint "nv_embed" "http://localhost:8019/v1/models" || preflight=1
  require_api_key || preflight=1
  if [[ "${preflight}" -ne 0 ]]; then
    write_status "launcher" "FAILED preflight"
    return 1
  fi

  run_pcec_retrieval_all || return 1
  run_reader_qa_all || return 1

  write_status "launcher" "DONE run_tag=${RUN_TAG}"
  log_msg "DONE ${RUN_TAG}"
  return 0
}

main "$@"
