#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"

RUN_TAG="pcec_qwen32b_graph_e2e_full1000_20260511"
ET32_ROOT="${ROOT_DIR}/run_logs/evidence_transition_graphragv3_variable_flow_qwen32b_nv2_full1000_20260511"
PCEC_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
FRESH_POOL_ROOT="${PCEC_ROOT}/fresh_pools"
READER_SAFE_ROOT="${PCEC_ROOT}/retrieval_reports_reader_docids"
READER_QA_ROOT="${PCEC_ROOT}/reader_qa"

DATASETS="${DATASETS:-2wikimultihopqa,hotpotqa,musique}"
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
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"

mkdir -p \
  "${PCEC_ROOT}/logs" \
  "${PCEC_ROOT}/status" \
  "${FRESH_POOL_ROOT}" \
  "${READER_SAFE_ROOT}" \
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

dataset_list() {
  printf '%s\n' "${DATASETS}" | tr ',' ' '
}

pcec_report_path() {
  local dataset="$1"
  echo "${PCEC_ROOT}/evals/${dataset}_pcec_fresh_e2e_prefix${PREFIX_BUDGET_M}_residual$((READER_BUDGET_K - PREFIX_BUDGET_M))_pool${PCEC_POOL_K}_limit${MAX_QUERIES}.json"
}

reader_safe_report_path() {
  local dataset="$1"
  echo "${READER_SAFE_ROOT}/${dataset}_pcec_fresh_e2e_prefix${PREFIX_BUDGET_M}_residual$((READER_BUDGET_K - PREFIX_BUDGET_M))_pool${PCEC_POOL_K}_limit${MAX_QUERIES}.json"
}

run_frozen_etv3_32b_graph() {
  local log_path="${PCEC_ROOT}/logs/01_frozen_etv3_32b_graph.log"
  write_status "01_frozen_etv3_32b_graph" "START graph_llm=${GRAPH_LLM_NAME} datasets=${DATASETS} max_queries=${MAX_QUERIES}"
  log_msg "START 32B frozen ETv3 graph/index + transition expansion"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" \
      evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_fresh_e2e.py \
      --output-root "${ET32_ROOT}" \
      --datasets "${DATASETS}" \
      --max-queries "${MAX_QUERIES}" \
      --llm-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${GRAPH_LLM_BASE_URL}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --candidate-pool-k "${ET_CANDIDATE_POOL_K}" \
      --top-k "${READER_BUDGET_K}" \
      --enable-variable-flow-traversal \
      --qwen-disable-thinking \
      --overwrite-fresh-index
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    write_status "01_frozen_etv3_32b_graph" "DONE output_root=${ET32_ROOT}"
    log_msg "DONE 32B frozen ETv3 graph/index + transition expansion"
  else
    write_status "01_frozen_etv3_32b_graph" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED 32B frozen ETv3 graph/index + transition expansion code=${code}"
  fi
  return "${code}"
}

export_fresh_pools() {
  local log_path="${PCEC_ROOT}/logs/02_export_32b_fresh_pools.log"
  write_status "02_export_32b_fresh_pools" "START frozen_runs_root=${ET32_ROOT}/runs"
  log_msg "START exporting 32B fresh pools for reader-safe doc-id repair"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" \
      scripts/verify_pcec_fresh_pool_parity.py \
      --datasets "${DATASETS}" \
      --limit "${MAX_QUERIES}" \
      --max-queries "${MAX_QUERIES}" \
      --pool-k "${PCEC_POOL_K}" \
      --reader-budget-k "${READER_BUDGET_K}" \
      --et-candidate-pool-k "${ET_CANDIDATE_POOL_K}" \
      --frozen-runs-root "${ET32_ROOT}/runs" \
      --fresh-pool-output-root "${FRESH_POOL_ROOT}" \
      --llm-name "${GRAPH_LLM_NAME}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --output-json "${PCEC_ROOT}/32b_fresh_pool_export_vs_8b_historical.json" \
      --output-md "${PCEC_ROOT}/32b_fresh_pool_export_vs_8b_historical.md" \
      --jobs 1
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    write_status "02_export_32b_fresh_pools" "DONE output_root=${FRESH_POOL_ROOT}"
    log_msg "DONE exporting 32B fresh pools"
  else
    write_status "02_export_32b_fresh_pools" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED exporting 32B fresh pools code=${code}"
  fi
  return "${code}"
}

run_pcec_retrieval_dataset() {
  local dataset="$1"
  local log_path="${PCEC_ROOT}/logs/03_pcec_retrieval_${dataset}.log"
  local output_json
  output_json="$(pcec_report_path "${dataset}")"
  write_status "03_pcec_retrieval_${dataset}" "START dataset=${dataset} frozen_runs_root=${ET32_ROOT}/runs"
  log_msg "START PCEC retrieval dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" \
      evidence_transition_graphragv4_composition/run_fresh_e2e.py \
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
      --qwen-disable-thinking \
      --progress-every "${PROGRESS_EVERY}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]] && [[ -s "${output_json}" ]]; then
    write_status "03_pcec_retrieval_${dataset}" "DONE output=${output_json}"
    log_msg "DONE PCEC retrieval dataset=${dataset}"
  else
    write_status "03_pcec_retrieval_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED PCEC retrieval dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_pcec_retrieval_all() {
  local status=0
  for dataset in $(dataset_list); do
    run_pcec_retrieval_dataset "${dataset}" || status=1
  done
  return "${status}"
}

repair_reader_doc_ids() {
  local log_path="${PCEC_ROOT}/logs/04_repair_reader_doc_ids.log"
  write_status "04_repair_reader_doc_ids" "START report_root=${PCEC_ROOT}/evals fresh_pool_root=${FRESH_POOL_ROOT}"
  log_msg "START repairing PCEC reports to external corpus doc ids"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" \
      scripts/repair_pcec_reader_doc_ids_from_fresh_pool.py \
      --datasets "${DATASETS}" \
      --report-root "${PCEC_ROOT}/evals" \
      --fresh-pool-root "${FRESH_POOL_ROOT}" \
      --output-root "${READER_SAFE_ROOT}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    write_status "04_repair_reader_doc_ids" "DONE output_root=${READER_SAFE_ROOT}"
    log_msg "DONE repairing PCEC reports to external corpus doc ids"
  else
    write_status "04_repair_reader_doc_ids" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED repairing PCEC reports code=${code}"
  fi
  return "${code}"
}

run_reader_qa_dataset() {
  local dataset="$1"
  local retrieval_json
  retrieval_json="$(reader_safe_report_path "${dataset}")"
  local output_json="${READER_QA_ROOT}/reports/${dataset}_pcec_qwen32b_graph_e2e_reader_qa_full1000.json"
  local output_md="${READER_QA_ROOT}/reports/${dataset}_pcec_qwen32b_graph_e2e_reader_qa_full1000.md"
  local log_path="${PCEC_ROOT}/logs/05_reader_qa_${dataset}.log"

  if [[ ! -s "${retrieval_json}" ]]; then
    write_status "05_reader_qa_${dataset}" "FAILED missing_retrieval=${retrieval_json}"
    log_msg "FAILED reader QA dataset=${dataset} missing retrieval=${retrieval_json}"
    return 2
  fi

  write_status "05_reader_qa_${dataset}" "START dataset=${dataset} model=${GRAPH_LLM_NAME}"
  log_msg "START 32B reader QA dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" \
      evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_reader_qa.py \
      --retrieval-reports "${retrieval_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${READER_BUDGET_K}" \
      --save-dir "${READER_QA_ROOT}/reader_runtime/${dataset}" \
      --llm-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${GRAPH_LLM_BASE_URL}" \
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
    write_status "05_reader_qa_${dataset}" "DONE output=${output_json}"
    log_msg "DONE 32B reader QA dataset=${dataset}"
  else
    write_status "05_reader_qa_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED 32B reader QA dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_reader_qa_all() {
  local status=0
  run_reader_qa_dataset 2wikimultihopqa &
  local pid_2wiki=$!
  run_reader_qa_dataset hotpotqa &
  local pid_hotpot=$!
  run_reader_qa_dataset musique &
  local pid_musique=$!
  wait "${pid_2wiki}" || status=1
  wait "${pid_hotpot}" || status=1
  wait "${pid_musique}" || status=1
  return "${status}"
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "32B scope: fresh ETv3 OpenIE/STO graph + transition expansion + final reader QA"
  log_msg "Readout scope: PCEC native readout with frozen DBEC requirements and ${BINDING_MODEL} binding cache"
  log_msg "Outputs: ${PCEC_ROOT}"

  local preflight=0
  check_endpoint "qwen32b" "${GRAPH_LLM_BASE_URL}/models" || preflight=1
  check_endpoint "nv_embed" "http://localhost:8019/v1/models" || preflight=1
  if [[ "${preflight}" -ne 0 ]]; then
    write_status "launcher" "FAILED preflight"
    return 1
  fi

  run_frozen_etv3_32b_graph || return 1
  export_fresh_pools || return 1
  run_pcec_retrieval_all || return 1
  repair_reader_doc_ids || return 1
  run_reader_qa_all || return 1

  write_status "launcher" "DONE run_tag=${RUN_TAG}"
  log_msg "DONE ${RUN_TAG}"
  return 0
}

main "$@"
