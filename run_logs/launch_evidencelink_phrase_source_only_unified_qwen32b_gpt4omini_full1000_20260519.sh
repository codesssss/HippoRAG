#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"

RUN_TAG="evidencelink_phrase_source_only_unified_qwen32b_gpt4omini_full1000_20260519"
ET32_ROOT="${ROOT_DIR}/run_logs/evidence_transition_graphragv3_variable_flow_qwen32b_nv2_full1000_20260511"
OUT_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

if [[ -s "${SECRET_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRET_ENV}"
  set +a
fi

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
POOL_K="${POOL_K:-100}"
SETWISE_POOL_K="${SETWISE_POOL_K:-100}"
ET_CANDIDATE_POOL_K="${ET_CANDIDATE_POOL_K:-200}"
READER_BUDGET_K="${READER_BUDGET_K:-5}"
PREFIX_BUDGET_M="${PREFIX_BUDGET_M:-4}"
QA_TOP_K="${QA_TOP_K:-5}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"

GRAPH_LLM_NAME="${GRAPH_LLM_NAME:-qwen3-32b-judge}"
GRAPH_LLM_BASE_URL="${GRAPH_LLM_BASE_URL:-http://localhost:8045/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EXPANDER_EMBEDDING_NAME="${EXPANDER_EMBEDDING_NAME:-nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-none}"

mkdir -p \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/status" \
  "${OUT_ROOT}/pools" \
  "${OUT_ROOT}/dbec_assets/phrase_source/evals" \
  "${OUT_ROOT}/pcec/phrase_source/evals" \
  "${OUT_ROOT}/pcec/phrase_source/runtime" \
  "${OUT_ROOT}/reader_qa/phrase_source/reports" \
  "${OUT_ROOT}/reader_qa/phrase_source/runtime"

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
  local dataset="$1"
  printf '%s/pools/%s_phrase_source_only_pool%s_limit%s.json' \
    "${OUT_ROOT}" "${dataset}" "${POOL_K}" "${MAX_QUERIES}"
}

dbec_report_path() {
  local dataset="$1"
  printf '%s/dbec_assets/phrase_source/evals/%s_phrase_source_only_pool%s_dbec_qwen32b_stable_limit%s.json' \
    "${OUT_ROOT}" "${dataset}" "${POOL_K}" "${MAX_QUERIES}"
}

dbec_binding_cache_path() {
  local dataset="$1"
  printf '%s/dbec_assets/phrase_source/evals/%s_phrase_source_only_pool%s_dbec_qwen32b.binding_cache.json' \
    "${OUT_ROOT}" "${dataset}" "${POOL_K}"
}

pcec_report_path() {
  local dataset="$1"
  printf '%s/pcec/phrase_source/evals/%s_phrase_source_only_pcec_native_pool_prefix%s_residual%s_pool%s_limit%s.json' \
    "${OUT_ROOT}" "${dataset}" "${PREFIX_BUDGET_M}" \
    "$((READER_BUDGET_K - PREFIX_BUDGET_M))" "${POOL_K}" "${MAX_QUERIES}"
}

reader_output_json() {
  local dataset="$1"
  printf '%s/reader_qa/phrase_source/reports/%s_phrase_source_only_unified_pcec_gpt4omini_none_reader_qa_full1000.json' \
    "${OUT_ROOT}" "${dataset}"
}

reader_output_md() {
  local dataset="$1"
  printf '%s/reader_qa/phrase_source/reports/%s_phrase_source_only_unified_pcec_gpt4omini_none_reader_qa_full1000.md' \
    "${OUT_ROOT}" "${dataset}"
}

run_export_pool_dataset() {
  local dataset="$1"
  local output_json log_path
  output_json="$(pool_json_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/01_export_phrase_source_pool_${dataset}.log"

  if [[ -s "${output_json}" ]]; then
    write_status "01_export_pool_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP phrase-source-only pool export dataset=${dataset}"
    return 0
  fi

  write_status "01_export_pool_${dataset}" "START output=${output_json}"
  log_msg "START phrase-source-only pool export dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" scripts/export_evidencelink_fresh_pool.py \
      --dataset "${dataset}" \
      --max-queries "${MAX_QUERIES}" \
      --pool-k "${POOL_K}" \
      --et-candidate-pool-k "${ET_CANDIDATE_POOL_K}" \
      --reader-budget-k "${READER_BUDGET_K}" \
      --data-root reproduce/dataset \
      --frozen-runs-root "${ET32_ROOT}/runs" \
      --llm-name "${GRAPH_LLM_NAME}" \
      --embedding-name "${EXPANDER_EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --role-graph-edge-policy phrase_source_only \
      --output-json "${output_json}" \
      --progress-every "${PROGRESS_EVERY}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "01_export_pool_${dataset}" "DONE output=${output_json}"
    log_msg "DONE phrase-source-only pool export dataset=${dataset}"
  else
    write_status "01_export_pool_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED phrase-source-only pool export dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_dbec_asset_dataset() {
  local dataset="$1"
  local pool_json output_json binding_json log_path
  pool_json="$(pool_json_path "${dataset}")"
  output_json="$(dbec_report_path "${dataset}")"
  binding_json="$(dbec_binding_cache_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/02_dbec32_phrase_source_${dataset}.log"

  if [[ ! -s "${pool_json}" ]]; then
    write_status "02_dbec32_${dataset}" "FAILED missing_pool=${pool_json}"
    log_msg "FAILED DBEC32 phrase-source asset dataset=${dataset} missing pool=${pool_json}"
    return 2
  fi
  if [[ -s "${output_json}" && -s "${binding_json}" ]]; then
    write_status "02_dbec32_${dataset}" "SKIP existing report=${output_json} binding=${binding_json}"
    log_msg "SKIP DBEC32 phrase-source asset dataset=${dataset}"
    return 0
  fi

  write_status "02_dbec32_${dataset}" "START pool=${pool_json}"
  log_msg "START DBEC32 phrase-source requirement/binding asset dataset=${dataset}"
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
      --output-root "${OUT_ROOT}/dbec_assets/phrase_source" \
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
    write_status "02_dbec32_${dataset}" "DONE report=${output_json} binding=${binding_json}"
    log_msg "DONE DBEC32 phrase-source requirement/binding asset dataset=${dataset}"
  else
    write_status "02_dbec32_${dataset}" "FAILED code=${code} report=${output_json} binding=${binding_json} log=${log_path}"
    log_msg "FAILED DBEC32 phrase-source requirement/binding asset dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_pcec_dataset() {
  local dataset="$1"
  local pool_json requirement_json binding_json output_json log_path
  pool_json="$(pool_json_path "${dataset}")"
  requirement_json="$(dbec_report_path "${dataset}")"
  binding_json="$(dbec_binding_cache_path "${dataset}")"
  output_json="$(pcec_report_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/03_pcec32_phrase_source_${dataset}.log"

  if [[ ! -s "${pool_json}" || ! -s "${requirement_json}" || ! -s "${binding_json}" ]]; then
    write_status "03_pcec32_${dataset}" "FAILED missing_input pool=${pool_json} report=${requirement_json} binding=${binding_json}"
    log_msg "FAILED phrase-source native PCEC dataset=${dataset} missing input"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "03_pcec32_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP phrase-source native PCEC dataset=${dataset}"
    return 0
  fi

  write_status "03_pcec32_${dataset}" "START pool=${pool_json}"
  log_msg "START phrase-source native PCEC dataset=${dataset}"
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
      --output-root "${OUT_ROOT}/pcec/phrase_source" \
      --data-root reproduce/dataset \
      --save-dir "${OUT_ROOT}/pcec/phrase_source/runtime/${dataset}" \
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
    write_status "03_pcec32_${dataset}" "DONE output=${output_json}"
    log_msg "DONE phrase-source native PCEC dataset=${dataset}"
  else
    write_status "03_pcec32_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED phrase-source native PCEC dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_reader_qa_dataset() {
  local dataset="$1"
  local retrieval_json output_json output_md log_path
  retrieval_json="$(pcec_report_path "${dataset}")"
  output_json="$(reader_output_json "${dataset}")"
  output_md="$(reader_output_md "${dataset}")"
  log_path="${OUT_ROOT}/logs/04_reader_phrase_source_${dataset}.log"

  if [[ ! -s "${retrieval_json}" ]]; then
    write_status "04_reader_${dataset}" "FAILED missing_retrieval=${retrieval_json}"
    log_msg "FAILED phrase-source GPT-4o-mini reader dataset=${dataset} missing retrieval=${retrieval_json}"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "04_reader_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP phrase-source GPT-4o-mini reader dataset=${dataset}"
    return 0
  fi

  write_status "04_reader_${dataset}" "START retrieval=${retrieval_json}"
  log_msg "START phrase-source GPT-4o-mini reader dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_reader_qa.py \
      --retrieval-reports "${retrieval_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${OUT_ROOT}/reader_qa/phrase_source/runtime/${dataset}" \
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
    write_status "04_reader_${dataset}" "DONE output=${output_json}"
    log_msg "DONE phrase-source GPT-4o-mini reader dataset=${dataset}"
  else
    write_status "04_reader_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED phrase-source GPT-4o-mini reader dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_graph_pipeline_all() {
  local dataset
  for dataset in ${DATASETS}; do
    run_export_pool_dataset "${dataset}" || return 1
    run_dbec_asset_dataset "${dataset}" || return 1
    run_pcec_dataset "${dataset}" || return 1
  done
}

run_readers_parallel() {
  local status=0
  local pids=()
  local dataset pid
  for dataset in ${DATASETS}; do
    run_reader_qa_dataset "${dataset}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  return "${status}"
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG} datasets=${DATASETS}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: phrase_source_only graph ablation + Qwen32B no-think DBEC/PCEC + GPT-4o-mini reader"
  log_msg "Ablation definition: keep source_endpoint_incidence/title_role_grounding only; remove sentence_grounded_transition/role_bridge/same_subject/same_object"
  log_msg "Frozen ET root: ${ET32_ROOT}/runs"
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

  run_graph_pipeline_all || return 1
  run_readers_parallel || return 1

  write_status "launcher" "DONE run_tag=${RUN_TAG}"
  log_msg "DONE ${RUN_TAG}"
  return 0
}

main "$@"
