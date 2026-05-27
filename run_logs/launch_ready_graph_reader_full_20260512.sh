#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"

RUN_TAG="ready_graph_reader_full_20260512"
RUN_DIR="${ROOT_DIR}/run_logs/${RUN_TAG}"

ETV4_GRAPH_DIR="${ROOT_DIR}/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512"
ETV4_READER_DIR="${ROOT_DIR}/run_logs/etv4_clean_mainline_qwen32b_reader_gpt4omini_20260512"

BASELINE_DIR="${ROOT_DIR}/run_logs/baseline_qwen32b_graph_top200_gpt4omini_full_20260512"

MAX_QUERIES="${MAX_QUERIES:-1000}"
QA_TOP_K="${QA_TOP_K:-5}"
POOL_K="${POOL_K:-200}"
READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

mkdir -p \
  "${RUN_DIR}/logs" \
  "${RUN_DIR}/status" \
  "${ETV4_READER_DIR}/logs" \
  "${ETV4_READER_DIR}/reports" \
  "${ETV4_READER_DIR}/reader_runtime" \
  "${BASELINE_DIR}/reader_qa/proprag" \
  "${BASELINE_DIR}/reader_runtime/proprag" \
  "${BASELINE_DIR}/reader_inputs/proprag"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${RUN_DIR}/logs/launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${RUN_DIR}/status/${name}.status"
}

require_api_key() {
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    log_msg "FAILED missing OPENAI_API_KEY for ${READER_LLM_NAME}"
    write_status "launcher" "FAILED missing OPENAI_API_KEY"
    return 1
  fi
}

run_etv4_reader() {
  local dataset="$1"
  local retrieval_json="${ETV4_GRAPH_DIR}/${dataset}/reports/${dataset}_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json"
  local output_json="${ETV4_READER_DIR}/reports/${dataset}_etv4_qwen32b_gpt4omini_reader_qa.json"
  local output_md="${ETV4_READER_DIR}/reports/${dataset}_etv4_qwen32b_gpt4omini_reader_qa.md"
  local log_path="${ETV4_READER_DIR}/logs/${dataset}.log"

  if [[ ! -s "${retrieval_json}" ]]; then
    write_status "etv4_${dataset}" "SKIP missing retrieval=${retrieval_json}"
    log_msg "SKIP ETV4 reader dataset=${dataset}; missing retrieval"
    return 0
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "etv4_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP ETV4 reader dataset=${dataset}; existing output"
    return 0
  fi

  write_status "etv4_${dataset}" "START retrieval=${retrieval_json} output=${output_json}"
  log_msg "START ETV4 reader dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_reader_qa.py \
      --retrieval-reports "${retrieval_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${ETV4_READER_DIR}/reader_runtime/${dataset}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens none \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${output_json}" \
      --output-md "${output_md}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "etv4_${dataset}" "DONE output=${output_json}"
    log_msg "DONE ETV4 reader dataset=${dataset}"
  else
    write_status "etv4_${dataset}" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED ETV4 reader dataset=${dataset} code=${code}"
    return "${code}"
  fi
}

wrap_proprag_reader_input() {
  local dataset="$1"
  local pool_json="${BASELINE_DIR}/pools/proprag/${dataset}_proprag_qwen32b_nothink_pool${POOL_K}.json"
  local input_json="${BASELINE_DIR}/reader_inputs/proprag/${dataset}_proprag_qwen32b_nothink_pool${POOL_K}_reader_input.json"
  local log_path="${RUN_DIR}/logs/wrap_proprag_${dataset}.log"

  if [[ ! -s "${pool_json}" ]]; then
    write_status "wrap_proprag_${dataset}" "SKIP missing pool=${pool_json}"
    return 0
  fi
  if [[ -s "${input_json}" ]]; then
    write_status "wrap_proprag_${dataset}" "SKIP existing input=${input_json}"
    return 0
  fi

  write_status "wrap_proprag_${dataset}" "START pool=${pool_json}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" scripts/make_external_pool_reader_inputs.py \
      --pool-json "${pool_json}" \
      --method-name "proprag_qwen32b_top${POOL_K}" \
      --output-json "${input_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${input_json}" ]]; then
    write_status "wrap_proprag_${dataset}" "DONE input=${input_json}"
  else
    write_status "wrap_proprag_${dataset}" "FAILED code=${code} log=${log_path}"
    return "${code}"
  fi
}

run_proprag_reader() {
  local dataset="$1"
  local input_json="${BASELINE_DIR}/reader_inputs/proprag/${dataset}_proprag_qwen32b_nothink_pool${POOL_K}_reader_input.json"
  local output_json="${BASELINE_DIR}/reader_qa/proprag/${dataset}_proprag_qwen32b_nothink_top${POOL_K}_gpt4omini_reader_top${QA_TOP_K}.json"
  local output_md="${BASELINE_DIR}/reader_qa/proprag/${dataset}_proprag_qwen32b_nothink_top${POOL_K}_gpt4omini_reader_top${QA_TOP_K}.md"
  local log_path="${BASELINE_DIR}/logs/reader_proprag_${dataset}.log"

  wrap_proprag_reader_input "${dataset}" || return 1
  if [[ ! -s "${input_json}" ]]; then
    write_status "proprag_${dataset}" "SKIP missing input=${input_json}"
    log_msg "SKIP PropRAG reader dataset=${dataset}; missing input"
    return 0
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "proprag_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP PropRAG reader dataset=${dataset}; existing output"
    return 0
  fi

  write_status "proprag_${dataset}" "START input=${input_json} output=${output_json}"
  log_msg "START PropRAG reader dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py \
      --input-files "${input_json}" \
      --method-name "proprag_qwen32b_top${POOL_K}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --doc-source external_pool_topk_docs \
      --save-dir "${BASELINE_DIR}/reader_runtime/proprag/${dataset}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens none \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${output_json}" \
      --output-md "${output_md}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "proprag_${dataset}" "DONE output=${output_json}"
    log_msg "DONE PropRAG reader dataset=${dataset}"
  else
    write_status "proprag_${dataset}" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED PropRAG reader dataset=${dataset} code=${code}"
    return "${code}"
  fi
}

main() {
  write_status "launcher" "START ready graph reader full"
  log_msg "BEGIN ready graph reader full"
  require_api_key || return 1

  local pids=()
  run_etv4_reader 2wikimultihopqa &
  pids+=("$!")
  run_proprag_reader 2wikimultihopqa &
  pids+=("$!")
  run_proprag_reader hotpotqa &
  pids+=("$!")

  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done

  if [[ "${rc}" -eq 0 ]]; then
    write_status "launcher" "DONE ready graph reader full"
    log_msg "DONE ready graph reader full"
  else
    write_status "launcher" "FAILED ready graph reader full"
    log_msg "FAILED ready graph reader full"
  fi
  return "${rc}"
}

main "$@"
