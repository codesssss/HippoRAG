#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
METHOD="evidence_transition_graphragv4_fact_witnessed_sto"
RUN_TAG="etv4_qwen32b_guard_repair_full1000_20260513"
OUT_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
SESSION="etv4_guard_repair_full1000_20260513"
CLEAN_INDEX_ROOT="${ROOT_DIR}/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512"

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
GRAPH_LLM_NAME="${GRAPH_LLM_NAME:-qwen3-32b-judge}"
GRAPH_LLM_BASE_URL="${GRAPH_LLM_BASE_URL:-http://localhost:8045/v1}"
READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
OPENIE_MAX_NEW_TOKENS="${OPENIE_MAX_NEW_TOKENS:-2048}"
MAX_RETRY_ATTEMPTS="${MAX_RETRY_ATTEMPTS:-20}"
CANDIDATE_POOL_K="${CANDIDATE_POOL_K:-200}"
DENSE_ROOT_COUNT="${DENSE_ROOT_COUNT:-20}"
TOP_K="${TOP_K:-5}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/status" "${OUT_ROOT}/reader_qa/reports" "${OUT_ROOT}/reader_qa/runtime"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/logs/launcher.log"
}

check_endpoint() {
  local name="$1"
  local url="$2"
  if curl -fsS --max-time 10 "${url}" >/dev/null; then
    log_msg "OK endpoint ${name}: ${url}"
    return 0
  fi
  log_msg "FAILED endpoint ${name}: ${url}"
  return 1
}

ensure_openai_key() {
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    return 0
  fi
  local session_name
  for session_name in baseline_qwen32b_top200_20260512 hipporag_valid_qwen32b_20260513_r2 etv4_qwen32b_graph_full1000_20260512; do
    if tmux has-session -t "${session_name}" 2>/dev/null; then
      local key_line
      key_line="$(tmux show-environment -t "${session_name}" OPENAI_API_KEY 2>/dev/null || true)"
      if [[ "${key_line}" == OPENAI_API_KEY=* ]]; then
        export "${key_line}"
        log_msg "OK reader API key imported from tmux session ${session_name}"
        return 0
      fi
    fi
  done
  log_msg "FAILED missing OPENAI_API_KEY for ${READER_LLM_NAME} reader"
  return 1
}

prepare_index_links() {
  local dataset
  for dataset in ${DATASETS}; do
    local src="${CLEAN_INDEX_ROOT}/${dataset}/index"
    local dst_parent="${OUT_ROOT}/${dataset}"
    local dst="${dst_parent}/index"
    if [[ ! -d "${src}" ]]; then
      log_msg "FAILED missing clean Qwen32B index: ${src}"
      return 1
    fi
    mkdir -p "${dst_parent}"
    ln -sfn "${src}" "${dst}"
  done
}

preflight() {
  local rc=0
  prepare_index_links || rc=1
  check_endpoint "embedding" "${EMBEDDING_BASE_URL%/embeddings}/models" || rc=1
  ensure_openai_key || rc=1
  return "${rc}"
}

retrieval_report_path() {
  local dataset="$1"
  printf '%s/%s/reports/%s_%s_retrieval.json' "${OUT_ROOT}" "${dataset}" "${dataset}" "${METHOD}"
}

reader_json_path() {
  local dataset="$1"
  printf '%s/reader_qa/reports/%s_%s_guard_repair_qa.json' "${OUT_ROOT}" "${dataset}" "${METHOD}"
}

run_retrieval_dataset() {
  local dataset="$1"
  local report
  report="$(retrieval_report_path "${dataset}")"
  local log_path="${OUT_ROOT}/logs/retrieval_${dataset}.log"
  local status_path="${OUT_ROOT}/status/retrieval_${dataset}.status"

  if [[ -s "${report}" ]]; then
    printf 'done_existing\t%s\t%s\n' "$(date -Is)" "${report}" > "${status_path}"
    echo "[SKIP] retrieval dataset=${dataset} existing report=${report}"
    return 0
  fi

  printf 'running\t%s\t%s\n' "$(date -Is)" "${report}" > "${status_path}"
  {
    echo "[START] retrieval dataset=${dataset} start=$(date -Is)"
    echo "[OUTPUT_ROOT] ${OUT_ROOT}"
    echo "[REUSE_INDEX] ${OUT_ROOT}/${dataset}/index -> ${CLEAN_INDEX_ROOT}/${dataset}/index"
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export HIPPORAG_RERANK_FORCE_NO_THINK=1
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_fresh_e2e.py \
      --datasets "${dataset}" \
      --output-root "${OUT_ROOT}" \
      --reuse-current-fresh-index \
      --max-queries "${MAX_QUERIES}" \
      --llm-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${GRAPH_LLM_BASE_URL}" \
      --max-new-tokens "${OPENIE_MAX_NEW_TOKENS}" \
      --max-retry-attempts "${MAX_RETRY_ATTEMPTS}" \
      --qwen-disable-thinking \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --readout-policy pipeline_guard_repair \
      --candidate-pool-k "${CANDIDATE_POOL_K}" \
      --dense-root-count "${DENSE_ROOT_COUNT}" \
      --top-k "${TOP_K}"
    rc=$?
    echo "[COMMAND_EXIT] retrieval dataset=${dataset} rc=${rc} end=$(date -Is)"
    exit "${rc}"
  } 2>&1 | tee "${log_path}"

  local rc=${PIPESTATUS[0]}
  if [[ "${rc}" -eq 0 && -s "${report}" ]]; then
    printf 'done\t%s\t%s\n' "$(date -Is)" "${report}" > "${status_path}"
  else
    printf 'failed\t%s\trc=%s\treport_exists=%s\tlog=%s\n' \
      "$(date -Is)" "${rc}" "$(test -s "${report}" && echo yes || echo no)" "${log_path}" > "${status_path}"
  fi
  return "${rc}"
}

run_reader_dataset() {
  local dataset="$1"
  local report
  report="$(retrieval_report_path "${dataset}")"
  local output_json
  output_json="$(reader_json_path "${dataset}")"
  local output_md="${output_json%.json}.md"
  local log_path="${OUT_ROOT}/logs/reader_${dataset}.log"
  local status_path="${OUT_ROOT}/status/reader_${dataset}.status"

  if [[ -s "${output_json}" ]]; then
    printf 'done_existing\t%s\t%s\n' "$(date -Is)" "${output_json}" > "${status_path}"
    echo "[SKIP] reader dataset=${dataset} existing output=${output_json}"
    return 0
  fi
  if [[ ! -s "${report}" ]]; then
    printf 'failed\t%s\tmissing_retrieval_report=%s\n' "$(date -Is)" "${report}" > "${status_path}"
    return 1
  fi

  printf 'running\t%s\t%s\n' "$(date -Is)" "${output_json}" > "${status_path}"
  {
    echo "[START] reader dataset=${dataset} start=$(date -Is)"
    echo "[RETRIEVAL_REPORT] ${report}"
    echo "[OUTPUT_JSON] ${output_json}"
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_reader_qa.py \
      --retrieval-reports "${report}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${TOP_K}" \
      --save-dir "${OUT_ROOT}/reader_qa/runtime/${dataset}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens none \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${output_json}" \
      --output-md "${output_md}"
    rc=$?
    echo "[COMMAND_EXIT] reader dataset=${dataset} rc=${rc} end=$(date -Is)"
    exit "${rc}"
  } 2>&1 | tee "${log_path}"

  local rc=${PIPESTATUS[0]}
  if [[ "${rc}" -eq 0 && -s "${output_json}" ]]; then
    printf 'done\t%s\t%s\n' "$(date -Is)" "${output_json}" > "${status_path}"
  else
    printf 'failed\t%s\trc=%s\toutput_exists=%s\tlog=%s\n' \
      "$(date -Is)" "${rc}" "$(test -s "${output_json}" && echo yes || echo no)" "${log_path}" > "${status_path}"
  fi
  return "${rc}"
}

run_parallel_stage() {
  local stage="$1"
  shift
  local pids=()
  local dataset
  for dataset in ${DATASETS}; do
    "$@" "${dataset}" &
    pids+=("$!")
  done

  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  if [[ "${rc}" -eq 0 ]]; then
    printf 'done\t%s\n' "$(date -Is)" > "${OUT_ROOT}/status/${stage}.status"
  else
    printf 'failed\t%s\n' "$(date -Is)" > "${OUT_ROOT}/status/${stage}.status"
  fi
  return "${rc}"
}

driver() {
  log_msg "START run_tag=${RUN_TAG} datasets=${DATASETS}"
  if ! preflight; then
    log_msg "ABORT preflight failed"
    printf 'failed_preflight\t%s\n' "$(date -Is)" > "${OUT_ROOT}/status/driver.status"
    exit 1
  fi

  printf 'running_retrieval\t%s\n' "$(date -Is)" > "${OUT_ROOT}/status/driver.status"
  run_parallel_stage retrieval run_retrieval_dataset || {
    log_msg "ABORT retrieval stage failed"
    printf 'failed_retrieval\t%s\n' "$(date -Is)" > "${OUT_ROOT}/status/driver.status"
    exit 1
  }

  printf 'running_reader\t%s\n' "$(date -Is)" > "${OUT_ROOT}/status/driver.status"
  run_parallel_stage reader run_reader_dataset || {
    log_msg "ABORT reader stage failed"
    printf 'failed_reader\t%s\n' "$(date -Is)" > "${OUT_ROOT}/status/driver.status"
    exit 1
  }

  printf 'done\t%s\n' "$(date -Is)" > "${OUT_ROOT}/status/driver.status"
  log_msg "DONE run_tag=${RUN_TAG}"
}

main() {
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    log_msg "SKIP tmux session already exists: ${SESSION}"
    exit 0
  fi
  tmux new-session -d -s "${SESSION}" "${BASH_SOURCE[0]} --driver"
  tmux set-option -t "${SESSION}" remain-on-exit on >/dev/null
  printf 'launched\t%s\t%s\n' "$(date -Is)" "${SESSION}" > "${OUT_ROOT}/status/driver.status"
  log_msg "LAUNCHED session=${SESSION}"
  log_msg "Monitor: tail -f ${OUT_ROOT}/logs/launcher.log"
}

if [[ "${1:-}" == "--driver" ]]; then
  driver
else
  main "$@"
fi
