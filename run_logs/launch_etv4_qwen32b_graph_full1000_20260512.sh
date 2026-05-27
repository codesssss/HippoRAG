#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
METHOD="evidence_transition_graphragv4_fact_witnessed_sto"
RUN_TAG="etv4_clean_mainline_qwen32b_nothink_full1000_20260512"
OUT_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
SESSION="etv4_qwen32b_graph_full1000_20260512"

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
GRAPH_LLM_NAME="${GRAPH_LLM_NAME:-qwen3-32b-judge}"
GRAPH_LLM_BASE_URL="${GRAPH_LLM_BASE_URL:-http://localhost:8045/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
OPENIE_MAX_NEW_TOKENS="${OPENIE_MAX_NEW_TOKENS:-2048}"
MAX_RETRY_ATTEMPTS="${MAX_RETRY_ATTEMPTS:-20}"
CANDIDATE_POOL_K="${CANDIDATE_POOL_K:-200}"
DENSE_ROOT_COUNT="${DENSE_ROOT_COUNT:-20}"
TOP_K="${TOP_K:-5}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/status" "${OUT_ROOT}/jobs"

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

report_path() {
  local dataset="$1"
  printf '%s/%s/reports/%s_%s_retrieval.json' "${OUT_ROOT}" "${dataset}" "${dataset}" "${METHOD}"
}

preflight() {
  local rc=0
  check_endpoint "qwen32b" "${GRAPH_LLM_BASE_URL%/}/models" || rc=1
  check_endpoint "embedding" "${EMBEDDING_BASE_URL%/embeddings}/models" || rc=1
  return "${rc}"
}

write_driver() {
  local job_script="${OUT_ROOT}/jobs/${SESSION}.sh"
  cat > "${job_script}" <<'JOB'
#!/usr/bin/env bash
set -u

run_dataset() {
  local dataset="$1"
  local report="${OUT_ROOT}/${dataset}/reports/${dataset}_${METHOD}_retrieval.json"
  local log_path="${OUT_ROOT}/logs/etv4_qwen32b_${dataset}.log"
  local status_path="${OUT_ROOT}/status/${dataset}.status"

  if [[ -s "${report}" ]]; then
    printf 'done_existing\t%s\t%s\n' "$(date -Is)" "${report}" > "${status_path}"
    echo "[SKIP] dataset=${dataset} existing report=${report}"
    return 0
  fi

  printf 'running\t%s\t%s\n' "$(date -Is)" "${report}" > "${status_path}"
  {
    echo "[START] dataset=${dataset} start=$(date -Is)"
    echo "[OUTPUT_ROOT] ${OUT_ROOT}"
    echo "[REPORT] ${report}"
    echo "[GRAPH_LLM] ${GRAPH_LLM_NAME} ${GRAPH_LLM_BASE_URL}"
    echo "[EMBEDDING] ${EMBEDDING_NAME} ${EMBEDDING_BASE_URL}"
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export HIPPORAG_RERANK_FORCE_NO_THINK=1
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_fresh_e2e.py \
      --datasets "${dataset}" \
      --output-root "${OUT_ROOT}" \
      --max-queries "${MAX_QUERIES}" \
      --llm-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${GRAPH_LLM_BASE_URL}" \
      --max-new-tokens "${OPENIE_MAX_NEW_TOKENS}" \
      --max-retry-attempts "${MAX_RETRY_ATTEMPTS}" \
      --qwen-disable-thinking \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --readout-policy clean_mainline \
      --candidate-pool-k "${CANDIDATE_POOL_K}" \
      --dense-root-count "${DENSE_ROOT_COUNT}" \
      --top-k "${TOP_K}"
    rc=$?
    echo "[COMMAND_EXIT] dataset=${dataset} rc=${rc} end=$(date -Is)"
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

overall_rc=0
for dataset in ${DATASETS}; do
  run_dataset "${dataset}" || overall_rc=$?
  if [[ "${overall_rc}" -ne 0 ]]; then
    break
  fi
done

if [[ "${overall_rc}" -eq 0 ]]; then
  printf 'done\t%s\n' "$(date -Is)" > "${OUT_ROOT}/status/driver.status"
else
  printf 'failed\t%s\trc=%s\n' "$(date -Is)" "${overall_rc}" > "${OUT_ROOT}/status/driver.status"
fi
exit "${overall_rc}"
JOB

  sed -i \
    -e "s|\${OUT_ROOT}|${OUT_ROOT}|g" \
    -e "s|\${METHOD}|${METHOD}|g" \
    -e "s|\${ROOT_DIR}|${ROOT_DIR}|g" \
    -e "s|\${PYTHON_BIN}|${PYTHON_BIN}|g" \
    -e "s|\${DATASETS}|${DATASETS}|g" \
    -e "s|\${MAX_QUERIES}|${MAX_QUERIES}|g" \
    -e "s|\${GRAPH_LLM_NAME}|${GRAPH_LLM_NAME}|g" \
    -e "s|\${GRAPH_LLM_BASE_URL}|${GRAPH_LLM_BASE_URL}|g" \
    -e "s|\${EMBEDDING_NAME}|${EMBEDDING_NAME}|g" \
    -e "s|\${EMBEDDING_BASE_URL}|${EMBEDDING_BASE_URL}|g" \
    -e "s|\${EMBEDDING_BATCH_SIZE}|${EMBEDDING_BATCH_SIZE}|g" \
    -e "s|\${OPENIE_MAX_NEW_TOKENS}|${OPENIE_MAX_NEW_TOKENS}|g" \
    -e "s|\${MAX_RETRY_ATTEMPTS}|${MAX_RETRY_ATTEMPTS}|g" \
    -e "s|\${CANDIDATE_POOL_K}|${CANDIDATE_POOL_K}|g" \
    -e "s|\${DENSE_ROOT_COUNT}|${DENSE_ROOT_COUNT}|g" \
    -e "s|\${TOP_K}|${TOP_K}|g" \
    "${job_script}"
  chmod +x "${job_script}"
  printf '%s\n' "${job_script}"
}

main() {
  log_msg "START launcher run_tag=${RUN_TAG} datasets=${DATASETS}"
  if ! preflight; then
    log_msg "ABORT preflight failed"
    exit 1
  fi

  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    log_msg "SKIP tmux session already exists: ${SESSION}"
    exit 0
  fi

  local job_script
  job_script="$(write_driver)"
  printf 'launched\t%s\t%s\n' "$(date -Is)" "${SESSION}" > "${OUT_ROOT}/status/driver.status"
  tmux new-session -d -s "${SESSION}" "${job_script}"
  tmux set-option -t "${SESSION}" remain-on-exit on >/dev/null
  log_msg "LAUNCHED session=${SESSION} job=${job_script}"
  log_msg "Monitor: tail -f ${OUT_ROOT}/logs/etv4_qwen32b_2wikimultihopqa.log"
}

main "$@"
