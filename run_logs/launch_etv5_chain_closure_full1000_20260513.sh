#!/usr/bin/env bash
set -u

echo "DISABLED: ETv5 chain_closure was diagnostic-only and is frozen after negative results. Use ETv4 clean_mainline for paper runs." >&2
exit 2

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
METHOD="evidence_transition_graphragv4_fact_witnessed_sto"
RUN_TAG="etv5_chain_closure_qwen32b_full1000_20260513"
OUT_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
SESSION="etv5_chain_closure_full1000_20260513"
CLEAN_INDEX_ROOT="${ROOT_DIR}/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512"

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
  return "${rc}"
}

retrieval_report_path() {
  local dataset="$1"
  printf '%s/%s/reports/%s_%s_retrieval.json' "${OUT_ROOT}" "${dataset}" "${dataset}" "${METHOD}"
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
    export PYTHONDONTWRITEBYTECODE=1
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
      --readout-policy chain_closure \
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

write_driver() {
  local job_script="${OUT_ROOT}/jobs/${SESSION}.sh"
  cat > "${job_script}" <<'JOB'
#!/usr/bin/env bash
set -u

overall_rc=0
pids=()
for dataset in ${DATASETS}; do
  run_retrieval_dataset "${dataset}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}" || overall_rc=$?
done

if [[ "${overall_rc}" -eq 0 ]]; then
  printf 'done\t%s\n' "$(date -Is)" > "${OUT_ROOT}/status/driver.status"
else
  printf 'failed\t%s\trc=%s\n' "$(date -Is)" "${overall_rc}" > "${OUT_ROOT}/status/driver.status"
fi
exit "${overall_rc}"
JOB
  {
    declare -f retrieval_report_path
    declare -f run_retrieval_dataset
    printf 'ROOT_DIR=%q\n' "${ROOT_DIR}"
    printf 'PYTHON_BIN=%q\n' "${PYTHON_BIN}"
    printf 'METHOD=%q\n' "${METHOD}"
    printf 'OUT_ROOT=%q\n' "${OUT_ROOT}"
    printf 'CLEAN_INDEX_ROOT=%q\n' "${CLEAN_INDEX_ROOT}"
    printf 'DATASETS=%q\n' "${DATASETS}"
    printf 'MAX_QUERIES=%q\n' "${MAX_QUERIES}"
    printf 'GRAPH_LLM_NAME=%q\n' "${GRAPH_LLM_NAME}"
    printf 'GRAPH_LLM_BASE_URL=%q\n' "${GRAPH_LLM_BASE_URL}"
    printf 'EMBEDDING_NAME=%q\n' "${EMBEDDING_NAME}"
    printf 'EMBEDDING_BASE_URL=%q\n' "${EMBEDDING_BASE_URL}"
    printf 'EMBEDDING_BATCH_SIZE=%q\n' "${EMBEDDING_BATCH_SIZE}"
    printf 'OPENIE_MAX_NEW_TOKENS=%q\n' "${OPENIE_MAX_NEW_TOKENS}"
    printf 'MAX_RETRY_ATTEMPTS=%q\n' "${MAX_RETRY_ATTEMPTS}"
    printf 'CANDIDATE_POOL_K=%q\n' "${CANDIDATE_POOL_K}"
    printf 'DENSE_ROOT_COUNT=%q\n' "${DENSE_ROOT_COUNT}"
    printf 'TOP_K=%q\n' "${TOP_K}"
    sed -n '/^overall_rc=/,$p' "${job_script}"
  } > "${job_script}.tmp"
  mv "${job_script}.tmp" "${job_script}"
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
  log_msg "Monitor: tail -f ${OUT_ROOT}/logs/retrieval_musique.log"
}

main "$@"
