#!/usr/bin/env bash
set -u

echo "DISABLED: ETv5 chain_closure fix2c was diagnostic-only and is frozen after negative results. Use ETv4 clean_mainline for paper runs." >&2
exit 2

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
RUN_TAG="etv5_chain_closure_fix2c_musique_full1000_20260513"
OUT_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
SESSION="${RUN_TAG}"
CLEAN_INDEX_ROOT="${ROOT_DIR}/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512"
DATASET="musique"

mkdir -p "${OUT_ROOT}/${DATASET}" "${OUT_ROOT}/logs" "${OUT_ROOT}/status"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/logs/launcher.log"
}

driver() {
  local index_src="${CLEAN_INDEX_ROOT}/${DATASET}/index"
  local index_dst="${OUT_ROOT}/${DATASET}/index"
  local report="${OUT_ROOT}/${DATASET}/reports/${DATASET}_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json"
  local log_path="${OUT_ROOT}/logs/retrieval_${DATASET}.log"
  local status_path="${OUT_ROOT}/status/retrieval_${DATASET}.status"

  if [[ ! -d "${index_src}" ]]; then
    log_msg "FAILED missing index ${index_src}"
    printf 'failed_preflight\t%s\tmissing_index=%s\n' "$(date -Is)" "${index_src}" > "${status_path}"
    exit 1
  fi
  ln -sfn "${index_src}" "${index_dst}"

  if [[ -s "${report}" ]]; then
    log_msg "SKIP existing report ${report}"
    printf 'done_existing\t%s\t%s\n' "$(date -Is)" "${report}" > "${status_path}"
    exit 0
  fi

  printf 'running\t%s\t%s\n' "$(date -Is)" "${report}" > "${status_path}"
  {
    echo "[START] dataset=${DATASET} start=$(date -Is)"
    echo "[OUTPUT_ROOT] ${OUT_ROOT}"
    echo "[REUSE_INDEX] ${index_dst} -> ${index_src}"
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export PYTHONDONTWRITEBYTECODE=1
    export HIPPORAG_RERANK_FORCE_NO_THINK=1
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_fresh_e2e.py \
      --datasets "${DATASET}" \
      --output-root "${OUT_ROOT}" \
      --reuse-current-fresh-index \
      --max-queries 1000 \
      --llm-name qwen3-32b-judge \
      --llm-base-url http://localhost:8045/v1 \
      --max-new-tokens 2048 \
      --max-retry-attempts 20 \
      --qwen-disable-thinking \
      --embedding-name VLLM/nvidia/NV-Embed-v2 \
      --embedding-base-url http://localhost:8019/v1/embeddings \
      --embedding-batch-size 16 \
      --readout-policy chain_closure \
      --candidate-pool-k 200 \
      --dense-root-count 20 \
      --top-k 5
    rc=$?
    echo "[COMMAND_EXIT] dataset=${DATASET} rc=${rc} end=$(date -Is)"
    exit "${rc}"
  } 2>&1 | tee "${log_path}"

  local rc=${PIPESTATUS[0]}
  if [[ "${rc}" -eq 0 && -s "${report}" ]]; then
    printf 'done\t%s\t%s\n' "$(date -Is)" "${report}" > "${status_path}"
    printf 'done\t%s\n' "$(date -Is)" > "${OUT_ROOT}/status/driver.status"
  else
    printf 'failed\t%s\trc=%s\treport_exists=%s\tlog=%s\n' \
      "$(date -Is)" "${rc}" "$(test -s "${report}" && echo yes || echo no)" "${log_path}" > "${status_path}"
    printf 'failed\t%s\n' "$(date -Is)" > "${OUT_ROOT}/status/driver.status"
  fi
  return "${rc}"
}

main() {
  if [[ "${1:-}" == "--driver" ]]; then
    driver
    return $?
  fi
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    log_msg "SKIP tmux session exists ${SESSION}"
    exit 0
  fi
  printf 'launched\t%s\t%s\n' "$(date -Is)" "${SESSION}" > "${OUT_ROOT}/status/driver.status"
  tmux new-session -d -s "${SESSION}" "/bin/bash ${BASH_SOURCE[0]} --driver"
  tmux set-option -t "${SESSION}" remain-on-exit on >/dev/null
  log_msg "LAUNCHED session=${SESSION}"
}

main "$@"
