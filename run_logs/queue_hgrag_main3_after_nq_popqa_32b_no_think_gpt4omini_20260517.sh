#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/mnt/nvme/code/HippoRAG}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/run_logs/launch_hgrag_nq_popqa_32b_no_think_gpt4omini_full1000_20260517.sh}"

WAIT_ROOT="${WAIT_ROOT:-${ROOT_DIR}/run_logs/hgrag_nq_popqa_32b_no_think_gpt4omini_full1000_20260517}"
RUN_TAG="${RUN_TAG:-hgrag_main3_32b_no_think_gpt4omini_full1000_20260517}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/run_logs/${RUN_TAG}}"

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
QWEN32_URL_A_DEFAULT="${QWEN32_URL_A_DEFAULT:-http://localhost:8045/v1}"
QWEN32_URL_B_DEFAULT="${QWEN32_URL_B_DEFAULT:-http://localhost:8046/v1}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/status"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/logs/queue.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee "${OUT_ROOT}/status/${name}.status"
}

current_hgrag_done() {
  local nq_status="${WAIT_ROOT}/status/hgrag_retrieval_nq.status"
  local popqa_status="${WAIT_ROOT}/status/hgrag_retrieval_popqa.status"
  [[ -f "${nq_status}" && -f "${popqa_status}" ]] || return 1
  grep -Eq 'DONE hgrag retrieval dataset=nq|SKIP existing output=' "${nq_status}" || return 1
  grep -Eq 'DONE hgrag retrieval dataset=popqa|SKIP existing output=' "${popqa_status}" || return 1
}

current_hgrag_failed() {
  grep -q 'FAILED' "${WAIT_ROOT}/status/hgrag_retrieval_nq.status" 2>/dev/null && return 0
  grep -q 'FAILED' "${WAIT_ROOT}/status/hgrag_retrieval_popqa.status" 2>/dev/null && return 0
  return 1
}

endpoint_for_dataset() {
  case "$1" in
    hotpotqa) printf '%s\n' "${QWEN32_URL_B_DEFAULT}" ;;
    *) printf '%s\n' "${QWEN32_URL_A_DEFAULT}" ;;
  esac
}

run_dataset() {
  local dataset="$1"
  local endpoint
  endpoint="$(endpoint_for_dataset "${dataset}")"
  write_status "queue_${dataset}" "START endpoint=${endpoint}"
  log_msg "START HGRAG main3 dataset=${dataset} endpoint=${endpoint}"
  (
    cd "${ROOT_DIR}" || exit 1
    RUN_TAG="${RUN_TAG}" \
    OUT_ROOT="${OUT_ROOT}" \
    DATASETS="${dataset}" \
    QWEN32_URL_A="${endpoint}" \
    QWEN32_URL_B="${endpoint}" \
    HGRAG_MAX_WORKERS="${HGRAG_MAX_WORKERS:-4}" \
    HGRAG_EMBEDDING_BATCH_SIZE="${HGRAG_EMBEDDING_BATCH_SIZE:-16}" \
    "${BASE_LAUNCHER}"
  ) > "${OUT_ROOT}/logs/${dataset}.launcher.log" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    write_status "queue_${dataset}" "DONE"
    log_msg "DONE HGRAG main3 dataset=${dataset}"
  else
    write_status "queue_${dataset}" "FAILED code=${code} log=${OUT_ROOT}/logs/${dataset}.launcher.log"
    log_msg "FAILED HGRAG main3 dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

main() {
  write_status "queue_launcher" "START run_tag=${RUN_TAG} waiting_for=${WAIT_ROOT}"
  log_msg "BEGIN queued HGRAG main3 run. Waiting for NQ/PopQA HGRAG retrieval to finish before starting ${DATASETS}."

  while ! current_hgrag_done; do
    if current_hgrag_failed; then
      write_status "queue_launcher" "FAILED dependency_failed wait_root=${WAIT_ROOT}"
      log_msg "FAILED dependency HGRAG NQ/PopQA retrieval failed."
      return 1
    fi
    log_msg "WAIT current HGRAG NQ/PopQA retrieval still running"
    sleep "${WAIT_SECONDS}"
  done

  log_msg "Dependency complete. Starting HGRAG main3 sequential run."
  local rc=0
  local dataset
  for dataset in ${DATASETS}; do
    run_dataset "${dataset}" || {
      rc=1
      break
    }
  done

  if [[ "${rc}" -eq 0 ]]; then
    write_status "queue_launcher" "DONE run_tag=${RUN_TAG}"
    log_msg "DONE queued HGRAG main3 run"
  else
    write_status "queue_launcher" "FAILED run_tag=${RUN_TAG} rc=${rc}"
    log_msg "FAILED queued HGRAG main3 run rc=${rc}"
  fi
  return "${rc}"
}

main "$@"
