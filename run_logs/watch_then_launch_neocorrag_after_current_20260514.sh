#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
RUN_TAG="watch_then_launch_neocorrag_after_current_20260514"
OUT_DIR="${ROOT_DIR}/run_logs/${RUN_TAG}"
POLL_SECONDS="${POLL_SECONDS:-180}"

ALL32_STATUS="${ROOT_DIR}/run_logs/all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514/status/launcher.status"
NOFACT_STATUS="${ROOT_DIR}/run_logs/pcec_no_fact_witness_qwen32b_gpt4omini_none_full1000_20260514/status/launcher.status"
NEOCOR_LAUNCHER="${ROOT_DIR}/run_logs/launch_neocorrag_aligned_k3_qwen32b_full1000_20260514.sh"

mkdir -p "${OUT_DIR}"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_DIR}/watcher.log"
}

write_status() {
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${OUT_DIR}/watcher.status"
}

state_of() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    printf 'MISSING'
    return 0
  fi
  if rg -q 'DONE' "${path}"; then
    printf 'DONE'
    return 0
  fi
  if rg -q 'FAILED' "${path}"; then
    printf 'FAILED'
    return 0
  fi
  printf 'RUNNING'
}

main() {
  write_status "START waiting all32=${ALL32_STATUS} nofact=${NOFACT_STATUS}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Waiting for both current jobs to finish before launching NeocorRAG."
  log_msg "all32_status=${ALL32_STATUS}"
  log_msg "nofact_status=${NOFACT_STATUS}"
  log_msg "neocor_launcher=${NEOCOR_LAUNCHER}"

  while true; do
    local all32_state nofact_state
    all32_state="$(state_of "${ALL32_STATUS}")"
    nofact_state="$(state_of "${NOFACT_STATUS}")"
    write_status "WAIT all32=${all32_state} nofact=${nofact_state}"
    log_msg "WAIT all32=${all32_state} nofact=${nofact_state}"

    if [[ "${all32_state}" == "FAILED" || "${nofact_state}" == "FAILED" ]]; then
      write_status "FAILED upstream all32=${all32_state} nofact=${nofact_state}; not launching NeocorRAG"
      log_msg "FAILED upstream all32=${all32_state} nofact=${nofact_state}; not launching NeocorRAG"
      return 1
    fi

    if [[ "${all32_state}" == "DONE" && "${nofact_state}" == "DONE" ]]; then
      break
    fi

    sleep "${POLL_SECONDS}"
  done

  if [[ ! -s "${NEOCOR_LAUNCHER}" ]]; then
    write_status "FAILED missing launcher=${NEOCOR_LAUNCHER}"
    log_msg "FAILED missing launcher=${NEOCOR_LAUNCHER}"
    return 2
  fi

  write_status "STARTING_NEOCOR launcher=${NEOCOR_LAUNCHER}"
  log_msg "STARTING NeocorRAG via ${NEOCOR_LAUNCHER}"
  /bin/bash "${NEOCOR_LAUNCHER}" > "${OUT_DIR}/neocor_launcher.stdout.log" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    write_status "DONE launched_and_completed_neocor code=0"
    log_msg "DONE NeocorRAG completed"
  else
    write_status "FAILED neocor code=${code}"
    log_msg "FAILED NeocorRAG code=${code}"
  fi
  return "${code}"
}

main "$@"
