#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
RUN_ROOT="${ROOT_DIR}/reports/pcec_m4_position_sensitivity_8b_20260511"
RETRIEVAL_ROOT="${RUN_ROOT}/retrieval_reports"
QA_ROOT="${RUN_ROOT}/reader_qa"
DATASETS="${DATASETS:-2wikimultihopqa,hotpotqa,musique}"
SLOTS="${SLOTS:-1,2,3,4}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
QA_TOP_K="${QA_TOP_K:-5}"
LLM_NAME="${LLM_NAME:-qwen3-8b-train}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
EMBEDDING_NAME="${EMBEDDING_NAME:-nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

mkdir -p "${QA_ROOT}/logs" "${QA_ROOT}/status" "${QA_ROOT}/reports" "${QA_ROOT}/reader_runtime"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${QA_ROOT}/logs/launcher.log"
}

dataset_port() {
  case "$1" in
    musique) echo "8041" ;;
    hotpotqa) echo "8042" ;;
    2wikimultihopqa) echo "8043" ;;
    *) echo "8043" ;;
  esac
}

csv_to_words() {
  printf '%s\n' "$1" | tr ',' ' '
}

retrieval_report() {
  local dataset="$1"
  local slot="$2"
  echo "${RETRIEVAL_ROOT}/slot${slot}/${dataset}_pcec_m4_admitted_slot${slot}_pool100_limit1000.json"
}

run_dataset_slot() {
  local dataset="$1"
  local slot="$2"
  local port
  port="$(dataset_port "${dataset}")"
  local base_url="http://localhost:${port}/v1"
  local retrieval_json
  retrieval_json="$(retrieval_report "${dataset}" "${slot}")"
  local output_json="${QA_ROOT}/reports/${dataset}_pcec_m4_admitted_slot${slot}_reader_qa_8b_full1000.json"
  local output_md="${QA_ROOT}/reports/${dataset}_pcec_m4_admitted_slot${slot}_reader_qa_8b_full1000.md"
  local status_path="${QA_ROOT}/status/${dataset}_slot${slot}.status"
  local log_path="${QA_ROOT}/logs/${dataset}_slot${slot}.log"

  if [[ ! -s "${retrieval_json}" ]]; then
    echo "[FAILED] dataset=${dataset} slot=${slot} missing_retrieval=${retrieval_json} time=$(date -Is)" > "${status_path}"
    log_msg "FAILED dataset=${dataset} slot=${slot} missing retrieval=${retrieval_json}"
    return 2
  fi

  if ! curl -fsS "${base_url}/models" >/dev/null 2>&1; then
    echo "[FAILED] dataset=${dataset} slot=${slot} endpoint=${base_url} time=$(date -Is)" > "${status_path}"
    log_msg "FAILED dataset=${dataset} slot=${slot} endpoint unavailable: ${base_url}"
    return 3
  fi

  echo "[START] dataset=${dataset} slot=${slot} model=${LLM_NAME} endpoint=${base_url} time=$(date -Is)" > "${status_path}"
  log_msg "START reader QA dataset=${dataset} slot=${slot} endpoint=${base_url}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" \
      evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_reader_qa.py \
      --retrieval-reports "${retrieval_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${QA_ROOT}/reader_runtime/${dataset}_slot${slot}" \
      --llm-name "${LLM_NAME}" \
      --llm-base-url "${base_url}" \
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
    echo "[DONE] dataset=${dataset} slot=${slot} output=${output_json} time=$(date -Is)" > "${status_path}"
    log_msg "DONE reader QA dataset=${dataset} slot=${slot}"
  else
    echo "[FAILED] dataset=${dataset} slot=${slot} code=${code} output=${output_json} time=$(date -Is)" > "${status_path}"
    log_msg "FAILED reader QA dataset=${dataset} slot=${slot} code=${code}"
  fi
  return "${code}"
}

run_dataset() {
  local dataset="$1"
  local status=0
  for slot in $(csv_to_words "${SLOTS}"); do
    run_dataset_slot "${dataset}" "${slot}" || status=1
  done
  return "${status}"
}

main() {
  echo "[START] pcec_m4_position_sensitivity_8b time=$(date -Is)" > "${QA_ROOT}/status/launcher.status"
  log_msg "PCEC m4 admitted-position sensitivity begin"
  log_msg "Slots: ${SLOTS}; slot5 is the canonical PCEC order and is already available from pcec_fresh_e2e_reader_qa."
  local status=0
  run_dataset 2wikimultihopqa &
  local pid_2wiki=$!
  run_dataset hotpotqa &
  local pid_hotpot=$!
  run_dataset musique &
  local pid_musique=$!
  wait "${pid_2wiki}" || status=1
  wait "${pid_hotpot}" || status=1
  wait "${pid_musique}" || status=1
  if [[ "${status}" -eq 0 ]]; then
    echo "[DONE] pcec_m4_position_sensitivity_8b time=$(date -Is)" > "${QA_ROOT}/status/launcher.status"
    log_msg "PCEC m4 admitted-position sensitivity complete"
  else
    echo "[FAILED] pcec_m4_position_sensitivity_8b status=${status} time=$(date -Is)" > "${QA_ROOT}/status/launcher.status"
    log_msg "PCEC m4 admitted-position sensitivity finished with failures"
  fi
  return "${status}"
}

main "$@"
