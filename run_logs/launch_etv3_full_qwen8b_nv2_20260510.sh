#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_ROOT="${ROOT_DIR}/run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_full_20260510"

LLM_MODEL="qwen3-8b-train"
EMBEDDING_NAME="nvidia/NV-Embed-v2"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
MAX_QUERIES=0
CANDIDATE_POOL_K=200
TOP_K=5
EMBEDDING_BATCH_SIZE=8
MAX_RETRY_ATTEMPTS=20

export HIPPORAG_RERANK_FORCE_NO_THINK=1

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/status" "${OUT_ROOT}/reports" "${OUT_ROOT}/runs" "${OUT_ROOT}/reader_runtime"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/logs/launcher.log"
}

dataset_port() {
  case "$1" in
    musique) echo "8041" ;;
    hotpotqa) echo "8042" ;;
    2wikimultihopqa) echo "8043" ;;
    *) return 2 ;;
  esac
}

check_endpoint() {
  local label="$1"
  local url="$2"
  if curl -fsS "${url}" > /dev/null; then
    log_msg "OK endpoint ${label} ${url}"
    return 0
  fi
  log_msg "FAILED endpoint ${label} ${url}"
  return 1
}

retrieval_report_path() {
  local dataset="$1"
  echo "${OUT_ROOT}/runs/${dataset}/${dataset}/reports/${dataset}_evidence_transition_graphragv3_variable_flow_retrieval.json"
}

run_retrieval() {
  local dataset="$1"
  local port="$2"
  local run_root="${OUT_ROOT}/runs/${dataset}"
  local output_json
  output_json="$(retrieval_report_path "${dataset}")"
  local status_path="${OUT_ROOT}/status/${dataset}_retrieval.status"
  local log_path="${OUT_ROOT}/logs/${dataset}_retrieval.log"

  if [[ -s "${output_json}" ]]; then
    echo "[SKIP] dataset=${dataset} stage=retrieval existing=${output_json} time=$(date -Is)" > "${status_path}"
    log_msg "SKIP retrieval dataset=${dataset} existing=${output_json}"
    return 0
  fi

  echo "[START] dataset=${dataset} stage=retrieval max_queries=${MAX_QUERIES} port=${port} time=$(date -Is)" > "${status_path}"
  log_msg "START retrieval dataset=${dataset} port=${port}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv3_variable_flow/run_fresh_e2e.py \
      --datasets "${dataset}" \
      --output-root "${run_root}" \
      --max-queries "${MAX_QUERIES}" \
      --llm-name "${LLM_MODEL}" \
      --llm-base-url "http://localhost:${port}/v1" \
      --max-retry-attempts "${MAX_RETRY_ATTEMPTS}" \
      --qwen-disable-thinking \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --candidate-pool-k "${CANDIDATE_POOL_K}" \
      --top-k "${TOP_K}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]] && [[ -s "${output_json}" ]]; then
    echo "[DONE] dataset=${dataset} stage=retrieval output=${output_json} time=$(date -Is)" > "${status_path}"
    log_msg "DONE retrieval dataset=${dataset} output=${output_json}"
    return 0
  fi
  echo "[FAILED] dataset=${dataset} stage=retrieval code=${code} expected=${output_json} time=$(date -Is)" > "${status_path}"
  log_msg "FAILED retrieval dataset=${dataset} code=${code}"
  return 1
}

run_reader_qa() {
  local dataset="$1"
  local port="$2"
  local retrieval_json
  retrieval_json="$(retrieval_report_path "${dataset}")"
  local output_json="${OUT_ROOT}/reports/${dataset}_evidence_transition_graphragv3_variable_flow_qa.json"
  local output_md="${OUT_ROOT}/reports/${dataset}_evidence_transition_graphragv3_variable_flow_qa.md"
  local status_path="${OUT_ROOT}/status/${dataset}_qa.status"
  local log_path="${OUT_ROOT}/logs/${dataset}_qa.log"

  if [[ ! -s "${retrieval_json}" ]]; then
    echo "[FAILED] dataset=${dataset} stage=qa missing_retrieval=${retrieval_json} time=$(date -Is)" > "${status_path}"
    log_msg "FAILED qa dataset=${dataset} missing retrieval report"
    return 1
  fi

  if [[ -s "${output_json}" ]]; then
    echo "[SKIP] dataset=${dataset} stage=qa existing=${output_json} time=$(date -Is)" > "${status_path}"
    log_msg "SKIP qa dataset=${dataset} existing=${output_json}"
    return 0
  fi

  echo "[START] dataset=${dataset} stage=qa max_queries=${MAX_QUERIES} port=${port} time=$(date -Is)" > "${status_path}"
  log_msg "START qa dataset=${dataset} port=${port}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv3_variable_flow/run_reader_qa.py \
      --retrieval-reports "${retrieval_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${TOP_K}" \
      --save-dir "${OUT_ROOT}/reader_runtime/${dataset}" \
      --llm-name "${LLM_MODEL}" \
      --llm-base-url "http://localhost:${port}/v1" \
      --max-new-tokens 2048 \
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
    echo "[DONE] dataset=${dataset} stage=qa output=${output_json} time=$(date -Is)" > "${status_path}"
    log_msg "DONE qa dataset=${dataset} output=${output_json}"
    return 0
  fi
  echo "[FAILED] dataset=${dataset} stage=qa code=${code} expected=${output_json} time=$(date -Is)" > "${status_path}"
  log_msg "FAILED qa dataset=${dataset} code=${code}"
  return 1
}

run_dataset() {
  local dataset="$1"
  local port
  port="$(dataset_port "${dataset}")" || return 2
  run_retrieval "${dataset}" "${port}" || return 1
  run_reader_qa "${dataset}" "${port}" || return 1
}

main() {
  echo "[START] etv3_full_qwen8b_nv2 time=$(date -Is)" > "${OUT_ROOT}/status/launcher.status"
  log_msg "ETv3 variable-flow Qwen3-8B + NV-Embed-v2 full run begin"
  log_msg "output_root=${OUT_ROOT}"
  log_msg "lanes: musique->8041 hotpotqa->8042 2wikimultihopqa->8043"
  log_msg "max_queries=${MAX_QUERIES} candidate_pool_k=${CANDIDATE_POOL_K} top_k=${TOP_K}"
  log_msg "fresh-index policy: rebuild per dataset in this output root; no legacy index reuse"

  local preflight=0
  check_endpoint "qwen8041" "http://localhost:8041/v1/models" || preflight=1
  check_endpoint "qwen8042" "http://localhost:8042/v1/models" || preflight=1
  check_endpoint "qwen8043" "http://localhost:8043/v1/models" || preflight=1
  check_endpoint "nv-embed" "http://localhost:8019/v1/models" || preflight=1
  if [[ "${preflight}" -ne 0 ]]; then
    echo "[FAILED] etv3_full_qwen8b_nv2 preflight_failed time=$(date -Is)" > "${OUT_ROOT}/status/launcher.status"
    return 1
  fi

  (run_dataset musique) &
  local pid_musique=$!
  (run_dataset hotpotqa) &
  local pid_hotpotqa=$!
  (run_dataset 2wikimultihopqa) &
  local pid_2wiki=$!

  local status=0
  if ! wait "${pid_musique}"; then
    log_msg "FAILED dataset=musique"
    status=1
  fi
  if ! wait "${pid_hotpotqa}"; then
    log_msg "FAILED dataset=hotpotqa"
    status=1
  fi
  if ! wait "${pid_2wiki}"; then
    log_msg "FAILED dataset=2wikimultihopqa"
    status=1
  fi

  if [[ "${status}" -eq 0 ]]; then
    echo "[DONE] etv3_full_qwen8b_nv2 time=$(date -Is)" > "${OUT_ROOT}/status/launcher.status"
    log_msg "ETv3 variable-flow Qwen3-8B + NV-Embed-v2 full run complete"
  else
    echo "[FAILED] etv3_full_qwen8b_nv2 status=${status} time=$(date -Is)" > "${OUT_ROOT}/status/launcher.status"
    log_msg "ETv3 variable-flow Qwen3-8B + NV-Embed-v2 full run finished with failures"
  fi
  return "${status}"
}

main "$@"
