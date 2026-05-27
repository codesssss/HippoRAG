#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_ROOT="${ROOT_DIR}/run_logs/etv3_dbec_latest_full1000_20260510"
BASE_RUN_ROOT="${ROOT_DIR}/run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_full_20260510"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
LIMIT=1000
POOL_K=100
SETWISE_POOL_K=100

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/status" "${OUT_ROOT}/pools" "${OUT_ROOT}/evals"

dataset_port() {
  case "$1" in
    musique) echo "8041" ;;
    hotpotqa) echo "8042" ;;
    2wikimultihopqa) echo "8043" ;;
    *) return 2 ;;
  esac
}

retrieval_report_path() {
  local dataset="$1"
  echo "${BASE_RUN_ROOT}/runs/${dataset}/${dataset}/reports/${dataset}_evidence_transition_graphragv3_variable_flow_retrieval.json"
}

openie_path() {
  local dataset="$1"
  echo "${BASE_RUN_ROOT}/runs/${dataset}/${dataset}/index/openie_results_ner_qwen3-8b-train.json"
}

run_dataset() {
  local dataset="$1"
  local port
  port="$(dataset_port "${dataset}")" || return 2
  local status_path="${OUT_ROOT}/status/${dataset}.status"
  local log_path="${OUT_ROOT}/logs/${dataset}.log"

  echo "[START] dataset=${dataset} time=$(date -Is) log=${log_path}" > "${status_path}"
  {
    echo "[START] dataset=${dataset} time=$(date -Is)"
    cd "${ROOT_DIR}" || exit 1
    PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
      evidence_transition_graphragv3_dbec_latest/run_eval.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --pool-k "${POOL_K}" \
      --setwise-pool-k "${SETWISE_POOL_K}" \
      --output-root "${OUT_ROOT}" \
      --retrieval-report "$(retrieval_report_path "${dataset}")" \
      --openie-results "$(openie_path "${dataset}")" \
      --llm-name qwen3-8b \
      --llm-request-name qwen3-8b-train \
      --llm-base-url "http://localhost:${port}/v1" \
      --qwen-disable-thinking \
      --embedding-name VLLM/nvidia/NV-Embed-v2 \
      --embedding-base-url "${EMBEDDING_BASE_URL}"
    rc=$?
    if [[ "${rc}" -eq 0 ]]; then
      echo "[DONE] dataset=${dataset} time=$(date -Is)"
      echo "[DONE] dataset=${dataset} time=$(date -Is)" > "${status_path}"
    else
      echo "[FAILED] dataset=${dataset} rc=${rc} time=$(date -Is)"
      echo "[FAILED] dataset=${dataset} rc=${rc} time=$(date -Is)" > "${status_path}"
    fi
    exit "${rc}"
  } > "${log_path}" 2>&1
}

main() {
  echo "[START] etv3_dbec_latest_full1000 time=$(date -Is)" | tee "${OUT_ROOT}/status/launcher.status"
  run_dataset 2wikimultihopqa &
  pid_2wiki=$!
  run_dataset hotpotqa &
  pid_hotpot=$!
  run_dataset musique &
  pid_musique=$!

  echo "2wikimultihopqa ${pid_2wiki}" > "${OUT_ROOT}/status/pids.txt"
  echo "hotpotqa ${pid_hotpot}" >> "${OUT_ROOT}/status/pids.txt"
  echo "musique ${pid_musique}" >> "${OUT_ROOT}/status/pids.txt"

  wait "${pid_2wiki}"
  rc_2wiki=$?
  wait "${pid_hotpot}"
  rc_hotpot=$?
  wait "${pid_musique}"
  rc_musique=$?

  echo "2wikimultihopqa rc=${rc_2wiki}" > "${OUT_ROOT}/status/final_rc.txt"
  echo "hotpotqa rc=${rc_hotpot}" >> "${OUT_ROOT}/status/final_rc.txt"
  echo "musique rc=${rc_musique}" >> "${OUT_ROOT}/status/final_rc.txt"

  if [[ "${rc_2wiki}" -eq 0 && "${rc_hotpot}" -eq 0 && "${rc_musique}" -eq 0 ]]; then
    echo "[DONE] etv3_dbec_latest_full1000 time=$(date -Is)" | tee -a "${OUT_ROOT}/status/launcher.status"
    exit 0
  fi

  echo "[FAILED] etv3_dbec_latest_full1000 time=$(date -Is)" | tee -a "${OUT_ROOT}/status/launcher.status"
  exit 1
}

main "$@"
