#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/ircot_style_limit100_20260506"
SAVE_DIR="outputs_step0_general_nvembed"
LIMIT=100
MAX_ITER=3
TOP_K_PER_ITER=5
QA_TOP_K=5
MODEL="qwen3-8b-train"

mkdir -p "${OUT_DIR}"

dataset_to_port() {
  case "$1" in
    2wikimultihopqa) echo 8041 ;;
    hotpotqa) echo 8042 ;;
    musique) echo 8043 ;;
    *) echo "[ERROR] unknown dataset=$1" >&2; return 2 ;;
  esac
}

run_dataset() {
  local dataset="$1"
  local port
  port="$(dataset_to_port "${dataset}")"
  local report_json="${OUT_DIR}/${dataset}_ircot_style.json"
  local report_md="${OUT_DIR}/${dataset}_ircot_style.md"
  local log_path="${OUT_DIR}/${dataset}.log"
  local status_path="${OUT_DIR}/${dataset}.status"

  echo "[START] dataset=${dataset} port=${port} start=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" scripts/bsgs_run_ircot_baseline.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --max_iter "${MAX_ITER}" \
      --top_k_per_iter "${TOP_K_PER_ITER}" \
      --qa_top_k "${QA_TOP_K}" \
      --final_doc_order round_robin \
      --qa_doc_max_chars 2048 \
      --save_dir "${SAVE_DIR}" \
      --llm_base_url "http://localhost:${port}/v1" \
      --llm_name qwen3-8b \
      --llm_request_name "${MODEL}" \
      --embedding_name VLLM/nvidia/NV-Embed-v2 \
      --embedding_base_url http://localhost:8019/v1/embeddings \
      --max_retry_attempts 20 \
      --openie_mode online \
      --report_json "${report_json}" \
      --report_md "${report_md}"
  ) > "${log_path}" 2>&1

  echo "[DONE] dataset=${dataset} end=$(date -Is) output=${report_json}" > "${status_path}"
}

if [[ "$#" -gt 0 ]]; then
  datasets=("$@")
else
  datasets=(2wikimultihopqa hotpotqa musique)
fi

echo "[START] launcher $(date -Is) datasets=${datasets[*]}" > "${OUT_DIR}/launcher.status"

rc=0
pids=()
for dataset in "${datasets[@]}"; do
  run_dataset "${dataset}" &
  pid=$!
  pids+=("${pid}")
  echo "${pid}" > "${OUT_DIR}/${dataset}.pid"
done

echo "[RUNNING] launcher $(date -Is) pids=${pids[*]}" > "${OUT_DIR}/launcher.status"

for pid in "${pids[@]}"; do
  wait "${pid}" || rc=1
done

if [[ "${rc}" -eq 0 ]]; then
  echo "[DONE] launcher $(date -Is)" > "${OUT_DIR}/launcher.status"
else
  echo "[FAILED] launcher $(date -Is)" > "${OUT_DIR}/launcher.status"
fi
exit "${rc}"
