#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
POOL_DIR="${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424"
OUT_DIR="${ROOT_DIR}/run_logs/llm_direct_select_proprag_limit100_20260506"
SAVE_DIR="outputs_step0_general_nvembed"
LIMIT=100
POOL_K=100
SELECTION_COUNT=5
SNIPPET_CHARS=128
MODEL="qwen3-8b-train"

mkdir -p "${OUT_DIR}/selected_pools" "${OUT_DIR}/caches"

dataset_to_port() {
  case "$1" in
    2wikimultihopqa) echo 8041 ;;
    hotpotqa) echo 8042 ;;
    musique) echo 8043 ;;
    *) echo "[ERROR] unknown dataset=$1" >&2; return 2 ;;
  esac
}

variant_to_context_mode() {
  case "$1" in
    title) echo title ;;
    snippet128) echo snippet ;;
    *) echo "[ERROR] unknown variant=$1" >&2; return 2 ;;
  esac
}

run_variant() {
  local dataset="$1"
  local variant="$2"
  local port="$3"
  local context_mode
  context_mode="$(variant_to_context_mode "${variant}")"

  local input_pool="${POOL_DIR}/${dataset}_pool100.json"
  local selected_pool="${OUT_DIR}/selected_pools/${dataset}_${variant}_pool100.json"
  local cache_jsonl="${OUT_DIR}/caches/${dataset}_${variant}.jsonl"
  local select_log="${OUT_DIR}/${dataset}_${variant}_select.log"
  local eval_log="${OUT_DIR}/${dataset}_${variant}_eval.log"
  local output_json="${OUT_DIR}/${dataset}_${variant}_llm_direct.json"
  local status_path="${OUT_DIR}/${dataset}_${variant}.status"

  if [[ ! -s "${input_pool}" ]]; then
    echo "[ERROR] missing input pool: ${input_pool}" > "${status_path}"
    return 2
  fi

  echo "[SELECT_START] dataset=${dataset} variant=${variant} port=${port} start=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" scripts/run_llm_direct_pool_selector.py \
      --input_pool_json "${input_pool}" \
      --output_pool_json "${selected_pool}" \
      --cache_jsonl "${cache_jsonl}" \
      --limit "${LIMIT}" \
      --pool_k "${POOL_K}" \
      --selection_count "${SELECTION_COUNT}" \
      --context_mode "${context_mode}" \
      --snippet_chars "${SNIPPET_CHARS}" \
      --llm_base_url "http://localhost:${port}/v1" \
      --model "${MODEL}" \
      --max_tokens 256 \
      --temperature 0.0 \
      --num_workers 1 \
      --resume
  ) > "${select_log}" 2>&1

  echo "[EVAL_START] dataset=${dataset} variant=${variant} port=${port} start=$(date -Is) selected_pool=${selected_pool}" > "${status_path}"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name "${MODEL}" \
      --max_retry_attempts 20 \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name VLLM/nvidia/NV-Embed-v2 \
      --embedding_base_url http://localhost:8019/v1/embeddings \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --external_pool_json "${selected_pool}" \
      --external_pool_source_name "llm_direct_${variant}_proprag_top100" \
      --external_pool_strict_questions true \
      --setwise_selector none \
      --output_json "${output_json}"
  ) > "${eval_log}" 2>&1

  echo "[DONE] dataset=${dataset} variant=${variant} end=$(date -Is) output=${output_json}" > "${status_path}"
}

run_dataset() {
  local dataset="$1"
  local port
  port="$(dataset_to_port "${dataset}")"
  local status_path="${OUT_DIR}/${dataset}.status"

  echo "[START] dataset=${dataset} port=${port} start=$(date -Is)" > "${status_path}"
  run_variant "${dataset}" title "${port}"
  run_variant "${dataset}" snippet128 "${port}"
  echo "[DONE] dataset=${dataset} end=$(date -Is)" > "${status_path}"
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
