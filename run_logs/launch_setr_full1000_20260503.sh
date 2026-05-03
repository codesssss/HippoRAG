#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/setr_full1000_20260503"
REPORT_DIR="${ROOT_DIR}/reports/setr_full1000_20260503"
SAVE_DIR="outputs_step0_general_nvembed"
LLM_MODEL="qwen3-8b-train"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"
LIMIT=1000

mkdir -p "${OUT_DIR}" "${REPORT_DIR}"

dataset_port() {
  case "$1" in
    2wikimultihopqa) echo 8041 ;;
    hotpotqa) echo 8042 ;;
    musique) echo 8043 ;;
    *) return 2 ;;
  esac
}

pool_path() {
  local pool="$1"
  local dataset="$2"
  case "${pool}" in
    dense) echo "${ROOT_DIR}/run_logs/dense_pool_exports_full1000_20260424/${dataset}_dense_pool100.json" ;;
    hipporag) echo "${ROOT_DIR}/run_logs/hipporag_pool_exports_full1000_20260503/${dataset}_hipporag_pool100.json" ;;
    proprag) echo "${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424/${dataset}_pool100.json" ;;
    *) return 2 ;;
  esac
}

log_msg() {
  echo "[$(date -Is)] $*" | tee -a "${OUT_DIR}/launcher.log"
}

run_cmd() {
  local label="$1"
  shift
  local log_path="${OUT_DIR}/${label}.log"
  local status_path="${OUT_DIR}/${label}.status"

  if [[ -f "${status_path}" ]] && grep -q '^DONE ' "${status_path}"; then
    log_msg "SKIP ${label}"
    return 0
  fi

  echo "START ${label} $(date -Is)" > "${status_path}"
  log_msg "START ${label}"
  (
    cd "${ROOT_DIR}" || exit 2
    "$@"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    echo "DONE ${label} $(date -Is)" > "${status_path}"
    log_msg "DONE ${label}"
  else
    echo "FAILED ${label} code=${code} $(date -Is)" > "${status_path}"
    log_msg "FAILED ${label} code=${code}"
    return "${code}"
  fi
}

eval_selected_pool() {
  local label="$1"
  local dataset="$2"
  local pool="$3"
  local selected_pool_json="$4"
  local output_json="$5"
  local port="$6"

  run_cmd "eval_${label}" \
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name "${LLM_MODEL}" \
      --max_retry_attempts 20 \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name "${EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --external_pool_json "${selected_pool_json}" \
      --external_pool_source_name "${label}_${pool}_full1000" \
      --external_pool_strict_questions true \
      --setwise_selector none \
      --output_json "${output_json}"
}

run_direct_setr() {
  local dataset="$1"
  local pool="$2"
  local variant="$3"
  local pool_k="$4"
  local doc_chars="$5"
  local max_tokens="$6"
  local port="$7"

  local source_pool prefix input_jsonl selection_jsonl selected_pool_json output_json
  source_pool="$(pool_path "${pool}" "${dataset}")" || return 2
  prefix="${OUT_DIR}/${dataset}_${pool}_${variant}"
  input_jsonl="${prefix}.inputs.jsonl"
  selection_jsonl="${prefix}.selection.jsonl"
  selected_pool_json="${prefix}.selected_pool.json"
  output_json="${REPORT_DIR}/${dataset}_${pool}_${variant}.eval.json"

  run_cmd "export_${dataset}_${pool}_${variant}" \
    "${PYTHON_BIN}" scripts/export_setr_inputs.py \
      --pool_json "${source_pool}" \
      --output_jsonl "${input_jsonl}" \
      --pool_k "${pool_k}" \
      --doc_max_chars "${doc_chars}" \
      --limit "${LIMIT}"

  run_cmd "select_${dataset}_${pool}_${variant}" \
    "${PYTHON_BIN}" scripts/run_setr_style_selector.py \
      --input_jsonl "${input_jsonl}" \
      --output_jsonl "${selection_jsonl}" \
      --llm_base_url "http://localhost:${port}/v1" \
      --model "${LLM_MODEL}" \
      --max_tokens "${max_tokens}" \
      --temperature 0 \
      --num_workers 1 \
      --resume

  run_cmd "apply_${dataset}_${pool}_${variant}" \
    "${PYTHON_BIN}" scripts/apply_setr_selection_to_pool.py \
      --pool_json "${source_pool}" \
      --selection_jsonl "${selection_jsonl}" \
      --output_pool_json "${selected_pool_json}" \
      --limit "${LIMIT}"

  eval_selected_pool "${variant}" "${dataset}" "${pool}" "${selected_pool_json}" "${output_json}" "${port}"
}

run_windowed_setr() {
  local dataset="$1"
  local pool="$2"
  local port="$3"
  local variant="setr_windowed_k50_doc768"

  local source_pool prefix selection_jsonl selected_pool_json output_json
  source_pool="$(pool_path "${pool}" "${dataset}")" || return 2
  prefix="${OUT_DIR}/${dataset}_${pool}_${variant}"
  selection_jsonl="${prefix}.selection.jsonl"
  selected_pool_json="${prefix}.selected_pool.json"
  output_json="${REPORT_DIR}/${dataset}_${pool}_${variant}.eval.json"

  run_cmd "select_${dataset}_${pool}_${variant}" \
    "${PYTHON_BIN}" scripts/run_setr_windowed_selector.py \
      --pool_json "${source_pool}" \
      --output_selection_jsonl "${selection_jsonl}" \
      --pool_k 50 \
      --window_size 20 \
      --stage1_max_per_window 5 \
      --stage2_max_candidates 25 \
      --final_max_count 5 \
      --doc_max_chars 768 \
      --llm_base_url "http://localhost:${port}/v1" \
      --model "${LLM_MODEL}" \
      --max_tokens 512 \
      --temperature 0 \
      --limit "${LIMIT}" \
      --num_workers 1 \
      --resume

  run_cmd "apply_${dataset}_${pool}_${variant}" \
    "${PYTHON_BIN}" scripts/apply_setr_selection_to_pool.py \
      --pool_json "${source_pool}" \
      --selection_jsonl "${selection_jsonl}" \
      --output_pool_json "${selected_pool_json}" \
      --limit "${LIMIT}"

  eval_selected_pool "${variant}" "${dataset}" "${pool}" "${selected_pool_json}" "${output_json}" "${port}"
}

run_dataset_worker() {
  local dataset="$1"
  local port
  port="$(dataset_port "${dataset}")" || return 2

  local pool
  for pool in dense hipporag proprag; do
    run_direct_setr "${dataset}" "${pool}" "setr_k20_doc768" 20 768 1024 "${port}" || return 1
    run_direct_setr "${dataset}" "${pool}" "setr_k100_doc160" 100 160 512 "${port}" || return 1
    run_windowed_setr "${dataset}" "${pool}" "${port}" || return 1
  done
}

main() {
  echo "[START] setr_full1000 $(date -Is)" > "${OUT_DIR}/launcher.status"
  echo "[START] setr_full1000 $(date -Is)" > "${OUT_DIR}/launcher.log"

  local pids=()
  local dataset
  for dataset in 2wikimultihopqa hotpotqa musique; do
    (
      run_dataset_worker "${dataset}"
      code=$?
      if [[ "${code}" -eq 0 ]]; then
        echo "DONE dataset=${dataset} $(date -Is)" > "${OUT_DIR}/${dataset}.status"
      else
        echo "FAILED dataset=${dataset} code=${code} $(date -Is)" > "${OUT_DIR}/${dataset}.status"
      fi
      exit "${code}"
    ) &
    pids+=("$!")
  done

  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done

  if [[ "${rc}" -eq 0 ]]; then
    echo "[DONE] setr_full1000 $(date -Is)" > "${OUT_DIR}/launcher.status"
  else
    echo "[FAILED] setr_full1000 rc=${rc} $(date -Is)" > "${OUT_DIR}/launcher.status"
  fi
  return "${rc}"
}

main "$@"
