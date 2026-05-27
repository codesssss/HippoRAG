#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/minimal_demand_repair_limit100_aligned_20260501"
SAVE_DIR="outputs_step0_general_nvembed"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
export HIPPORAG_RERANK_FORCE_NO_THINK=1

mkdir -p "${OUT_DIR}"

dataset_port() {
  case "$1" in
    2wikimultihopqa) echo 8041 ;;
    hotpotqa) echo 8042 ;;
    musique) echo 8043 ;;
    *) return 2 ;;
  esac
}

pool_json() {
  case "$1" in
    2wikimultihopqa) echo "${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json" ;;
    hotpotqa) echo "${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424/hotpotqa_pool100.json" ;;
    musique) echo "${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424/musique_pool100.json" ;;
    *) return 2 ;;
  esac
}

run_one() {
  local dataset="$1"
  local port pool output_json log_path status_path
  port="$(dataset_port "${dataset}")" || return 2
  pool="$(pool_json "${dataset}")" || return 2
  output_json="${OUT_DIR}/${dataset}_minimal_demand_repair_limit100_aligned.json"
  log_path="${OUT_DIR}/${dataset}_minimal_demand_repair_limit100_aligned.log"
  status_path="${OUT_DIR}/${dataset}_minimal_demand_repair_limit100_aligned.status"

  echo "[START] dataset=${dataset} variant=minimal_demand_repair limit=100 aligned=true port=${port} start=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit 100 \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name qwen3-8b-train \
      --max_retry_attempts 20 \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name VLLM/nvidia/NV-Embed-v2 \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --external_pool_json "${pool}" \
      --external_pool_source_name proprag_pool100 \
      --external_pool_strict_questions true \
      --setwise_selector minimal_demand_repair \
      --setwise_pool_k 100 \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --dtc_decomposition_mode llm \
      --dtc_binding_max_candidates 5 \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]] && [[ -s "${output_json}" ]]; then
    echo "[DONE] dataset=${dataset} variant=minimal_demand_repair limit=100 aligned=true end=$(date -Is) output=${output_json}" > "${status_path}"
  else
    echo "[FAILED] dataset=${dataset} variant=minimal_demand_repair limit=100 aligned=true code=${code} end=$(date -Is) log=${log_path}" > "${status_path}"
  fi
  return "${code}"
}

main() {
  echo "[START] minimal_demand_repair_limit100_aligned start=$(date -Is)" > "${OUT_DIR}/launcher.status"
  local pids=()
  local dataset
  for dataset in 2wikimultihopqa hotpotqa musique; do
    run_one "${dataset}" &
    pids+=("$!")
  done
  local rc=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  if [[ "${rc}" -eq 0 ]]; then
    echo "[DONE] minimal_demand_repair_limit100_aligned end=$(date -Is)" > "${OUT_DIR}/launcher.status"
  else
    echo "[FAILED] minimal_demand_repair_limit100_aligned rc=${rc} end=$(date -Is)" > "${OUT_DIR}/launcher.status"
  fi
  return "${rc}"
}

main "$@"
