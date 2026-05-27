#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/daec_llm_binding_audit_limit100_20260503"
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
  local pool="$1"
  local dataset="$2"
  case "${pool}" in
    dense) echo "${ROOT_DIR}/run_logs/dense_pool_exports_full1000_20260424/${dataset}_dense_pool100.json" ;;
    hipporag) echo "${ROOT_DIR}/run_logs/hipporag_pool_exports_full1000_20260503/${dataset}_hipporag_pool100.json" ;;
    proprag) echo "${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424/${dataset}_pool100.json" ;;
    *) return 2 ;;
  esac
}

run_one() {
  local pool="$1"
  local match_mode="$2"
  local dataset="$3"
  local port pool_path stem output_json log_path status_path cache_path

  port="$(dataset_port "${dataset}")" || return 2
  pool_path="$(pool_json "${pool}" "${dataset}")" || return 2
  stem="${dataset}_${pool}_${match_mode}_daec_llm_limit100"
  output_json="${OUT_DIR}/${stem}.json"
  log_path="${OUT_DIR}/${stem}.log"
  status_path="${OUT_DIR}/${stem}.status"
  cache_path="${OUT_DIR}/${stem}.binding_cache.json"

  echo "[START] pool=${pool} match=${match_mode} dataset=${dataset} limit=100 start=$(date -Is)" > "${status_path}"
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
      --external_pool_json "${pool_path}" \
      --external_pool_source_name "${pool}_pool100" \
      --external_pool_strict_questions true \
      --setwise_selector daec_noisyor_llm \
      --setwise_pool_k 100 \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --dtc_decomposition_mode llm \
      --dtc_binding_max_candidates 5 \
      --llm_binding_url "http://localhost:${port}/v1" \
      --llm_binding_model "qwen3-8b-train" \
      --llm_binding_cache_path "${cache_path}" \
      --llm_binding_title_match_mode "${match_mode}" \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]] && [[ -s "${output_json}" ]]; then
    echo "[DONE] pool=${pool} match=${match_mode} dataset=${dataset} limit=100 end=$(date -Is)" > "${status_path}"
  else
    echo "[FAILED] pool=${pool} match=${match_mode} dataset=${dataset} limit=100 code=${code} end=$(date -Is)" > "${status_path}"
  fi
  return "${code}"
}

run_wave() {
  local pool="$1"
  local match_mode="$2"
  echo "[WAVE_START] pool=${pool} match=${match_mode} start=$(date -Is)" | tee -a "${OUT_DIR}/launcher.log"
  local pids=()
  local dataset
  for dataset in 2wikimultihopqa hotpotqa musique; do
    run_one "${pool}" "${match_mode}" "${dataset}" &
    pids+=("$!")
  done
  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  echo "[WAVE_DONE] pool=${pool} match=${match_mode} rc=${rc} end=$(date -Is)" | tee -a "${OUT_DIR}/launcher.log"
  return "${rc}"
}

main() {
  echo "[START] daec_llm_binding_audit_limit100 start=$(date -Is)" > "${OUT_DIR}/launcher.status"
  local rc=0
  local pool match_mode
  for pool in dense hipporag proprag; do
    for match_mode in substring exact; do
      run_wave "${pool}" "${match_mode}" || rc=1
    done
  done
  if [[ "${rc}" -eq 0 ]]; then
    echo "[DONE] daec_llm_binding_audit_limit100 end=$(date -Is)" > "${OUT_DIR}/launcher.status"
  else
    echo "[FAILED] daec_llm_binding_audit_limit100 rc=${rc} end=$(date -Is)" > "${OUT_DIR}/launcher.status"
  fi
  return "${rc}"
}

main "$@"
