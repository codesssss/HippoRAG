#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
OUT_DIR="${ROOT_DIR}/run_logs/layer1_targeted_nobinding_20260424"
SAVE_DIR="outputs_step0_general_nvembed"
mkdir -p "${OUT_DIR}"

run_nobinding() {
  local dataset="$1"
  local pool_source="$2"
  local pool_json="$3"
  local port="$4"
  local label="${dataset}_${pool_source}_nobinding"
  local output_json="${OUT_DIR}/${label}.json"
  local log_path="${OUT_DIR}/${label}.log"
  local status_path="${OUT_DIR}/${label}.status"

  if [[ -s "${output_json}" ]]; then
    echo "[SKIP] ${label} already exists: ${output_json}" | tee "${status_path}"
    return 0
  fi
  if [[ ! -s "${pool_json}" ]]; then
    echo "[ERROR] missing pool_json=${pool_json}" | tee "${status_path}"
    return 2
  fi

  echo "[START] ${label} port=${port} $(date -Is)" | tee "${status_path}"
  (
    cd "${ROOT_DIR}"
    .venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" --limit 1000 --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b --llm_request_name qwen3-8b-train \
      --max_retry_attempts 20 \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name VLLM/nvidia/NV-Embed-v2 \
      --embedding_base_url http://localhost:8019/v1/embeddings \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --external_pool_json "${pool_json}" \
      --external_pool_source_name "${pool_source}" \
      --setwise_selector dtc_embed \
      --setwise_pool_k 100 \
      --setwise_reserve_top_m 0 \
      --dtc_rank_weight 0.2 \
      --dtc_include_satisfiable_by true \
      --dtc_repairable_filter_enabled true \
      --dtc_satisfiable_by_policy strict \
      --dtc_enable_dependency_binding false \
      --output_json "${output_json}"
  ) >> "${log_path}" 2>&1
  echo "[DONE] ${label} $(date -Is) output=${output_json}" | tee "${status_path}"
}

echo "[START] targeted nobinding ablations $(date -Is)" | tee "${OUT_DIR}/launcher.status"

(
  run_nobinding \
    "hotpotqa" \
    "proprag_clean_nothink_top100" \
    "${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424/hotpotqa_pool100.json" \
    "8041"
  run_nobinding \
    "musique" \
    "dense_nvembed_top100" \
    "${ROOT_DIR}/run_logs/dense_pool_exports_full1000_20260424/musique_dense_pool100.json" \
    "8041"
) &
pid_a=$!

(
  run_nobinding \
    "musique" \
    "proprag_clean_nothink_top100" \
    "${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424/musique_pool100.json" \
    "8042"
) &
pid_b=$!

(
  run_nobinding \
    "2wikimultihopqa" \
    "dense_nvembed_top100" \
    "${ROOT_DIR}/run_logs/dense_pool_exports_full1000_20260424/2wikimultihopqa_dense_pool100.json" \
    "8043"
) &
pid_c=$!

echo "${pid_a}" > "${OUT_DIR}/queue_a.pid"
echo "${pid_b}" > "${OUT_DIR}/queue_b.pid"
echo "${pid_c}" > "${OUT_DIR}/queue_c.pid"

wait "${pid_a}" "${pid_b}" "${pid_c}"
echo "[DONE] targeted nobinding ablations $(date -Is)" | tee "${OUT_DIR}/launcher.status"
