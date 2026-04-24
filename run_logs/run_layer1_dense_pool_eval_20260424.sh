#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="run_logs/layer1_dense_pool_eval_20260424"
POOL_DIR="run_logs/dense_pool_exports_full1000_20260424"
mkdir -p "${RUN_DIR}"

run_one() {
  local dataset="$1"
  local port="$2"
  local pool_json="${POOL_DIR}/${dataset}_dense_pool100.json"
  local output_json="${RUN_DIR}/${dataset}_dense_pool_daec_oracle.json"
  local log_file="${RUN_DIR}/${dataset}.log"
  local status_file="${RUN_DIR}/${dataset}.status"

  echo "START dataset=${dataset} port=${port} start=$(date '+%F %T') pool=${pool_json}" > "${status_file}"
  .venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
    --dataset "${dataset}" --limit 1000 \
    --save_dir outputs_step0_general_nvembed \
    --llm_name qwen3-8b --llm_request_name qwen3-8b-train \
    --max_retry_attempts 20 \
    --llm_base_url "http://localhost:${port}/v1" \
    --embedding_name VLLM/nvidia/NV-Embed-v2 \
    --embedding_base_url http://localhost:8019/v1/embeddings \
    --qa_top_k 5 \
    --qa_doc_max_chars 2048 \
    --external_pool_json "${pool_json}" \
    --external_pool_source_name dense_nvembed_top100 \
    --oracle_select_k 100 \
    --setwise_selector dtc_embed \
    --setwise_pool_k 100 \
    --setwise_reserve_top_m 0 \
    --dtc_rank_weight 0.2 \
    --dtc_include_satisfiable_by true \
    --dtc_repairable_filter_enabled true \
    --dtc_satisfiable_by_policy binding_override \
    --dtc_enable_dependency_binding true \
    --dtc_binding_max_candidates 4 \
    --dtc_binding_entity_hit_required true \
    --output_json "${output_json}" >> "${log_file}" 2>&1

  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    echo "DONE dataset=${dataset} end=$(date '+%F %T') output=${output_json}" > "${status_file}"
  else
    echo "FAILED dataset=${dataset} code=${code} end=$(date '+%F %T') output=${output_json}" > "${status_file}"
  fi
  return "${code}"
}

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 <dataset> <port>" >&2
  echo "Example: $0 2wikimultihopqa 8041" >&2
  exit 2
fi

run_one "$1" "$2"
