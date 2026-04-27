#!/usr/bin/env bash
set -euo pipefail

run_variant() {
  local dataset="$1"
  local outdir="$2"
  local limit="$3"
  local output_json="${outdir}/eval_reports/dtc_embed_nvembed_soft_instrfix_limit${limit}_anchor2_fresh8043.json"

  mkdir -p "${outdir}/eval_reports"
  echo "[$(date '+%F %T')] START ${dataset}/limit${limit} -> ${output_json}"
  PYTHONUNBUFFERED=1 .venv-hipporag/bin/python -u scripts/eval_causal_qwen3.py \
    --dataset "${dataset}" \
    --limit "${limit}" \
    --save_dir outputs_step0_general_nvembed \
    --llm_name qwen3-8b \
    --llm_request_name qwen3-8b-train \
    --llm_base_url http://localhost:8043/v1 \
    --embedding_name VLLM/nvidia/NV-Embed-v2 \
    --embedding_base_url http://localhost:8019/v1/embeddings \
    --qa_top_k 5 \
    --setwise_selector dtc_embed \
    --setwise_pool_k 100 \
    --setwise_anchor_count 2 \
    --setwise_reserve_top_m 0 \
    --setwise_non_anchor_title_dedup true \
    --dtc_decomposition_mode llm \
    --dtc_enforce_dependencies true \
    --dtc_require_new_crossing false \
    --dtc_enable_dependency_binding false \
    --dtc_max_steps 4 \
    --dtc_match_threshold 0.35 \
    --dtc_redundancy_weight 0.10 \
    --dtc_base_weight 0.05 \
    --dtc_anchor_bonus_weight 0.10 \
    --dtc_dependency_bonus_weight 0.10 \
    --dtc_max_completion_tokens 512 \
    --output_json "${output_json}"
  echo "[$(date '+%F %T')] DONE ${dataset}/limit${limit}"
}

run_suite() {
  local limit="$1"
  run_variant 2wikimultihopqa outputs_step0_general_nvembed_2wikimultihopqa "${limit}"
  run_variant hotpotqa outputs_step0_general_nvembed_hotpotqa "${limit}"
  run_variant musique outputs_step0_general_nvembed_musique "${limit}"
}

run_suite 20
run_suite 100

echo "[$(date '+%F %T')] ALL_DONE"
