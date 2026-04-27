#!/usr/bin/env bash
set -euo pipefail

cd /mnt/nvme/code/HippoRAG

run_dataset() {
  local dataset="$1"
  local outdir="$2"
  local output_json="${outdir}/eval_reports/dtc_embed_limit1000_v1_anchor2_fresh8043.json"
  mkdir -p "${outdir}/eval_reports"
  echo "[$(date '+%F %T')] START ${dataset} -> ${output_json}"
  PYTHONUNBUFFERED=1 .venv-hipporag/bin/python -u scripts/eval_causal_qwen3.py \
    --dataset "${dataset}" \
    --limit 1000 \
    --save_dir outputs_step0_general \
    --llm_name qwen3-8b \
    --llm_request_name qwen3-8b-train \
    --llm_base_url http://localhost:8043/v1 \
    --qa_top_k 5 \
    --setwise_selector dtc_embed \
    --setwise_pool_k 100 \
    --setwise_anchor_count 2 \
    --setwise_reserve_top_m 0 \
    --setwise_non_anchor_title_dedup true \
    --dtc_decomposition_mode llm \
    --dtc_enforce_dependencies true \
    --dtc_max_steps 4 \
    --dtc_match_threshold 0.35 \
    --dtc_redundancy_weight 0.10 \
    --dtc_base_weight 0.05 \
    --dtc_anchor_bonus_weight 0.10 \
    --dtc_dependency_bonus_weight 0.10 \
    --dtc_max_completion_tokens 512 \
    --output_json "${output_json}"
  echo "[$(date '+%F %T')] DONE ${dataset}"
}

run_dataset 2wikimultihopqa outputs_step0_general_2wikimultihopqa
run_dataset hotpotqa outputs_step0_general_hotpotqa
run_dataset musique outputs_step0_general_musique

echo "[$(date '+%F %T')] ALL DONE"
