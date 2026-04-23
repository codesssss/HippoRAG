#!/usr/bin/env bash
set -euo pipefail

cd /mnt/nvme/code/HippoRAG

export PYTHONUNBUFFERED=1

DATASETS=(2wikimultihopqa hotpotqa musique)
ALPHAS=(0.50 0.67 0.80 0.90 1.00)

for dataset in "${DATASETS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    tag="${alpha//./p}"
    output_json="outputs_step0_general_nvembed_${dataset}/eval_reports/dtc_embed_nvembed_rankw0p2_demandgate_a${tag}_pilot100_anchor2_8043.json"
    echo "[$(date '+%F %T')] START ${dataset} demand_gate_alpha=${alpha} -> ${output_json}"
    .venv-hipporag/bin/python -u scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit 100 \
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
      --dtc_demand_gate_enabled true \
      --dtc_demand_gate_alpha "${alpha}" \
      --dtc_max_steps 4 \
      --dtc_match_threshold 0.35 \
      --dtc_redundancy_weight 0.10 \
      --dtc_base_weight 0.05 \
      --dtc_rank_weight 0.2 \
      --dtc_anchor_bonus_weight 0.10 \
      --dtc_dependency_bonus_weight 0.10 \
      --dtc_max_completion_tokens 512 \
      --output_json "${output_json}"
    echo "[$(date '+%F %T')] DONE ${dataset} demand_gate_alpha=${alpha}"
  done
done
