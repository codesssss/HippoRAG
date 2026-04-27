#!/usr/bin/env bash
set -euo pipefail

cd /mnt/nvme/code/HippoRAG

run_variant() {
  local dataset="$1"
  local outdir="$2"
  local variant="$3"
  shift 3
  local output_json="${outdir}/eval_reports/dtc_embed_ablate_${variant}_pilot100_anchor2_fresh8042.json"
  mkdir -p "${outdir}/eval_reports"
  echo "[$(date '+%F %T')] START ${dataset}/${variant} -> ${output_json}"
  PYTHONUNBUFFERED=1 .venv-hipporag/bin/python -u scripts/eval_causal_qwen3.py \
    --dataset "${dataset}" \
    --limit 100 \
    --save_dir outputs_step0_general \
    --llm_name qwen3-8b \
    --llm_request_name qwen3-8b-train \
    --llm_base_url http://localhost:8042/v1 \
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
    "$@" \
    --output_json "${output_json}"
  echo "[$(date '+%F %T')] DONE ${dataset}/${variant}"
}

run_dataset_variants() {
  local dataset="$1"
  local outdir="$2"
  run_variant "${dataset}" "${outdir}" nodecomp --dtc_decomposition_mode query
  run_variant "${dataset}" "${outdir}" nodep --dtc_enforce_dependencies false
  run_variant "${dataset}" "${outdir}" nored --dtc_redundancy_weight 0.0
  run_variant "${dataset}" "${outdir}" noreserve --setwise_anchor_count 0 --setwise_reserve_top_m 0
}

run_dataset_variants 2wikimultihopqa outputs_step0_general_2wikimultihopqa
run_dataset_variants hotpotqa outputs_step0_general_hotpotqa
run_dataset_variants musique outputs_step0_general_musique

echo "[$(date '+%F %T')] ALL DONE"
