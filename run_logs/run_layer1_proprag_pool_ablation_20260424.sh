#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
POOL_DIR="${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424"
OUT_DIR="${ROOT_DIR}/run_logs/layer1_proprag_pool_ablation_20260424"
SAVE_DIR="outputs_step0_general_nvembed"

mkdir -p "${OUT_DIR}"

run_variant() {
  local dataset="$1"
  local port="$2"
  local variant="$3"
  shift 3

  local pool_json="${POOL_DIR}/${dataset}_pool100.json"
  local output_json="${OUT_DIR}/${dataset}_${variant}.json"
  local log_path="${OUT_DIR}/${dataset}_${variant}.log"

  if [[ ! -s "${pool_json}" ]]; then
    echo "[ERROR] Missing pool JSON: ${pool_json}" | tee "${OUT_DIR}/${dataset}_${variant}.status"
    return 2
  fi

  echo "[START] dataset=${dataset} variant=${variant} $(date -Is)" | tee "${OUT_DIR}/${dataset}_${variant}.status"
  (
    cd "${ROOT_DIR}"
    .venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" --limit 1000 --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b --llm_request_name qwen3-8b-train \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name VLLM/nvidia/NV-Embed-v2 \
      --embedding_base_url http://localhost:8019/v1/embeddings \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --external_pool_json "${pool_json}" \
      --external_pool_source_name proprag_clean_nothink_top100 \
      --setwise_selector dtc_embed \
      --setwise_pool_k 100 \
      --setwise_reserve_top_m 0 \
      "$@" \
      --output_json "${output_json}"
  ) 2>&1 | tee "${log_path}"
  echo "[DONE] dataset=${dataset} variant=${variant} $(date -Is) output=${output_json}" | tee "${OUT_DIR}/${dataset}_${variant}.status"
}

run_dataset() {
  local dataset="$1"
  local port="$2"

  # Full current DAEC/DtC instantiation.
  run_variant "${dataset}" "${port}" full \
    --dtc_rank_weight 0.2 \
    --dtc_include_satisfiable_by true \
    --dtc_repairable_filter_enabled true \
    --dtc_satisfiable_by_policy binding_override \
    --dtc_enable_dependency_binding true \
    --dtc_binding_max_candidates 4 \
    --dtc_binding_entity_hit_required true

  # -binding: disables dependency binding while preserving demand typing and rank prior.
  run_variant "${dataset}" "${port}" nobinding \
    --dtc_rank_weight 0.2 \
    --dtc_include_satisfiable_by true \
    --dtc_repairable_filter_enabled true \
    --dtc_satisfiable_by_policy strict \
    --dtc_enable_dependency_binding false

  # -repair typing: all requirements can drive coverage; keeps binding and rank prior.
  run_variant "${dataset}" "${port}" norepairtyping \
    --dtc_rank_weight 0.2 \
    --dtc_include_satisfiable_by true \
    --dtc_repairable_filter_enabled false \
    --dtc_satisfiable_by_policy binding_override \
    --dtc_enable_dependency_binding true \
    --dtc_binding_max_candidates 4 \
    --dtc_binding_entity_hit_required true

  # -rank prior: removes rank regularization; keeps binding and repair typing.
  run_variant "${dataset}" "${port}" norank \
    --dtc_rank_weight 0.0 \
    --dtc_include_satisfiable_by true \
    --dtc_repairable_filter_enabled true \
    --dtc_satisfiable_by_policy binding_override \
    --dtc_enable_dependency_binding true \
    --dtc_binding_max_candidates 4 \
    --dtc_binding_entity_hit_required true
}

run_dataset 2wikimultihopqa 8041
run_dataset hotpotqa 8042
run_dataset musique 8043
