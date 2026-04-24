#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
POOL_DIR="${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424"
OUT_DIR="${ROOT_DIR}/run_logs/layer1_proprag_pool_ablation_2wiki_20260424"
SAVE_DIR="outputs_step0_general_nvembed"
DATASET="2wikimultihopqa"
PORT="8041"

mkdir -p "${OUT_DIR}"

wait_for_dense_2wiki() {
  local dense_pid_file="${ROOT_DIR}/run_logs/layer1_dense_pool_eval_20260424/launcher_2wiki.pid"
  if [[ ! -s "${dense_pid_file}" ]]; then
    return 0
  fi

  local dense_pid
  dense_pid="$(cat "${dense_pid_file}")"
  if [[ -z "${dense_pid}" ]]; then
    return 0
  fi

  while ps -p "${dense_pid}" > /dev/null 2>&1; do
    echo "[WAIT] dense 2Wiki still running pid=${dense_pid} at $(date -Is)" | tee -a "${OUT_DIR}/launcher.status"
    sleep 60
  done
}

run_variant() {
  local variant="$1"
  shift 1

  local pool_json="${POOL_DIR}/${DATASET}_pool100.json"
  local output_json="${OUT_DIR}/${DATASET}_${variant}.json"
  local log_path="${OUT_DIR}/${DATASET}_${variant}.log"
  local status_path="${OUT_DIR}/${DATASET}_${variant}.status"

  if [[ ! -s "${pool_json}" ]]; then
    echo "[ERROR] Missing pool JSON: ${pool_json}" | tee "${status_path}"
    return 2
  fi

  echo "[START] dataset=${DATASET} variant=${variant} $(date -Is)" | tee "${status_path}"
  (
    cd "${ROOT_DIR}"
    .venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
      --dataset "${DATASET}" --limit 1000 --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b --llm_request_name qwen3-8b-train \
      --max_retry_attempts 20 \
      --llm_base_url "http://localhost:${PORT}/v1" \
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
  ) >> "${log_path}" 2>&1

  echo "[DONE] dataset=${DATASET} variant=${variant} $(date -Is) output=${output_json}" | tee "${status_path}"
}

echo "[START] 2Wiki PropRAG-pool ablation queue $(date -Is)" | tee "${OUT_DIR}/launcher.status"
wait_for_dense_2wiki

# Baseline full method is already available at:
# run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json
# Run only component removals here to avoid duplicating the expensive full variant.

# -binding: disables dependency binding while preserving demand typing and rank prior.
run_variant nobinding \
  --dtc_rank_weight 0.2 \
  --dtc_include_satisfiable_by true \
  --dtc_repairable_filter_enabled true \
  --dtc_satisfiable_by_policy strict \
  --dtc_enable_dependency_binding false

# -repair typing: all requirements can drive coverage; keeps binding and rank prior.
run_variant norepairtyping \
  --dtc_rank_weight 0.2 \
  --dtc_include_satisfiable_by true \
  --dtc_repairable_filter_enabled false \
  --dtc_satisfiable_by_policy binding_override \
  --dtc_enable_dependency_binding true \
  --dtc_binding_max_candidates 4 \
  --dtc_binding_entity_hit_required true

# -rank prior: removes rank regularization; keeps binding and repair typing.
run_variant norank \
  --dtc_rank_weight 0.0 \
  --dtc_include_satisfiable_by true \
  --dtc_repairable_filter_enabled true \
  --dtc_satisfiable_by_policy binding_override \
  --dtc_enable_dependency_binding true \
  --dtc_binding_max_candidates 4 \
  --dtc_binding_entity_hit_required true

echo "[DONE] 2Wiki PropRAG-pool ablation queue $(date -Is)" | tee "${OUT_DIR}/launcher.status"
