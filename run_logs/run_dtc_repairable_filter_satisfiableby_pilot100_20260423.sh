#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
RUN_DIR="${ROOT_DIR}/run_logs/dtc_repairable_filter_satisfiableby_pilot100_20260423"
mkdir -p "${RUN_DIR}"
cd "${ROOT_DIR}" || exit 1

run_one() {
  local dataset="$1"
  local port="$2"
  local output_json="outputs_step0_general_nvembed_${dataset}/eval_reports/dtc_embed_nvembed_rankw0p2_repairable_filter_satisfiableby_pilot100_anchor2_${port}.json"
  local log_file="${RUN_DIR}/${dataset}.log"
  local status_file="${RUN_DIR}/${dataset}.status"

  echo "RUNNING dataset=${dataset} port=${port} start=$(date '+%F %T') output=${output_json}" > "${status_file}"
  echo "[$(date '+%F %T')] START ${dataset} -> ${output_json}" >> "${log_file}"

  .venv-hipporag/bin/python -u scripts/eval_causal_qwen3.py \
    --dataset "${dataset}" \
    --limit 100 \
    --save_dir outputs_step0_general_nvembed \
    --llm_name qwen3-8b \
    --llm_request_name qwen3-8b-train \
    --llm_base_url "http://localhost:${port}/v1" \
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
    --dtc_repairable_filter_enabled true \
    --dtc_enable_dependency_binding true \
    --dtc_binding_max_candidates 4 \
    --dtc_binding_entity_hit_required true \
    --dtc_max_steps 4 \
    --dtc_match_threshold 0.35 \
    --dtc_redundancy_weight 0.10 \
    --dtc_base_weight 0.05 \
    --dtc_rank_weight 0.2 \
    --dtc_anchor_bonus_weight 0.10 \
    --dtc_dependency_bonus_weight 0.10 \
    --dtc_max_completion_tokens 512 \
    --dtc_ser_enabled false \
    --output_json "${output_json}" >> "${log_file}" 2>&1

  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    echo "DONE dataset=${dataset} port=${port} end=$(date '+%F %T') output=${output_json}" > "${status_file}"
    echo "[$(date '+%F %T')] DONE ${dataset}" >> "${log_file}"
  else
    echo "FAILED dataset=${dataset} port=${port} code=${code} end=$(date '+%F %T') output=${output_json}" > "${status_file}"
    echo "[$(date '+%F %T')] FAILED ${dataset} code=${code}" >> "${log_file}"
  fi
}

run_one 2wikimultihopqa 8041 &
pid_2wiki=$!
run_one hotpotqa 8042 &
pid_hotpot=$!
run_one musique 8043 &
pid_musique=$!

echo "Launched satisfiable_by pilot100 jobs: 2wiki=${pid_2wiki}, hotpot=${pid_hotpot}, musique=${pid_musique}" > "${RUN_DIR}/launcher.status"
wait "${pid_2wiki}" "${pid_hotpot}" "${pid_musique}"
