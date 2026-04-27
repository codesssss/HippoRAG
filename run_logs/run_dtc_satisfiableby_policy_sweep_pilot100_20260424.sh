#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
RUN_DIR="${ROOT_DIR}/run_logs/dtc_satisfiableby_policy_sweep_pilot100_20260424"
mkdir -p "${RUN_DIR}"
cd "${ROOT_DIR}" || exit 1

run_one() {
  local policy="$1"
  local dataset="$2"
  local port="$3"
  local output_json="outputs_step0_general_nvembed_${dataset}/eval_reports/dtc_embed_nvembed_rankw0p2_repairable_filter_satisfiableby_${policy}_pilot100_anchor2_${port}.json"
  local log_file="${RUN_DIR}/${policy}_${dataset}.log"
  local status_file="${RUN_DIR}/${policy}_${dataset}.status"

  echo "RUNNING policy=${policy} dataset=${dataset} port=${port} start=$(date '+%F %T') output=${output_json}" > "${status_file}"
  echo "[$(date '+%F %T')] START policy=${policy} dataset=${dataset} -> ${output_json}" >> "${log_file}"

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
    --dtc_include_satisfiable_by true \
    --dtc_enforce_dependencies true \
    --dtc_require_new_crossing false \
    --dtc_repairable_filter_enabled true \
    --dtc_satisfiable_by_policy "${policy}" \
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
    echo "DONE policy=${policy} dataset=${dataset} port=${port} end=$(date '+%F %T') output=${output_json}" > "${status_file}"
    echo "[$(date '+%F %T')] DONE policy=${policy} dataset=${dataset}" >> "${log_file}"
  else
    echo "FAILED policy=${policy} dataset=${dataset} port=${port} code=${code} end=$(date '+%F %T') output=${output_json}" > "${status_file}"
    echo "[$(date '+%F %T')] FAILED policy=${policy} dataset=${dataset} code=${code}" >> "${log_file}"
  fi
}

run_policy() {
  local policy="$1"
  run_one "${policy}" 2wikimultihopqa 8041 &
  local pid_2wiki=$!
  run_one "${policy}" hotpotqa 8042 &
  local pid_hotpot=$!
  run_one "${policy}" musique 8043 &
  local pid_musique=$!
  echo "Launched policy=${policy}: 2wiki=${pid_2wiki}, hotpot=${pid_hotpot}, musique=${pid_musique}" >> "${RUN_DIR}/launcher.status"
  wait "${pid_2wiki}" "${pid_hotpot}" "${pid_musique}"
}

echo "START policy sweep $(date '+%F %T')" > "${RUN_DIR}/launcher.status"
run_policy grounded_override
run_policy regex_only
echo "DONE policy sweep $(date '+%F %T')" >> "${RUN_DIR}/launcher.status"
