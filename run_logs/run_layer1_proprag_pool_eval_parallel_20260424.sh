#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
POOL_DIR="${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424"
OUT_DIR="${ROOT_DIR}/run_logs/layer1_proprag_pool_eval_20260424"
SAVE_DIR="outputs_step0_general_nvembed"

mkdir -p "${OUT_DIR}"

run_dataset() {
  local dataset="$1"
  local port="$2"
  local pool_json="${POOL_DIR}/${dataset}_pool100.json"
  local output_json="${OUT_DIR}/${dataset}_proprag_pool_daec_oracle.json"
  local log_path="${OUT_DIR}/${dataset}.log"
  local status_path="${OUT_DIR}/${dataset}.status"

  if [[ ! -s "${pool_json}" ]]; then
    echo "[ERROR] Missing pool JSON: ${pool_json}" > "${status_path}"
    return 2
  fi

  echo "[START] dataset=${dataset} port=${port} start=$(date -Is) pool=${pool_json}" > "${status_path}"
  {
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
      --external_pool_source_name proprag_clean_nothink_top100 \
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
      --output_json "${output_json}"
  } > "${log_path}" 2>&1

  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    echo "[DONE] dataset=${dataset} end=$(date -Is) output=${output_json}" > "${status_path}"
  else
    echo "[FAILED] dataset=${dataset} code=${code} end=$(date -Is) output=${output_json}" > "${status_path}"
  fi
  return "${code}"
}

echo "[START] launcher $(date -Is)" > "${OUT_DIR}/launcher.status"

run_dataset 2wikimultihopqa 8041 &
pid_2wiki=$!
echo "${pid_2wiki}" > "${OUT_DIR}/2wikimultihopqa.pid"

run_dataset hotpotqa 8042 &
pid_hotpot=$!
echo "${pid_hotpot}" > "${OUT_DIR}/hotpotqa.pid"

run_dataset musique 8043 &
pid_musique=$!
echo "${pid_musique}" > "${OUT_DIR}/musique.pid"

echo "[RUNNING] launcher $(date -Is) pids=${pid_2wiki},${pid_hotpot},${pid_musique}" > "${OUT_DIR}/launcher.status"

rc=0
wait "${pid_2wiki}" || rc=1
wait "${pid_hotpot}" || rc=1
wait "${pid_musique}" || rc=1

if [[ "${rc}" -eq 0 ]]; then
  echo "[DONE] launcher $(date -Is)" > "${OUT_DIR}/launcher.status"
else
  echo "[FAILED] launcher $(date -Is)" > "${OUT_DIR}/launcher.status"
fi
exit "${rc}"
