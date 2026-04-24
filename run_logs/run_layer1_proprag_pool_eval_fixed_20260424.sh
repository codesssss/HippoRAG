#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
POOL_DIR="${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424"
OUT_DIR="${ROOT_DIR}/run_logs/layer1_proprag_pool_eval_fixed_20260424"
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

dataset_to_port() {
  case "$1" in
    2wikimultihopqa) echo 8041 ;;
    hotpotqa) echo 8042 ;;
    musique) echo 8043 ;;
    *) echo "[ERROR] Unknown dataset: $1" >&2; return 2 ;;
  esac
}

if [[ "$#" -gt 0 ]]; then
  datasets=("$@")
else
  datasets=(2wikimultihopqa hotpotqa musique)
fi

echo "[START] launcher $(date -Is) datasets=${datasets[*]}" > "${OUT_DIR}/launcher.status"

rc=0
pids=()
for dataset in "${datasets[@]}"; do
  port="$(dataset_to_port "${dataset}")"
  run_dataset "${dataset}" "${port}" &
  pid=$!
  pids+=("${pid}")
  echo "${pid}" > "${OUT_DIR}/${dataset}.pid"
done

echo "[RUNNING] launcher $(date -Is) pids=${pids[*]}" > "${OUT_DIR}/launcher.status"

for pid in "${pids[@]}"; do
  wait "${pid}" || rc=1
done

if [[ "${rc}" -eq 0 ]]; then
  echo "[DONE] launcher $(date -Is)" > "${OUT_DIR}/launcher.status"
else
  echo "[FAILED] launcher $(date -Is)" > "${OUT_DIR}/launcher.status"
fi
exit "${rc}"
