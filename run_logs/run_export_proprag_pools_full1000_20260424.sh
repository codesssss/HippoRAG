#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
RUN_DIR="${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424"
mkdir -p "${RUN_DIR}"
cd "${ROOT_DIR}" || exit 1

run_one() {
  local dataset="$1"
  local port="$2"
  local output_json="${RUN_DIR}/${dataset}_pool100.json"
  local log_file="${RUN_DIR}/${dataset}.log"
  local status_file="${RUN_DIR}/${dataset}.status"

  echo "RUNNING dataset=${dataset} start=$(date '+%F %T') output=${output_json}" > "${status_file}"
  echo "[$(date '+%F %T')] START dataset=${dataset} output=${output_json}" > "${log_file}"

  .venv-hipporag/bin/python -u scripts/export_proprag_pool.py \
    --dataset "${dataset}" \
    --limit 1000 \
    --pool_k 100 \
    --llm_name qwen3-8b-train \
    --llm_base_url "http://localhost:${port}/v1" \
    --embedding_name VLLM/nvidia/NV-Embed-v2 \
    --embedding_base_url http://localhost:8019/v1/embeddings \
    --save_dir /mnt/nvme/code/PropRAG/outputs_aligned_clean_nothink_top100_nvembed_rebuild_20260423 \
    --retrieval_top_k 100 \
    --qa_top_k 5 \
    --embedding_batch_size 32 \
    --max_new_tokens 2048 \
    --reuse_preextracted_openie true \
    --openie_llm_name qwen3-8b-train \
    --force_index_from_scratch false \
    --force_openie_from_scratch false \
    --openie_mode online \
    --use_propositions true \
    --use_beam_search true \
    --beam_width 4 \
    --max_path_length 3 \
    --second_stage_filter_k 40 \
    --sim_threshold 0.75 \
    --output_json "${output_json}" >> "${log_file}" 2>&1

  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    echo "DONE dataset=${dataset} end=$(date '+%F %T') output=${output_json}" > "${status_file}"
    echo "[$(date '+%F %T')] DONE dataset=${dataset}" >> "${log_file}"
  else
    echo "FAILED dataset=${dataset} code=${code} end=$(date '+%F %T') output=${output_json}" > "${status_file}"
    echo "[$(date '+%F %T')] FAILED dataset=${dataset} code=${code}" >> "${log_file}"
  fi
}

echo "START sequential PropRAG full1000 pool exports $(date '+%F %T')" > "${RUN_DIR}/launcher.status"
run_one 2wikimultihopqa 8041
run_one hotpotqa 8042
run_one musique 8043
echo "DONE all PropRAG full1000 pool exports $(date '+%F %T')" >> "${RUN_DIR}/launcher.status"
