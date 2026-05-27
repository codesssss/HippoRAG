#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/daec_hc_proprag_full1000_20260503"
SAVE_DIR="outputs_step0_general_nvembed"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
export HIPPORAG_RERANK_FORCE_NO_THINK=1

output_json="${OUT_DIR}/musique_daec_hc_proprag_full1000.json"
log_path="${OUT_DIR}/musique_daec_hc_proprag_full1000.log"
status_path="${OUT_DIR}/musique_daec_hc_proprag_full1000.status"
pool="${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424/musique_pool100.json"

echo "[START] dataset=musique selector=daec_noisyor pool=proprag limit=1000 port=8041 start=$(date -Is)" > "${status_path}"
(
  cd "${ROOT_DIR}" || exit 1
  "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
    --dataset musique \
    --limit 1000 \
    --save_dir "${SAVE_DIR}" \
    --llm_name qwen3-8b \
    --llm_request_name qwen3-8b-train \
    --max_retry_attempts 20 \
    --llm_base_url "http://localhost:8041/v1" \
    --embedding_name VLLM/nvidia/NV-Embed-v2 \
    --embedding_base_url "${EMBEDDING_BASE_URL}" \
    --external_pool_json "${pool}" \
    --external_pool_source_name proprag_pool100 \
    --external_pool_strict_questions true \
    --setwise_selector daec_noisyor \
    --setwise_pool_k 100 \
    --qa_top_k 5 \
    --qa_doc_max_chars 2048 \
    --dtc_decomposition_mode llm \
    --dtc_binding_max_candidates 5 \
    --output_json "${output_json}"
) > "${log_path}" 2>&1
code=$?
if [[ "${code}" -eq 0 ]] && [[ -s "${output_json}" ]]; then
  echo "[DONE] dataset=musique selector=daec_noisyor pool=proprag limit=1000 end=$(date -Is)" > "${status_path}"
else
  echo "[FAILED] dataset=musique selector=daec_noisyor pool=proprag limit=1000 code=${code} end=$(date -Is)" > "${status_path}"
fi
exit "${code}"
