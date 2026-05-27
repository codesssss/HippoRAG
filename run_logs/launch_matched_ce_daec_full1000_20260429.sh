#!/usr/bin/env bash
set -u

ROOT="/mnt/nvme/code/HippoRAG"
cd "${ROOT}" || exit 1

RUN_TAG="matched_ce_daec_full1000_20260429"
PY="${PY:-.venv-hipporag/bin/python}"
EVAL_SCRIPT="scripts/eval_causal_qwen3.py"
DIV_SCRIPT="scripts/diversity_selector_study.py"

LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8041/v1}"
LLM_NAME="${LLM_NAME:-qwen3-8b}"
LLM_REQUEST_NAME="${LLM_REQUEST_NAME:-qwen3-8b-train}"
EMBED_NAME="${EMBED_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://localhost:8019/v1/embeddings}"
SAVE_DIR="${SAVE_DIR:-outputs_step0_general_nvembed}"
CE_DEVICE="${CE_DEVICE:-cuda:0}"

OUT_DIR="run_logs/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

STATUS="${OUT_DIR}/status.txt"
LOG="${OUT_DIR}/launcher.log"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

json_done() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  "${PY}" - "${path}" <<'PY' >/dev/null 2>&1
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    json.load(f)
PY
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "${LOG}"
}

run_eval_if_needed() {
  local name="$1"
  shift
  local out_json="$1"
  shift
  if json_done "${out_json}"; then
    log "SKIP ${name}: ${out_json}"
    return 0
  fi
  echo "RUNNING ${name}" > "${STATUS}"
  log "START ${name}"
  "${PY}" "${EVAL_SCRIPT}" "$@" --output_json "${out_json}" >> "${OUT_DIR}/${name}.log" 2>&1
  local code=$?
  if [[ "${code}" -ne 0 ]]; then
    log "FAILED ${name} exit=${code}; see ${OUT_DIR}/${name}.log"
    return "${code}"
  fi
  log "DONE ${name}: ${out_json}"
}

run_diversity_if_needed() {
  local name="$1"
  local out_json="$2"
  shift 2
  if json_done "${out_json}"; then
    log "SKIP ${name}: ${out_json}"
    return 0
  fi
  echo "RUNNING ${name}" > "${STATUS}"
  log "START ${name}"
  "${PY}" "${DIV_SCRIPT}" "$@" --output_json "${out_json}" --resume_from_output >> "${OUT_DIR}/${name}.log" 2>&1
  local code=$?
  if [[ "${code}" -ne 0 ]]; then
    log "FAILED ${name} exit=${code}; see ${OUT_DIR}/${name}.log"
    return "${code}"
  fi
  log "DONE ${name}: ${out_json}"
}

common_eval_args() {
  local dataset="$1"
  local pool_json="$2"
  local pool_source="$3"
  local qa_top_k="$4"
  printf '%s\n' \
    --dataset "${dataset}" \
    --limit 1000 \
    --save_dir "${SAVE_DIR}" \
    --llm_name "${LLM_NAME}" \
    --llm_request_name "${LLM_REQUEST_NAME}" \
    --llm_base_url "${LLM_BASE_URL}" \
    --embedding_name "${EMBED_NAME}" \
    --embedding_base_url "${EMBED_BASE_URL}" \
    --external_pool_json "${pool_json}" \
    --external_pool_source_name "${pool_source}" \
    --external_pool_strict_questions true \
    --max_retry_attempts 20 \
    --qa_top_k "${qa_top_k}"
}

run_ce100() {
  local dataset="$1"
  local pool_json="$2"
  local qa_top_k="$3"
  local out_json="${OUT_DIR}/${dataset}_dense_pool100_ce100_qatopk${qa_top_k}.json"
  mapfile -t args < <(common_eval_args "${dataset}" "${pool_json}" dense_pool100 "${qa_top_k}")
  run_eval_if_needed \
    "${dataset}_ce100_k${qa_top_k}" \
    "${out_json}" \
    "${args[@]}" \
    --setwise_selector none \
    --cross_encoder_rerank true \
    --ce_model /mnt/nvme/bge-reranker-v2-m3 \
    --ce_alpha 0.0 \
    --ce_window 100 \
    --ce_device "${CE_DEVICE}"
}

run_bridge_append_ce() {
  local dataset="$1"
  local pool_json="$2"
  local qa_top_k="$3"
  local out_json="${OUT_DIR}/${dataset}_dense_pool100_bridge_append_ce_qatopk${qa_top_k}.json"
  mapfile -t args < <(common_eval_args "${dataset}" "${pool_json}" dense_pool100 "${qa_top_k}")
  run_eval_if_needed \
    "${dataset}_bridge_append_ce_k${qa_top_k}" \
    "${out_json}" \
    "${args[@]}" \
    --setwise_selector bridge_append \
    --setwise_score_mode bridge \
    --setwise_pool_k 100 \
    --setwise_non_anchor_title_dedup true \
    --expand_base_k 10 \
    --append_max_docs 3 \
    --append_policy bridge \
    --append_random_seed 0 \
    --expand_min_structure_score 0.35 \
    --assemble_mode cross_encoder \
    --structure_relation_probe_mode general_factual \
    --ce_model /mnt/nvme/bge-reranker-v2-m3 \
    --ce_device "${CE_DEVICE}"
}

run_daec_musique_k7() {
  local out_json="${OUT_DIR}/musique_dense_pool100_daec_noisyor_qatopk7.json"
  mapfile -t args < <(common_eval_args musique run_logs/dense_pool_exports_full1000_20260424/musique_dense_pool100.json dense_pool100 7)
  run_eval_if_needed \
    "musique_daec_noisyor_k7" \
    "${out_json}" \
    "${args[@]}" \
    --setwise_selector daec_noisyor \
    --setwise_pool_k 100 \
    --dtc_decomposition_mode llm \
    --dtc_binding_max_candidates 5
}

run_oracle_musique_k7() {
  local out_json="${OUT_DIR}/musique_dense_pool100_oracle100_qatopk7.json"
  mapfile -t args < <(common_eval_args musique run_logs/dense_pool_exports_full1000_20260424/musique_dense_pool100.json dense_pool100 7)
  run_eval_if_needed \
    "musique_oracle100_k7" \
    "${out_json}" \
    "${args[@]}" \
    --setwise_selector none \
    --oracle_select_k 100
}

run_mmr_dpp() {
  local dataset="$1"
  local qa_top_k="$2"
  local source_report="$3"
  local out_json="${OUT_DIR}/${dataset}_dense_pool100_mmr_dpp_qatopk${qa_top_k}.json"
  run_diversity_if_needed \
    "${dataset}_mmr_dpp_k${qa_top_k}" \
    "${out_json}" \
    --dataset "${dataset}" \
    --report_json "${source_report}" \
    --pool_k 100 \
    --qa_top_k "${qa_top_k}" \
    --selectors baseline,mmr,dpp \
    --mmr_lambda 0.5 \
    --dpp_quality_power 1.0 \
    --order_mode original_rank \
    --llm_name "${LLM_NAME}" \
    --llm_request_name "${LLM_REQUEST_NAME}" \
    --llm_base_url "${LLM_BASE_URL}" \
    --embedding_name "${EMBED_NAME}" \
    --embedding_base_url "${EMBED_BASE_URL}" \
    --max_retry_attempts 20
}

log "Matched CE/DAEC full1000 queue started"

run_ce100 2wikimultihopqa run_logs/dense_pool_exports_full1000_20260424/2wikimultihopqa_dense_pool100.json 5 || exit $?
run_ce100 hotpotqa run_logs/dense_pool_exports_full1000_20260424/hotpotqa_dense_pool100.json 5 || exit $?
run_ce100 musique run_logs/dense_pool_exports_full1000_20260424/musique_dense_pool100.json 7 || exit $?

run_bridge_append_ce 2wikimultihopqa run_logs/dense_pool_exports_full1000_20260424/2wikimultihopqa_dense_pool100.json 5 || exit $?
run_bridge_append_ce hotpotqa run_logs/dense_pool_exports_full1000_20260424/hotpotqa_dense_pool100.json 5 || exit $?
run_bridge_append_ce musique run_logs/dense_pool_exports_full1000_20260424/musique_dense_pool100.json 7 || exit $?

run_daec_musique_k7 || exit $?
run_oracle_musique_k7 || exit $?

run_mmr_dpp 2wikimultihopqa 5 run_logs/daec_noisyor_dense_pool100_2wiki_fixclean_full1000_20260428.json || exit $?
run_mmr_dpp hotpotqa 5 run_logs/daec_noisyor_dense_pool100_hotpotqa_fixclean_full1000_20260428.json || exit $?
run_mmr_dpp musique 7 "${OUT_DIR}/musique_dense_pool100_ce100_qatopk7.json" || exit $?

echo "DONE" > "${STATUS}"
log "Matched CE/DAEC full1000 queue finished"
