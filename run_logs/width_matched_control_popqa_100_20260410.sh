#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

DATASET="popqa"
DATE_TAG="${DATE_TAG:-20260410nqpopqa100dual}"
LIMIT="${LIMIT:-100}"
STATUS="run_logs/width_matched_control_${DATASET}_100_${DATE_TAG}.status"
LOG="run_logs/width_matched_control_${DATASET}_100_${DATE_TAG}.log"
PID_FILE="run_logs/width_matched_control_${DATASET}_100_${DATE_TAG}.pid"
SUMMARY_PY="run_logs/build_width_matched_control_nq_popqa_100_summary.py"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local-placeholder}"

PY=".venv-hipporag/bin/python"
SCRIPT="scripts/eval_causal_qwen3.py"
LLM_NAME="${LLM_NAME:-qwen3-8b}"
LLM_REQUEST_NAME="${LLM_REQUEST_NAME:-qwen3-8b}"
LLM_BASE_URL="${LLM_BASE_URL:-http://36.133.236.142:8002/v1}"
EMBED_NAME="${EMBED_NAME:-VLLM//mnt/nvme/Qwen3-Embedding-8B}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://localhost:8018/v1/embeddings}"
SAVE_DIR="${SAVE_DIR:-outputs_step0_general}"
QA_TOP_K="${QA_TOP_K:-5}"
CE_DEVICE="${CE_DEVICE:-cuda:0}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"
}

build_summary() {
  "$PY" "$SUMMARY_PY" --date-tag "$DATE_TAG" --limit "$LIMIT"
}

run_baseline() {
  local out_json="outputs_step0_general_${DATASET}/eval_reports/causal_eval_${DATASET}_${LIMIT}_qwen3-8b_qatopk${QA_TOP_K}_baseline_${DATE_TAG}.json"
  local retrieval_cache="outputs_step0_general_${DATASET}/eval_reports/causal_eval_${DATASET}_${LIMIT}_qwen3-8b_qatopk${QA_TOP_K}_baseline_${DATE_TAG}.retrieval_cache.json"

  if [[ -f "$out_json" ]]; then
    log "SKIP existing dataset=${DATASET} run=baseline_top${QA_TOP_K} report=${out_json}"
    return 0
  fi

  echo "RUNNING dataset=${DATASET} run=baseline_top${QA_TOP_K}" > "$STATUS"
  log "START dataset=${DATASET} run=baseline_top${QA_TOP_K} llm=${LLM_BASE_URL} request_name=${LLM_REQUEST_NAME}"
  "$PY" "$SCRIPT" \
    --dataset "$DATASET" \
    --limit "$LIMIT" \
    --save_dir "$SAVE_DIR" \
    --max_retry_attempts 12 \
    --llm_name "$LLM_NAME" \
    --llm_request_name "$LLM_REQUEST_NAME" \
    --llm_base_url "$LLM_BASE_URL" \
    --embedding_name "$EMBED_NAME" \
    --embedding_base_url "$EMBED_BASE_URL" \
    --openie_mode online \
    --causal_enabled false \
    --causal_engine_version v2 \
    --causal_v2_graph_mode causal \
    --causal_v2_base_retrieval_mode legacy_fact_graph \
    --save_retrieval_cache_json "$retrieval_cache" \
    --qa_top_k "$QA_TOP_K" \
    --output_json "$out_json" >> "$LOG" 2>&1
  log "DONE dataset=${DATASET} run=baseline_top${QA_TOP_K} report=${out_json}"
}

run_width_matched_ce() {
  local run_name="$1"
  local append_policy="$2"
  local append_max_docs="$3"
  local append_random_seed="$4"
  local out_json="outputs_step0_general_${DATASET}/eval_reports/${run_name}_${DATE_TAG}.json"
  local baseline_report="outputs_step0_general_${DATASET}/eval_reports/causal_eval_${DATASET}_${LIMIT}_qwen3-8b_qatopk${QA_TOP_K}_baseline_${DATE_TAG}.json"
  local retrieval_cache="outputs_step0_general_${DATASET}/eval_reports/causal_eval_${DATASET}_${LIMIT}_qwen3-8b_qatopk${QA_TOP_K}_baseline_${DATE_TAG}.retrieval_cache.json"
  local extra_args=()

  if [[ -f "$out_json" ]]; then
    log "SKIP existing dataset=${DATASET} run=${run_name} report=${out_json}"
    return 0
  fi

  if [[ -f "$retrieval_cache" ]]; then
    extra_args+=(--retrieval_cache_json "$retrieval_cache")
  fi

  echo "RUNNING dataset=${DATASET} run=${run_name}" > "$STATUS"
  log "START dataset=${DATASET} run=${run_name} append_policy=${append_policy} append_max_docs=${append_max_docs} seed=${append_random_seed} ce_device=${CE_DEVICE}"
  "$PY" "$SCRIPT" \
    --dataset "$DATASET" \
    --limit "$LIMIT" \
    --save_dir "$SAVE_DIR" \
    --max_retry_attempts 12 \
    --llm_name "$LLM_NAME" \
    --llm_request_name "$LLM_REQUEST_NAME" \
    --llm_base_url "$LLM_BASE_URL" \
    --embedding_name "$EMBED_NAME" \
    --embedding_base_url "$EMBED_BASE_URL" \
    --openie_mode online \
    --causal_enabled false \
    --causal_engine_version v2 \
    --causal_v2_graph_mode causal \
    --causal_v2_base_retrieval_mode legacy_fact_graph \
    --baseline_report_json "$baseline_report" \
    "${extra_args[@]}" \
    --setwise_selector bridge_append \
    --setwise_score_mode bridge \
    --setwise_pool_k 100 \
    --setwise_non_anchor_title_dedup true \
    --expand_base_k 10 \
    --append_max_docs "$append_max_docs" \
    --append_policy "$append_policy" \
    --append_random_seed "$append_random_seed" \
    --expand_min_structure_score 0.35 \
    --assemble_mode cross_encoder \
    --structure_relation_probe_mode general_factual \
    --qa_top_k "$QA_TOP_K" \
    --ce_device "$CE_DEVICE" \
    --output_json "$out_json" >> "$LOG" 2>&1
  log "DONE dataset=${DATASET} run=${run_name} report=${out_json}"
}

trap 'code=$?; log "FAILED exit=${code} while running: $(cat "$STATUS" 2>/dev/null || echo unknown)"; build_summary || true; exit $code' ERR

: > "$LOG"
printf '%s\n' "$$" > "$PID_FILE"
echo "RUNNING boot" > "$STATUS"
log "Queue started dataset=${DATASET} limit=${LIMIT} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

run_baseline
build_summary
run_width_matched_ce "width_match_baseline_top10_plus_ce_qatopk5" "bridge" 0 0
build_summary
run_width_matched_ce "width_match_random3_deep_plus_ce_qatopk5" "random_deep" 3 0
build_summary
run_width_matched_ce "width_match_bridge_append_plus_ce_qatopk5" "bridge" 3 0
build_summary

echo "DONE" > "$STATUS"
build_summary
log "Queue finished dataset=${DATASET}"
