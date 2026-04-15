#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

DATE_TAG="${DATE_TAG:-20260410bridge_diag}"
CONTROL_DATE_TAG="${CONTROL_DATE_TAG:-20260410nqpopqa100dual}"
LIMIT="${LIMIT:-100}"
STATUS="run_logs/diagnose_bridge_failure_nq_popqa_100_${DATE_TAG}.status"
LOG="run_logs/diagnose_bridge_failure_nq_popqa_100_${DATE_TAG}.log"
PID_FILE="run_logs/diagnose_bridge_failure_nq_popqa_100_${DATE_TAG}.pid"
SUMMARY_PY="run_logs/build_bridge_failure_diagnosis_nq_popqa_100_summary.py"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local-placeholder}"

PY=".venv-hipporag/bin/python"
SCRIPT="scripts/eval_causal_qwen3.py"
LLM_NAME="${LLM_NAME:-qwen3-8b}"
EMBED_NAME="${EMBED_NAME:-VLLM//mnt/nvme/Qwen3-Embedding-8B}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://localhost:8018/v1/embeddings}"
SAVE_DIR="${SAVE_DIR:-outputs_step0_general}"
QA_TOP_K="${QA_TOP_K:-5}"
CE_DEVICE="${CE_DEVICE:-cpu}"
NQ_LLM_REQUEST_NAME="${NQ_LLM_REQUEST_NAME:-qwen3-8b-train}"
NQ_LLM_BASE_URL="${NQ_LLM_BASE_URL:-http://localhost:8043/v1}"
POPQA_LLM_REQUEST_NAME="${POPQA_LLM_REQUEST_NAME:-qwen3-8b}"
POPQA_LLM_BASE_URL="${POPQA_LLM_BASE_URL:-http://36.133.236.142:8002/v1}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"
}

build_summary() {
  "$PY" "$SUMMARY_PY" --diag-date-tag "$DATE_TAG" --control-date-tag "$CONTROL_DATE_TAG" --limit "$LIMIT"
}

run_bridge_diag() {
  local dataset="$1"
  local run_name="$2"
  local llm_request_name="$3"
  local llm_base_url="$4"
  local query_entity_source="$5"
  local threshold="$6"
  local -a launch_prefix=()

  local out_json="outputs_step0_general_${dataset}/eval_reports/${run_name}_${DATE_TAG}.json"
  local baseline_report="outputs_step0_general_${dataset}/eval_reports/causal_eval_${dataset}_${LIMIT}_qwen3-8b_qatopk${QA_TOP_K}_baseline_${CONTROL_DATE_TAG}.json"
  local retrieval_cache="outputs_step0_general_${dataset}/eval_reports/causal_eval_${dataset}_${LIMIT}_qwen3-8b_qatopk${QA_TOP_K}_baseline_${CONTROL_DATE_TAG}.retrieval_cache.json"

  if [[ -f "$out_json" ]]; then
    log "SKIP existing dataset=${dataset} run=${run_name} report=${out_json}"
    return 0
  fi

  echo "RUNNING dataset=${dataset} run=${run_name}" > "$STATUS"
  log "START dataset=${dataset} run=${run_name} q_source=${query_entity_source} threshold=${threshold} ce_device=${CE_DEVICE} llm=${llm_base_url}"
  if [[ "$CE_DEVICE" == "cpu" ]]; then
    launch_prefix=(env CUDA_VISIBLE_DEVICES=)
  fi
  "${launch_prefix[@]}" "$PY" "$SCRIPT" \
    --dataset "$dataset" \
    --limit "$LIMIT" \
    --save_dir "$SAVE_DIR" \
    --max_retry_attempts 12 \
    --llm_name "$LLM_NAME" \
    --llm_request_name "$llm_request_name" \
    --llm_base_url "$llm_base_url" \
    --embedding_name "$EMBED_NAME" \
    --embedding_base_url "$EMBED_BASE_URL" \
    --openie_mode online \
    --causal_enabled false \
    --causal_engine_version v2 \
    --causal_v2_graph_mode causal \
    --causal_v2_base_retrieval_mode legacy_fact_graph \
    --baseline_report_json "$baseline_report" \
    --retrieval_cache_json "$retrieval_cache" \
    --setwise_selector bridge_append \
    --setwise_score_mode bridge \
    --setwise_pool_k 100 \
    --setwise_non_anchor_title_dedup true \
    --setwise_query_entity_source "$query_entity_source" \
    --expand_base_k 10 \
    --append_max_docs 3 \
    --append_policy bridge \
    --append_random_seed 0 \
    --expand_min_structure_score "$threshold" \
    --assemble_mode cross_encoder \
    --structure_relation_probe_mode general_factual \
    --qa_top_k "$QA_TOP_K" \
    --ce_device "$CE_DEVICE" \
    --output_json "$out_json" >> "$LOG" 2>&1
  log "DONE dataset=${dataset} run=${run_name} report=${out_json}"
}

trap 'code=$?; log "FAILED exit=${code} while running: $(cat "$STATUS" 2>/dev/null || echo unknown)"; build_summary || true; exit $code' ERR

: > "$LOG"
printf '%s\n' "$$" > "$PID_FILE"
echo "RUNNING boot" > "$STATUS"
log "Bridge diagnosis started date_tag=${DATE_TAG} control_date_tag=${CONTROL_DATE_TAG} ce_device=${CE_DEVICE}"

run_bridge_diag "nq" "bridge_diag_nq_seed_thr000" "$NQ_LLM_REQUEST_NAME" "$NQ_LLM_BASE_URL" "seed" "0.0"
build_summary
run_bridge_diag "nq" "bridge_diag_nq_question_thr035" "$NQ_LLM_REQUEST_NAME" "$NQ_LLM_BASE_URL" "question" "0.35"
build_summary
run_bridge_diag "nq" "bridge_diag_nq_question_thr000" "$NQ_LLM_REQUEST_NAME" "$NQ_LLM_BASE_URL" "question" "0.0"
build_summary

run_bridge_diag "popqa" "bridge_diag_popqa_seed_thr000" "$POPQA_LLM_REQUEST_NAME" "$POPQA_LLM_BASE_URL" "seed" "0.0"
build_summary
run_bridge_diag "popqa" "bridge_diag_popqa_question_thr035" "$POPQA_LLM_REQUEST_NAME" "$POPQA_LLM_BASE_URL" "question" "0.35"
build_summary
run_bridge_diag "popqa" "bridge_diag_popqa_question_thr000" "$POPQA_LLM_REQUEST_NAME" "$POPQA_LLM_BASE_URL" "question" "0.0"
build_summary

echo "DONE" > "$STATUS"
build_summary
log "Bridge diagnosis finished"
