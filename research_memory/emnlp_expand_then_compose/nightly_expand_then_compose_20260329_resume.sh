#!/bin/bash
set -euo pipefail

cd /mnt/nvme/code/HippoRAG
export OPENAI_API_KEY=EMPTY
export PYTHONPATH=.

ROOT="/mnt/nvme/code/HippoRAG/research_memory/emnlp_expand_then_compose"
QUEUE_LOG="$ROOT/nightly_queue_20260329_resume.log"
STATUS="$ROOT/nightly_queue_20260329_resume.status"
mkdir -p "$ROOT"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"
}

run_step() {
  local name="$1"
  local step_log="$2"
  shift 2
  log "START: ${name}"
  echo "RUNNING: ${name}" > "$STATUS"
  "$@" >> "$step_log" 2>&1
  log "DONE: ${name}"
}

trap 'code=$?; log "FAILED (exit=${code}) while running: $(cat "$STATUS" 2>/dev/null || echo unknown)"; exit $code' ERR

echo "RUNNING: boot" > "$STATUS"
log "Resume queue started"

run_step "MuSiQue-1000 oracle sweep + reorder@20" "$ROOT/musique_reorder20_1000_20260329_resume.log" \
  .venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
  --dataset musique \
  --limit 1000 \
  --save_dir outputs_step0_general \
  --max_retry_attempts 12 \
  --oracle_reorder_k 20 \
  --oracle_select_k 20,30,50,100 \
  --output_json outputs_step0_general_musique/eval_reports/oracle_select_sweep_1000.json

run_step "Cross-dataset oracle summary" "$ROOT/cross_dataset_summary_20260329_resume.log" \
  .venv-hipporag/bin/python scripts/compile_oracle_select_summary.py \
  --reports \
  outputs_step0_general_2wikimultihopqa/eval_reports/oracle_select_sweep_1000_reorder20.json \
  outputs_step0_general_hotpotqa/eval_reports/oracle_select_sweep_1000.json \
  outputs_step0_general_musique/eval_reports/oracle_select_sweep_1000.json \
  --output_md research_memory/emnlp_expand_then_compose/oracle_select_cross_dataset_summary_20260329_resume.md \
  --output_json research_memory/emnlp_expand_then_compose/oracle_select_cross_dataset_summary_20260329_resume.json

run_step "HotpotQA-1000 reorder@20 control rerun" "$ROOT/hotpot_reorder20_1000_20260329_resume.log" \
  .venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
  --dataset hotpotqa \
  --limit 1000 \
  --save_dir outputs_step0_general \
  --max_retry_attempts 12 \
  --oracle_reorder_k 20 \
  --oracle_select_k 20,30,50,100 \
  --output_json outputs_step0_general_hotpotqa/eval_reports/oracle_select_sweep_1000_reorder20.json

echo "DONE" > "$STATUS"
log "Resume queue finished"
