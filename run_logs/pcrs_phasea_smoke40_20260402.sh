#!/bin/bash
set -euo pipefail

cd /mnt/nvme/code/HippoRAG
export OPENAI_API_KEY=EMPTY
export PYTHONPATH=.

ROOT="/mnt/nvme/code/HippoRAG/run_logs"
QUEUE_LOG="$ROOT/pcrs_phasea_smoke40_20260402.log"
STATUS="$ROOT/pcrs_phasea_smoke40_20260402.status"
SUMMARY_JSON="$ROOT/pcrs_phasea_smoke40_20260402_summary.json"

mkdir -p "$ROOT"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"
}

run_step() {
  local step_name="$1"
  shift
  echo "RUNNING: ${step_name}" > "$STATUS"
  log "START ${step_name}"
  "$@" >> "$QUEUE_LOG" 2>&1
  log "DONE ${step_name}"
}

append_summary() {
  local dataset="$1"
  local cache_path="$2"
  local output_json="$3"
  .venv-hipporag/bin/python - <<PY
import json
from pathlib import Path

summary_path = Path(${SUMMARY_JSON@Q})
report_path = Path(${output_json@Q})
cache_path = Path(${cache_path@Q})
dataset = ${dataset@Q}

summary = {"runs": []}
if summary_path.exists():
    summary = json.loads(summary_path.read_text())

report = json.loads(report_path.read_text())
selector = report.get("setwise_selector_qa", {}) or {}

summary["runs"].append({
    "dataset": dataset,
    "cache_path": str(cache_path),
    "output_json": str(report_path),
    "baseline_EM": (((report.get("baseline_qa") or {}).get("overall") or {}).get("em")),
    "baseline_F1": (((report.get("baseline_qa") or {}).get("overall") or {}).get("f1")),
    "selector_EM": ((selector.get("selector_qa") or {}).get("overall") or {}).get("em"),
    "selector_F1": ((selector.get("selector_qa") or {}).get("overall") or {}).get("f1"),
    "avg_support_completeness": selector.get("avg_requirement_support_completeness"),
    "avg_counterfactual_leakage": selector.get("avg_requirement_counterfactual_leakage"),
    "avg_frontier_size": selector.get("avg_requirement_frontier_size"),
    "requirement_mode": selector.get("requirement_mode"),
    "requirement_cache_hit_count": selector.get("requirement_cache_hit_count"),
})
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

run_dataset() {
  local dataset="$1"
  local cache_path="research_memory/emnlp_expand_then_compose/models/${dataset}_requirement_cache_pool100_ann50_limit40.json"
  local report_dir="outputs_step0_general_${dataset}/eval_reports"
  local output_json="${report_dir}/requirement_beam_oracle_smoke40_v2_legacyfact_pool100_ann50_reserve3_dedup.json"

  mkdir -p "${report_dir}"

  if [[ -f "${cache_path}" ]]; then
    log "SKIP ${dataset} build_requirement_cache smoke40 (cache exists: ${cache_path})"
  else
    run_step "${dataset} build_requirement_cache smoke40" \
      .venv-hipporag/bin/python scripts/build_requirement_cache.py \
        --dataset "${dataset}" \
        --limit 40 \
        --save_dir outputs_step0_general \
        --output_path "${cache_path}" \
        --setwise_pool_k 100 \
        --annotation_pool_k 50 \
        --qa_top_k 5 \
        --max_retry_attempts 12
  fi

  run_step "${dataset} requirement_beam oracle smoke40" \
    .venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit 40 \
      --save_dir outputs_step0_general \
      --max_retry_attempts 12 \
      --causal_enabled false \
      --causal_engine_version v2 \
      --causal_v2_base_retrieval_mode legacy_fact_graph \
      --setwise_selector requirement_beam \
      --setwise_requirement_cache_path "${cache_path}" \
      --setwise_requirement_mode oracle \
      --setwise_pool_k 100 \
      --setwise_anchor_count 2 \
      --setwise_reserve_top_m 3 \
      --setwise_non_anchor_title_dedup true \
      --setwise_beam_width 4 \
      --setwise_beam_expand_per_state 4 \
      --setwise_beam_projected_shortlist_factor 1 \
      --output_json "${output_json}"

  append_summary "${dataset}" "${cache_path}" "${output_json}"
}

: > "$QUEUE_LOG"
echo "QUEUED" > "$STATUS"
echo '{"runs":[]}' > "$SUMMARY_JSON"

log "Queue start"
run_dataset "2wikimultihopqa"
run_dataset "musique"
echo "DONE" > "$STATUS"
log "Queue finished"
