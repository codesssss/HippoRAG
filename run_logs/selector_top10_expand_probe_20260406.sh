#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

STATUS="run_logs/selector_top10_expand_probe_20260406.status"
LOG="run_logs/selector_top10_expand_probe_20260406.log"
SUMMARY_JSON="run_logs/selector_top10_expand_probe_20260406.summary.json"
SUMMARY_MD="run_logs/selector_top10_expand_probe_20260406.summary.md"
PID_FILE="run_logs/selector_top10_expand_probe_20260406.pid"

PY=".venv-hipporag/bin/python"
SCRIPT="scripts/eval_causal_qwen3.py"
LLM_NAME="qwen3-8b"
LLM_REQUEST_NAME="qwen3-8b-train"
LLM_BASE_URL="http://localhost:8043/v1"
EMBED_NAME="VLLM//mnt/nvme/Qwen3-Embedding-8B"
EMBED_BASE_URL="http://localhost:8018/v1/embeddings"
SAVE_DIR="outputs_step0_general"
DATE_TAG="20260406"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"
}

run_eval() {
  local dataset="$1"
  local out_json="outputs_step0_general_${dataset}/eval_reports/setwise_bridge_beam_bridge_100_legacy_reserve3_bridge7_nogate_generalfactual_qatopk10_qwen3-8b_${DATE_TAG}.json"
  echo "RUNNING ${dataset} selector@10" > "$STATUS"
  log "START dataset=${dataset} selector@10"
  "$PY" "$SCRIPT" \
    --dataset "$dataset" \
    --limit 100 \
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
    --setwise_selector bridge_beam \
    --setwise_score_mode bridge \
    --setwise_pool_k 100 \
    --setwise_anchor_count 2 \
    --setwise_reserve_top_m 3 \
    --setwise_max_bridge_slots 7 \
    --setwise_non_anchor_title_dedup true \
    --setwise_beam_width 4 \
    --setwise_beam_expand_per_state 4 \
    --setwise_beam_projected_shortlist_factor 1 \
    --setwise_gate_mode none \
    --structure_relation_probe_mode general_factual \
    --qa_top_k 10 \
    --output_json "$out_json" >> "$LOG" 2>&1
  log "DONE dataset=${dataset} selector@10 report=${out_json}"
}

build_summary() {
  "$PY" - <<'PY'
import json
from pathlib import Path

date_tag = "20260406"
datasets = ["musique", "hotpotqa", "2wikimultihopqa"]
rows = []
for dataset in datasets:
    baseline_path = Path(
        f"outputs_step0_general_{dataset}/eval_reports/causal_eval_{dataset}_100_qwen3-8b_qatopk10_baseline_{date_tag}.json"
    )
    selector_path = Path(
        f"outputs_step0_general_{dataset}/eval_reports/setwise_bridge_beam_bridge_100_legacy_reserve3_bridge7_nogate_generalfactual_qatopk10_qwen3-8b_{date_tag}.json"
    )
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        overall = baseline.get("overall_recomputed", {})
        rows.append({
            "dataset": dataset,
            "run_type": "baseline@10",
            "em": overall.get("ExactMatch"),
            "f1": overall.get("F1"),
            "recall5": overall.get("Recall@5"),
            "report": str(baseline_path),
        })
    if selector_path.exists():
        selector = json.loads(selector_path.read_text())
        overall = selector.get("overall_recomputed", {})
        selector_qa = selector.get("setwise_selector_qa", {})
        rows.append({
            "dataset": dataset,
            "run_type": "selector@10",
            "em": selector_qa.get("selector_EM"),
            "f1": selector_qa.get("selector_F1"),
            "recall5": ((selector_qa.get("selector_retrieval_metrics") or {}).get("Recall@5")),
            "top_level_em": overall.get("ExactMatch"),
            "top_level_f1": overall.get("F1"),
            "selector_em": selector_qa.get("selector_EM"),
            "selector_f1": selector_qa.get("selector_F1"),
            "baseline_em_internal": selector_qa.get("baseline_EM"),
            "baseline_f1_internal": selector_qa.get("baseline_F1"),
            "report": str(selector_path),
        })

Path("run_logs/selector_top10_expand_probe_20260406.summary.json").write_text(
    json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

lines = [
    "# Selector Top-10 Expand Probe",
    "",
    "| Dataset | Run | EM | F1 | Recall@5 |",
    "|---|---|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['dataset']} | {row['run_type']} | "
        f"{row['em']:.4f} | {row['f1']:.4f} | {row['recall5']:.4f} |"
    )
Path("run_logs/selector_top10_expand_probe_20260406.summary.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
PY
}

trap 'code=$?; log "FAILED exit=${code} while running: $(cat "$STATUS" 2>/dev/null || echo unknown)"; exit $code' ERR

printf '%s\n' "$$" > "$PID_FILE"
: > "$LOG"
echo "RUNNING boot" > "$STATUS"
log "Queue started"

for dataset in musique hotpotqa 2wikimultihopqa; do
  run_eval "$dataset"
  build_summary
done

echo "DONE" > "$STATUS"
log "Queue finished"
