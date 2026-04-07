#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

STATUS="run_logs/baseline_qa_topk_sweep_20260406.status"
LOG="run_logs/baseline_qa_topk_sweep_20260406.log"
SUMMARY_JSON="run_logs/baseline_qa_topk_sweep_20260406.summary.json"
SUMMARY_MD="run_logs/baseline_qa_topk_sweep_20260406.summary.md"

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
  local qa_top_k="$2"
  local out_json="outputs_step0_general_${dataset}/eval_reports/causal_eval_${dataset}_100_qwen3-8b_qatopk${qa_top_k}_baseline_${DATE_TAG}.json"
  echo "RUNNING ${dataset} qa_top_k=${qa_top_k}" > "$STATUS"
  log "START dataset=${dataset} qa_top_k=${qa_top_k}"
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
    --qa_top_k "$qa_top_k" \
    --output_json "$out_json" >> "$LOG" 2>&1
  log "DONE dataset=${dataset} qa_top_k=${qa_top_k} report=${out_json}"
}

build_summary() {
  "$PY" - <<'PY'
import json
from pathlib import Path

date_tag = "20260406"
datasets = ["musique", "hotpotqa", "2wikimultihopqa"]
ks = [3, 5, 7, 10]
summary_rows = []
for dataset in datasets:
    for k in ks:
        path = Path(f"outputs_step0_general_{dataset}/eval_reports/causal_eval_{dataset}_100_qwen3-8b_qatopk{k}_baseline_{date_tag}.json")
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        overall = data.get("overall_recomputed", {})
        summary_rows.append({
            "dataset": dataset,
            "qa_top_k": k,
            "em": overall.get("ExactMatch"),
            "f1": overall.get("F1"),
            "recall5": overall.get("Recall@5"),
            "report": str(path),
        })

Path("run_logs/baseline_qa_topk_sweep_20260406.summary.json").write_text(
    json.dumps({"rows": summary_rows}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

lines = ["# Baseline QA Top-K Sweep", "", "| Dataset | qa_top_k | EM | F1 | Recall@5 |", "|---|---:|---:|---:|---:|"]
for row in summary_rows:
    lines.append(
        f"| {row['dataset']} | {row['qa_top_k']} | "
        f"{row['em']:.4f} | {row['f1']:.4f} | {row['recall5']:.4f} |"
    )
Path("run_logs/baseline_qa_topk_sweep_20260406.summary.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
PY
}

trap 'code=$?; log "FAILED exit=${code} while running: $(cat "$STATUS" 2>/dev/null || echo unknown)"; exit $code' ERR

: > "$LOG"
echo "RUNNING boot" > "$STATUS"
log "Queue started"

for dataset in musique hotpotqa 2wikimultihopqa; do
  for qa_top_k in 3 5 7 10; do
    run_eval "$dataset" "$qa_top_k"
  done
done

build_summary

echo "DONE" > "$STATUS"
log "Queue finished"
