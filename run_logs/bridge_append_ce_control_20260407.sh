#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

STATUS="run_logs/bridge_append_ce_control_20260407.status"
LOG="run_logs/bridge_append_ce_control_20260407.log"
SUMMARY_JSON="run_logs/bridge_append_ce_control_20260407.summary.json"
SUMMARY_MD="run_logs/bridge_append_ce_control_20260407.summary.md"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PY=".venv-hipporag/bin/python"
SCRIPT="scripts/eval_causal_qwen3.py"
LLM_NAME="qwen3-8b"
LLM_REQUEST_NAME="qwen3-8b-train"
LLM_BASE_URL="http://localhost:8043/v1"
EMBED_NAME="VLLM//mnt/nvme/Qwen3-Embedding-8B"
EMBED_BASE_URL="http://localhost:8018/v1/embeddings"
SAVE_DIR="outputs_step0_general"
CONTROL_DATE_TAG="20260407"
EXPAND_DATE_TAG="20260406"
BASELINE_DATE_TAG="20260406"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"
}

run_eval() {
  local dataset="$1"
  local qa_top_k="$2"
  local out_json="outputs_step0_general_${dataset}/eval_reports/bridge_append_cross_encoder_control_topk${qa_top_k}_${CONTROL_DATE_TAG}.json"
  echo "RUNNING ${dataset} control_ce qa_top_k=${qa_top_k}" > "$STATUS"
  log "START dataset=${dataset} control_ce qa_top_k=${qa_top_k}"
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
    --setwise_selector bridge_append \
    --setwise_score_mode bridge \
    --setwise_pool_k 100 \
    --setwise_non_anchor_title_dedup true \
    --expand_base_k "$qa_top_k" \
    --append_max_docs 0 \
    --expand_min_structure_score 0.35 \
    --assemble_mode cross_encoder \
    --structure_relation_probe_mode general_factual \
    --qa_top_k "$qa_top_k" \
    --ce_device cuda:0 \
    --output_json "$out_json" >> "$LOG" 2>&1
  log "DONE dataset=${dataset} control_ce qa_top_k=${qa_top_k} report=${out_json}"
}

build_summary() {
  "$PY" - <<'PY'
import json
from pathlib import Path

control_date_tag = "20260407"
expand_date_tag = "20260406"
baseline_date_tag = "20260406"
datasets = ["musique", "hotpotqa", "2wikimultihopqa"]
qa_top_ks = [5, 7, 10]
rows = []

for dataset in datasets:
    for qa_top_k in qa_top_ks:
        baseline_path = Path(
            f"outputs_step0_general_{dataset}/eval_reports/causal_eval_{dataset}_100_qwen3-8b_qatopk{qa_top_k}_baseline_{baseline_date_tag}.json"
        )
        control_path = Path(
            f"outputs_step0_general_{dataset}/eval_reports/bridge_append_cross_encoder_control_topk{qa_top_k}_{control_date_tag}.json"
        )
        expand_path = Path(
            f"outputs_step0_general_{dataset}/eval_reports/bridge_append_cross_encoder_pool100_base10_append3_qatopk{qa_top_k}_{expand_date_tag}.json"
        )
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            overall = baseline.get("overall_recomputed", {})
            rows.append({
                "dataset": dataset,
                "run_type": "baseline",
                "qa_top_k": qa_top_k,
                "em": overall.get("ExactMatch"),
                "f1": overall.get("F1"),
                "report": str(baseline_path),
            })
        if control_path.exists():
            control = json.loads(control_path.read_text())
            method = control.get("expand_assemble_qa", {})
            rows.append({
                "dataset": dataset,
                "run_type": "baseline_plus_ce",
                "qa_top_k": qa_top_k,
                "em": method.get("method_EM"),
                "f1": method.get("method_F1"),
                "delta_vs_baseline_em": method.get("EM_delta"),
                "delta_vs_baseline_f1": method.get("F1_delta"),
                "report": str(control_path),
            })
        if expand_path.exists():
            expand = json.loads(expand_path.read_text())
            method = expand.get("expand_assemble_qa", {})
            rows.append({
                "dataset": dataset,
                "run_type": "expand_plus_ce",
                "qa_top_k": qa_top_k,
                "em": method.get("method_EM"),
                "f1": method.get("method_F1"),
                "delta_vs_baseline_em": method.get("EM_delta"),
                "delta_vs_baseline_f1": method.get("F1_delta"),
                "report": str(expand_path),
            })

Path("run_logs/bridge_append_ce_control_20260407.summary.json").write_text(
    json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

lines = [
    "# Baseline+CE Control",
    "",
    "| Dataset | Run | qa_top_k | EM | F1 | ΔEM vs baseline |",
    "|---|---|---:|---:|---:|---:|",
]
for row in rows:
    delta = row.get("delta_vs_baseline_em")
    delta_str = "—" if delta is None else f"{delta:.4f}"
    lines.append(
        f"| {row['dataset']} | {row['run_type']} | {row['qa_top_k']} | "
        f"{row['em']:.4f} | {row['f1']:.4f} | {delta_str} |"
    )
Path("run_logs/bridge_append_ce_control_20260407.summary.md").write_text(
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
  for qa_top_k in 5 7 10; do
    run_eval "$dataset" "$qa_top_k"
    build_summary
  done
done

echo "DONE" > "$STATUS"
build_summary
log "Queue finished"
