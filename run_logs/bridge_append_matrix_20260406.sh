#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

STATUS="run_logs/bridge_append_matrix_20260406.status"
LOG="run_logs/bridge_append_matrix_20260406.log"
SUMMARY_JSON="run_logs/bridge_append_matrix_20260406.summary.json"
SUMMARY_MD="run_logs/bridge_append_matrix_20260406.summary.md"
PID_FILE="run_logs/bridge_append_matrix_20260406.pid"

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
DATE_TAG="20260406"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"
}

run_eval() {
  local dataset="$1"
  local assemble_mode="$2"
  local qa_top_k="$3"
  local out_json="outputs_step0_general_${dataset}/eval_reports/bridge_append_${assemble_mode}_pool100_base10_append3_qatopk${qa_top_k}_${DATE_TAG}.json"
  echo "RUNNING ${dataset} assemble=${assemble_mode} qa_top_k=${qa_top_k}" > "$STATUS"
  log "START dataset=${dataset} assemble=${assemble_mode} qa_top_k=${qa_top_k}"
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
    --expand_base_k 10 \
    --append_max_docs 3 \
    --expand_min_structure_score 0.35 \
    --assemble_mode "$assemble_mode" \
    --structure_relation_probe_mode general_factual \
    --qa_top_k "$qa_top_k" \
    --ce_device cuda:0 \
    --output_json "$out_json" >> "$LOG" 2>&1
  log "DONE dataset=${dataset} assemble=${assemble_mode} qa_top_k=${qa_top_k} report=${out_json}"
}

build_summary() {
  "$PY" - <<'PY'
import json
from pathlib import Path

date_tag = "20260406"
datasets = ["musique", "hotpotqa", "2wikimultihopqa"]
assemble_modes = ["none", "base_score", "embedding_similarity", "cross_encoder"]
qa_top_ks = [5, 7, 10]
rows = []

for dataset in datasets:
    for qa_top_k in qa_top_ks:
        baseline_path = Path(
            f"outputs_step0_general_{dataset}/eval_reports/causal_eval_{dataset}_100_qwen3-8b_qatopk{qa_top_k}_baseline_{date_tag}.json"
        )
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            overall = baseline.get("overall_recomputed", {})
            rows.append({
                "dataset": dataset,
                "assemble_mode": "baseline",
                "qa_top_k": qa_top_k,
                "em": overall.get("ExactMatch"),
                "f1": overall.get("F1"),
                "recall5": overall.get("Recall@5"),
                "report": str(baseline_path),
            })
        for assemble_mode in assemble_modes:
            report_path = Path(
                f"outputs_step0_general_{dataset}/eval_reports/bridge_append_{assemble_mode}_pool100_base10_append3_qatopk{qa_top_k}_{date_tag}.json"
            )
            if not report_path.exists():
                continue
            report = json.loads(report_path.read_text())
            method = report.get("expand_assemble_qa", {})
            rows.append({
                "dataset": dataset,
                "assemble_mode": assemble_mode,
                "qa_top_k": qa_top_k,
                "em": method.get("method_EM"),
                "f1": method.get("method_F1"),
                "recall5": ((method.get("method_retrieval_metrics") or {}).get("Recall@5")),
                "baseline_em_internal": method.get("baseline_EM"),
                "baseline_f1_internal": method.get("baseline_F1"),
                "delta_em": method.get("EM_delta"),
                "delta_f1": method.get("F1_delta"),
                "report": str(report_path),
            })

Path("run_logs/bridge_append_matrix_20260406.summary.json").write_text(
    json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

lines = [
    "# Bridge Append Matrix",
    "",
    "| Dataset | Assemble | qa_top_k | EM | F1 | Recall@5 | ΔEM |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    delta_em = row.get("delta_em")
    delta_str = "—" if delta_em is None else f"{delta_em:.4f}"
    lines.append(
        f"| {row['dataset']} | {row['assemble_mode']} | {row['qa_top_k']} | "
        f"{row['em']:.4f} | {row['f1']:.4f} | {row['recall5']:.4f} | {delta_str} |"
    )
Path("run_logs/bridge_append_matrix_20260406.summary.md").write_text(
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
  for qa_top_k in 5 7 10; do
    for assemble_mode in none base_score embedding_similarity cross_encoder; do
      run_eval "$dataset" "$assemble_mode" "$qa_top_k"
      build_summary
    done
  done
done

echo "DONE" > "$STATUS"
build_summary
log "Queue finished"
