#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

STATUS="run_logs/smoke_width_matched_control_nq_popqa_20260410.status"
LOG="run_logs/smoke_width_matched_control_nq_popqa_20260410.log"
SUMMARY_JSON="run_logs/smoke_width_matched_control_nq_popqa_20260410.summary.json"
SUMMARY_MD="run_logs/smoke_width_matched_control_nq_popqa_20260410.summary.md"
PID_FILE="run_logs/smoke_width_matched_control_nq_popqa_20260410.pid"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

PY=".venv-hipporag/bin/python"
SCRIPT="scripts/eval_causal_qwen3.py"
LLM_NAME="qwen3-8b"
LLM_REQUEST_NAME="qwen3-8b-train"
LLM_BASE_URL="http://localhost:8043/v1"
EMBED_NAME="VLLM//mnt/nvme/Qwen3-Embedding-8B"
EMBED_BASE_URL="http://localhost:8018/v1/embeddings"
SAVE_DIR="outputs_step0_general"
DATE_TAG="20260410nqpopqa_lightsmoke"
LIMIT="5"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"
}

run_width_matched_ce() {
  local dataset="$1"
  local run_name="$2"
  local append_policy="$3"
  local append_max_docs="$4"
  local append_random_seed="$5"
  local out_json="outputs_step0_general_${dataset}/eval_reports/${run_name}_${DATE_TAG}.json"

  if [[ -f "$out_json" ]]; then
    log "SKIP existing dataset=${dataset} run=${run_name} report=${out_json}"
    return 0
  fi

  echo "RUNNING dataset=${dataset} run=${run_name}" > "$STATUS"
  log "START dataset=${dataset} run=${run_name} append_policy=${append_policy} append_max_docs=${append_max_docs} seed=${append_random_seed}"
  "$PY" "$SCRIPT" \
    --dataset "$dataset" \
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
    --qa_top_k 5 \
    --ce_device cuda:0 \
    --output_json "$out_json" >> "$LOG" 2>&1
  log "DONE dataset=${dataset} run=${run_name} report=${out_json}"
}

build_summary() {
  "$PY" - <<'PY'
import json
from pathlib import Path

datasets = ["nq", "popqa"]
date_tag = "20260410nqpopqa_lightsmoke"
rows = []

for dataset in datasets:
    for run_type in [
        "width_match_baseline_top10_plus_ce_qatopk5",
    ]:
        report_path = Path(f"outputs_step0_general_{dataset}/eval_reports/{run_type}_{date_tag}.json")
        if not report_path.exists():
            continue
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        method = payload.get("expand_assemble_qa", {}) or {}
        rows.append({
            "dataset": dataset,
            "run_type": run_type,
            "em": method.get("method_EM"),
            "f1": method.get("method_F1"),
            "report": str(report_path),
        })

Path("run_logs/smoke_width_matched_control_nq_popqa_20260410.summary.json").write_text(
    json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

lines = [
    "# Smoke Width-Matched Control: NQ + PopQA top-5",
    "",
    "Naming note:",
    "- `nq` resolves to the packaged `nq_rear` files already present in `reproduce/dataset/`.",
    "- `nq_rear` and `popqa` here are the repo/HippoRAG-bundle `1000`-query evaluation subsets.",
    "- This light smoke uses `limit=5` and only runs the width-matched no-append CE control.",
    "",
    "| Dataset | Run | EM | F1 |",
    "|---|---|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['dataset']} | {row['run_type']} | {float(row['em']):.4f} | {float(row['f1']):.4f} |"
    )
Path("run_logs/smoke_width_matched_control_nq_popqa_20260410.summary.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
PY
}

trap 'code=$?; log "FAILED exit=${code} while running: $(cat "$STATUS" 2>/dev/null || echo unknown)"; build_summary || true; exit $code' ERR

: > "$LOG"
printf '%s\n' "$$" > "$PID_FILE"
echo "RUNNING boot" > "$STATUS"
log "NQ+PopQA smoke queue started on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for dataset in nq popqa; do
  run_width_matched_ce "$dataset" "width_match_baseline_top10_plus_ce_qatopk5" "bridge" 0 0
  build_summary
done

echo "DONE" > "$STATUS"
build_summary
log "NQ+PopQA smoke queue finished"
