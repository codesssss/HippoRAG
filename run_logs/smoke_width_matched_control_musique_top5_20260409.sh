#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

STATUS="run_logs/smoke_width_matched_control_musique_top5_20260409.status"
LOG="run_logs/smoke_width_matched_control_musique_top5_20260409.log"
SUMMARY_JSON="run_logs/smoke_width_matched_control_musique_top5_20260409.summary.json"
SUMMARY_MD="run_logs/smoke_width_matched_control_musique_top5_20260409.summary.md"
PID_FILE="run_logs/smoke_width_matched_control_musique_top5_20260409.pid"

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
DATE_TAG="20260409smoke"
DATASET="musique"
QA_TOP_K="5"
EXPAND_BASE_K="10"
LIMIT="100"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"
}

run_baseline() {
  local out_json="outputs_step0_general_${DATASET}/eval_reports/causal_eval_${DATASET}_${LIMIT}_qwen3-8b_qatopk${QA_TOP_K}_baseline_${DATE_TAG}.json"

  if [[ -f "$out_json" ]]; then
    log "SKIP existing run=baseline_top${QA_TOP_K} report=${out_json}"
    return 0
  fi

  echo "RUNNING dataset=${DATASET} run=baseline_top${QA_TOP_K}" > "$STATUS"
  log "START dataset=${DATASET} run=baseline_top${QA_TOP_K}"
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

  if [[ -f "$out_json" ]]; then
    log "SKIP existing run=${run_name} report=${out_json}"
    return 0
  fi

  echo "RUNNING dataset=${DATASET} run=${run_name}" > "$STATUS"
  log "START dataset=${DATASET} run=${run_name} append_policy=${append_policy} append_max_docs=${append_max_docs} seed=${append_random_seed}"
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
    --setwise_selector bridge_append \
    --setwise_score_mode bridge \
    --setwise_pool_k 100 \
    --setwise_non_anchor_title_dedup true \
    --expand_base_k "$EXPAND_BASE_K" \
    --append_max_docs "$append_max_docs" \
    --append_policy "$append_policy" \
    --append_random_seed "$append_random_seed" \
    --expand_min_structure_score 0.35 \
    --assemble_mode cross_encoder \
    --structure_relation_probe_mode general_factual \
    --qa_top_k "$QA_TOP_K" \
    --ce_device cuda:0 \
    --output_json "$out_json" >> "$LOG" 2>&1
  log "DONE dataset=${DATASET} run=${run_name} report=${out_json}"
}

build_summary() {
  "$PY" - <<'PY'
import json
from pathlib import Path

dataset = "musique"
qa_top_k = 5
limit = 100
date_tag = "20260409smoke"
rows = []

baseline_path = Path(
    f"outputs_step0_general_{dataset}/eval_reports/causal_eval_{dataset}_{limit}_qwen3-8b_qatopk{qa_top_k}_baseline_{date_tag}.json"
)
run_paths = {
    "baseline_top10_plus_ce": Path(
        f"outputs_step0_general_{dataset}/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_{date_tag}.json"
    ),
    "random3_deep_plus_ce": Path(
        f"outputs_step0_general_{dataset}/eval_reports/width_match_random3_deep_plus_ce_qatopk5_{date_tag}.json"
    ),
    "bridge_append_plus_ce": Path(
        f"outputs_step0_general_{dataset}/eval_reports/width_match_bridge_append_plus_ce_qatopk5_{date_tag}.json"
    ),
}

baseline_top10_row = None

if baseline_path.exists():
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    overall = baseline.get("overall_recomputed", {}) or {}
    rows.append(
        {
            "dataset": dataset,
            "run_type": "baseline_top5",
            "qa_top_k": qa_top_k,
            "em": overall.get("ExactMatch"),
            "f1": overall.get("F1"),
            "report": str(baseline_path),
        }
    )

for run_type, report_path in run_paths.items():
    if not report_path.exists():
        continue
    report = json.loads(report_path.read_text(encoding="utf-8"))
    method = report.get("expand_assemble_qa", {}) or {}
    row = {
        "dataset": dataset,
        "run_type": run_type,
        "qa_top_k": qa_top_k,
        "em": method.get("method_EM"),
        "f1": method.get("method_F1"),
        "delta_vs_baseline_em": method.get("EM_delta"),
        "delta_vs_baseline_f1": method.get("F1_delta"),
        "report": str(report_path),
    }
    rows.append(row)
    if run_type == "baseline_top10_plus_ce":
        baseline_top10_row = row

if baseline_top10_row is not None:
    for row in rows:
        if row["run_type"] in {"baseline_top10_plus_ce", "random3_deep_plus_ce", "bridge_append_plus_ce"}:
            row["delta_vs_baseline_top10_plus_ce_em"] = round(float(row["em"]) - float(baseline_top10_row["em"]), 4)
            row["delta_vs_baseline_top10_plus_ce_f1"] = round(float(row["f1"]) - float(baseline_top10_row["f1"]), 4)

Path("run_logs/smoke_width_matched_control_musique_top5_20260409.summary.json").write_text(
    json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

lines = [
    "# Smoke Width-Matched Control: MuSiQue top-5",
    "",
    "| Run | EM | F1 | ΔEM vs baseline top-5 | ΔEM vs baseline top-10+CE |",
    "|---|---:|---:|---:|---:|",
]
for row in rows:
    delta_baseline = row.get("delta_vs_baseline_em")
    delta_top10 = row.get("delta_vs_baseline_top10_plus_ce_em")
    delta_baseline_str = "—" if delta_baseline is None else f"{float(delta_baseline):+.4f}"
    delta_top10_str = "—" if delta_top10 is None else f"{float(delta_top10):+.4f}"
    lines.append(
        f"| {row['run_type']} | {float(row['em']):.4f} | {float(row['f1']):.4f} | {delta_baseline_str} | {delta_top10_str} |"
    )
Path("run_logs/smoke_width_matched_control_musique_top5_20260409.summary.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
PY
}

trap 'code=$?; log "FAILED exit=${code} while running: $(cat "$STATUS" 2>/dev/null || echo unknown)"; build_summary || true; exit $code' ERR

: > "$LOG"
printf '%s\n' "$$" > "$PID_FILE"
echo "RUNNING boot" > "$STATUS"
log "Queue started on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

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
log "Queue finished"
