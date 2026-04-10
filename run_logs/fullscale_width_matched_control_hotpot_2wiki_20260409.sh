#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

STATUS="run_logs/fullscale_width_matched_control_hotpot_2wiki_20260409.status"
LOG="run_logs/fullscale_width_matched_control_hotpot_2wiki_20260409.log"
SUMMARY_JSON="run_logs/fullscale_width_matched_control_hotpot_2wiki_20260409.summary.json"
SUMMARY_MD="run_logs/fullscale_width_matched_control_hotpot_2wiki_20260409.summary.md"
PID_FILE="run_logs/fullscale_width_matched_control_hotpot_2wiki_20260409.pid"

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
DATE_TAG="20260409fullfix"
LIMIT="0"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"
}

run_baseline() {
  local dataset="$1"
  local qa_top_k="$2"
  local out_json="outputs_step0_general_${dataset}/eval_reports/causal_eval_${dataset}_full_qwen3-8b_qatopk${qa_top_k}_baseline_${DATE_TAG}.json"
  local retrieval_cache="outputs_step0_general_${dataset}/eval_reports/causal_eval_${dataset}_full_qwen3-8b_qatopk${qa_top_k}_baseline_${DATE_TAG}.retrieval_cache.json"

  if [[ -f "$out_json" ]]; then
    log "SKIP existing dataset=${dataset} run=baseline_top${qa_top_k} report=${out_json}"
    return 0
  fi

  echo "RUNNING dataset=${dataset} run=baseline_top${qa_top_k}" > "$STATUS"
  log "START dataset=${dataset} run=baseline_top${qa_top_k}"
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
    --save_retrieval_cache_json "$retrieval_cache" \
    --qa_top_k "$qa_top_k" \
    --output_json "$out_json" >> "$LOG" 2>&1
  log "DONE dataset=${dataset} run=baseline_top${qa_top_k} report=${out_json}"
}

run_width_matched_ce() {
  local dataset="$1"
  local qa_top_k="$2"
  local run_name="$3"
  local append_policy="$4"
  local append_max_docs="$5"
  local append_random_seed="$6"
  local out_json="outputs_step0_general_${dataset}/eval_reports/${run_name}_${DATE_TAG}.json"
  local baseline_report="outputs_step0_general_${dataset}/eval_reports/causal_eval_${dataset}_full_qwen3-8b_qatopk${qa_top_k}_baseline_${DATE_TAG}.json"
  local retrieval_cache="outputs_step0_general_${dataset}/eval_reports/causal_eval_${dataset}_full_qwen3-8b_qatopk${qa_top_k}_baseline_${DATE_TAG}.retrieval_cache.json"
  local extra_args=()

  if [[ -f "$out_json" ]]; then
    log "SKIP existing dataset=${dataset} run=${run_name} report=${out_json}"
    return 0
  fi

  if [[ -f "$retrieval_cache" ]]; then
    extra_args+=(--retrieval_cache_json "$retrieval_cache")
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
    --qa_top_k "$qa_top_k" \
    --ce_device cuda:0 \
    --output_json "$out_json" >> "$LOG" 2>&1
  log "DONE dataset=${dataset} run=${run_name} report=${out_json}"
}

build_summary() {
  "$PY" - <<'PY'
import json
from pathlib import Path

date_tag = "20260409fullfix"
rows = []

datasets = ["hotpotqa", "2wikimultihopqa"]
qa_top_k = 5

for dataset in datasets:
    baseline_path = Path(
        f"outputs_step0_general_{dataset}/eval_reports/causal_eval_{dataset}_full_qwen3-8b_qatopk{qa_top_k}_baseline_{date_tag}.json"
    )
    run_paths = {
        "baseline_top10_plus_ce": Path(
            f"outputs_step0_general_{dataset}/eval_reports/width_match_baseline_top10_plus_ce_qatopk{qa_top_k}_{date_tag}.json"
        ),
        "random3_deep_plus_ce": Path(
            f"outputs_step0_general_{dataset}/eval_reports/width_match_random3_deep_plus_ce_qatopk{qa_top_k}_{date_tag}.json"
        ),
        "bridge_append_plus_ce": Path(
            f"outputs_step0_general_{dataset}/eval_reports/width_match_bridge_append_plus_ce_qatopk{qa_top_k}_{date_tag}.json"
        ),
    }

    baseline_top10_row = None

    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        overall = baseline.get("overall_recomputed", {}) or {}
        rows.append({
            "dataset": dataset,
            "run_type": "baseline_top5",
            "qa_top_k": qa_top_k,
            "em": overall.get("ExactMatch"),
            "f1": overall.get("F1"),
            "recall_at_5": overall.get("Recall@5"),
            "recall_at_20": overall.get("Recall@20"),
            "recall_at_100": overall.get("Recall@100"),
            "num_queries": overall.get("num_queries"),
            "report": str(baseline_path),
        })

    dataset_rows = []
    for run_type, report_path in run_paths.items():
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        method = report.get("expand_assemble_qa", {}) or {}
        retrieval = method.get("method_retrieval_metrics", {}) or {}
        row = {
            "dataset": dataset,
            "run_type": run_type,
            "qa_top_k": qa_top_k,
            "em": method.get("method_EM"),
            "f1": method.get("method_F1"),
            "recall_at_5": retrieval.get("Recall@5"),
            "recall_at_20": retrieval.get("Recall@20"),
            "recall_at_100": retrieval.get("Recall@100"),
            "delta_vs_baseline_em": method.get("EM_delta"),
            "delta_vs_baseline_f1": method.get("F1_delta"),
            "report": str(report_path),
        }
        dataset_rows.append(row)
        if run_type == "baseline_top10_plus_ce":
            baseline_top10_row = row
    rows.extend(dataset_rows)

    if baseline_top10_row is not None:
        for row in dataset_rows:
            if row["run_type"] in {"baseline_top10_plus_ce", "random3_deep_plus_ce", "bridge_append_plus_ce"}:
                row["delta_vs_baseline_top10_plus_ce_em"] = round(float(row["em"]) - float(baseline_top10_row["em"]), 4)
                row["delta_vs_baseline_top10_plus_ce_f1"] = round(float(row["f1"]) - float(baseline_top10_row["f1"]), 4)

    if baseline_top10_row is not None:
        for row in rows:
            if row["dataset"] != dataset:
                continue
            if row["run_type"] in {"baseline_top10_plus_ce", "random3_deep_plus_ce", "bridge_append_plus_ce"}:
                row["delta_vs_baseline_top10_plus_ce_em"] = round(float(row["em"]) - float(baseline_top10_row["em"]), 4)
                row["delta_vs_baseline_top10_plus_ce_f1"] = round(float(row["f1"]) - float(baseline_top10_row["f1"]), 4)

Path("run_logs/fullscale_width_matched_control_hotpot_2wiki_20260409.summary.json").write_text(
    json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

def fmt(value):
    return "—" if value is None else f"{float(value):.4f}"

lines = [
    "# Full-Scale Width-Matched Control: HotpotQA + 2Wiki top-5",
    "",
    "| Dataset | Run | EM | F1 | R@5 | R@20 | R@100 | num_queries | ΔEM vs baseline top-5 | ΔEM vs baseline top-10+CE |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    delta_baseline = row.get("delta_vs_baseline_em")
    delta_top10 = row.get("delta_vs_baseline_top10_plus_ce_em")
    delta_baseline_str = "—" if delta_baseline is None else f"{float(delta_baseline):+.4f}"
    delta_top10_str = "—" if delta_top10 is None else f"{float(delta_top10):+.4f}"
    lines.append(
        f"| {row['dataset']} | {row['run_type']} | {fmt(row.get('em'))} | {fmt(row.get('f1'))} | "
        f"{fmt(row.get('recall_at_5'))} | {fmt(row.get('recall_at_20'))} | {fmt(row.get('recall_at_100'))} | "
        f"{row.get('num_queries', '—')} | {delta_baseline_str} | {delta_top10_str} |"
    )
Path("run_logs/fullscale_width_matched_control_hotpot_2wiki_20260409.summary.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
PY
}

trap 'code=$?; log "FAILED exit=${code} while running: $(cat "$STATUS" 2>/dev/null || echo unknown)"; build_summary || true; exit $code' ERR

: > "$LOG"
printf '%s\n' "$$" > "$PID_FILE"
echo "RUNNING boot" > "$STATUS"
log "HotpotQA+2Wiki queue started on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for dataset in hotpotqa 2wikimultihopqa; do
  run_baseline "$dataset" 5
  build_summary
  run_width_matched_ce "$dataset" 5 "width_match_baseline_top10_plus_ce_qatopk5" "bridge" 0 0
  build_summary
  run_width_matched_ce "$dataset" 5 "width_match_random3_deep_plus_ce_qatopk5" "random_deep" 3 0
  build_summary
  run_width_matched_ce "$dataset" 5 "width_match_bridge_append_plus_ce_qatopk5" "bridge" 3 0
  build_summary
done

echo "DONE" > "$STATUS"
build_summary
log "HotpotQA+2Wiki queue finished"
