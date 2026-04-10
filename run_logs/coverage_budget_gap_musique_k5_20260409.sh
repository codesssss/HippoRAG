#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

STATUS="run_logs/coverage_budget_gap_musique_k5_20260409.status"
LOG="run_logs/coverage_budget_gap_musique_k5_20260409.log"
SUMMARY_JSON="run_logs/coverage_budget_gap_musique_k5_20260409.summary.json"
SUMMARY_MD="run_logs/coverage_budget_gap_musique_k5_20260409.summary.md"
TRACE_JSON="run_logs/coverage_budget_gap_musique_k5_20260409.trace.json"
TRACE_MD="run_logs/coverage_budget_gap_musique_k5_20260409.trace.md"
PID_FILE="run_logs/coverage_budget_gap_musique_k5_20260409.pid"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

PY=".venv-hipporag/bin/python"
SCRIPT="scripts/eval_causal_qwen3.py"
TRACE_SCRIPT="scripts/analyze_coverage_v2_trace.py"
LLM_NAME="qwen3-8b"
LLM_REQUEST_NAME="qwen3-8b-train"
LLM_BASE_URL="http://localhost:8043/v1"
EMBED_NAME="VLLM//mnt/nvme/Qwen3-Embedding-8B"
EMBED_BASE_URL="http://localhost:8018/v1/embeddings"
SAVE_DIR="outputs_step0_general"
DATE_TAG="20260409bg"
DATASET="musique"
QA_TOP_K="5"
APPEND_MAX_DOCS="3"
ATOM_SOURCE="baseline_prefix"

LEGACY_V1_REPORT="outputs_step0_general_${DATASET}/eval_reports/bridge_append_coverage_pool100_base10_append${APPEND_MAX_DOCS}_qatopk${QA_TOP_K}_atomscandidate_pool_20260408.json"
LEGACY_APPEND0_REPORT="outputs_step0_general_${DATASET}/eval_reports/bridge_append_coverage_pool100_base10_append0_qatopk${QA_TOP_K}_atomsbaseline_prefix_20260408.json"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"
}

report_path_for_mode() {
  local admissibility_mode="$1"
  printf 'outputs_step0_general_%s/eval_reports/bridge_append_coverage_pool100_base10_append%s_qatopk%s_atoms%s_adm%s_%s.json' \
    "$DATASET" "$APPEND_MAX_DOCS" "$QA_TOP_K" "$ATOM_SOURCE" "$admissibility_mode" "$DATE_TAG"
}

run_eval() {
  local admissibility_mode="$1"
  local out_json
  out_json="$(report_path_for_mode "$admissibility_mode")"

  if [[ -f "$out_json" ]]; then
    log "SKIP existing qa_top_k=${QA_TOP_K} append=${APPEND_MAX_DOCS} atom_source=${ATOM_SOURCE} admissibility=${admissibility_mode} report=${out_json}"
    return 0
  fi

  echo "RUNNING dataset=${DATASET} qa_top_k=${QA_TOP_K} append=${APPEND_MAX_DOCS} atom_source=${ATOM_SOURCE} admissibility=${admissibility_mode}" > "$STATUS"
  log "START dataset=${DATASET} qa_top_k=${QA_TOP_K} append=${APPEND_MAX_DOCS} atom_source=${ATOM_SOURCE} admissibility=${admissibility_mode}"
  "$PY" "$SCRIPT" \
    --dataset "$DATASET" \
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
    --append_max_docs "$APPEND_MAX_DOCS" \
    --expand_min_structure_score 0.35 \
    --assemble_mode coverage \
    --coverage_score_variant qe_ce \
    --coverage_atom_source "$ATOM_SOURCE" \
    --coverage_admissibility_mode "$admissibility_mode" \
    --structure_relation_probe_mode general_factual \
    --qa_top_k "$QA_TOP_K" \
    --ce_device cuda:0 \
    --output_json "$out_json" >> "$LOG" 2>&1
  log "DONE dataset=${DATASET} qa_top_k=${QA_TOP_K} append=${APPEND_MAX_DOCS} atom_source=${ATOM_SOURCE} admissibility=${admissibility_mode} report=${out_json}"
}

build_summary() {
  "$PY" - <<'PY'
import json
from pathlib import Path

dataset = "musique"
qa_top_k = 5
append_max_docs = 3
atom_source = "baseline_prefix"
date_tag = "20260409bg"
modes = ["off", "budget_gap"]
rows = []

for mode in modes:
    path = Path(
        f"outputs_step0_general_{dataset}/eval_reports/bridge_append_coverage_pool100_base10_append{append_max_docs}_qatopk{qa_top_k}_atoms{atom_source}_adm{mode}_{date_tag}.json"
    )
    if not path.exists():
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    method = payload.get("expand_assemble_qa", {})
    rows.append(
        {
            "dataset": dataset,
            "qa_top_k": qa_top_k,
            "append_max_docs": append_max_docs,
            "atom_source": atom_source,
            "coverage_admissibility_mode": method.get("coverage_admissibility_mode", mode),
            "em": method.get("method_EM"),
            "f1": method.get("method_F1"),
            "delta_em": method.get("EM_delta"),
            "delta_f1": method.get("F1_delta"),
            "report": str(path),
        }
    )

Path("run_logs/coverage_budget_gap_musique_k5_20260409.summary.json").write_text(
    json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

def fmt(value):
    return "NA" if value is None else f"{float(value):.4f}"

lines = [
    "# Coverage Budget-Gap MuSiQue K=5",
    "",
    "| K | Append | Atom Source | Admissibility | EM | F1 | ΔEM | ΔF1 |",
    "|---:|---:|---|---|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['qa_top_k']} | {row['append_max_docs']} | {row['atom_source']} | {row['coverage_admissibility_mode']} | "
        f"{fmt(row['em'])} | {fmt(row['f1'])} | {fmt(row['delta_em'])} | {fmt(row['delta_f1'])} |"
    )
Path("run_logs/coverage_budget_gap_musique_k5_20260409.summary.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
PY
}

run_trace_diag() {
  local target_report
  target_report="$(report_path_for_mode budget_gap)"
  if [[ ! -f "$target_report" ]]; then
    return 0
  fi
  if [[ ! -f "$LEGACY_V1_REPORT" || ! -f "$LEGACY_APPEND0_REPORT" ]]; then
    log "SKIP trace diagnostic because legacy reference reports are missing"
    return 0
  fi

  log "TRACE start target=${target_report}"
  "$PY" "$TRACE_SCRIPT" \
    --dataset "$DATASET" \
    --qa_top_k "$QA_TOP_K" \
    --report_v1 "$LEGACY_V1_REPORT" \
    --report_target "$target_report" \
    --report_append0 "$LEGACY_APPEND0_REPORT" \
    --target_label budget_gap \
    --save_dir "$SAVE_DIR" \
    --output_json "$TRACE_JSON" \
    --output_md "$TRACE_MD" >> "$LOG" 2>&1
  log "TRACE done target=${target_report}"
}

trap 'code=$?; log "FAILED exit=${code} while running: $(cat "$STATUS" 2>/dev/null || echo unknown)"; build_summary || true; exit $code' ERR

: > "$LOG"
echo "$$" > "$PID_FILE"
echo "RUNNING boot" > "$STATUS"
log "Queue started on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

run_eval off
build_summary
run_eval budget_gap
build_summary
run_trace_diag

echo "DONE" > "$STATUS"
build_summary
log "Queue finished"
