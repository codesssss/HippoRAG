#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

STATUS="run_logs/action_swap_noisyor_smoke_20260414.status"
LOG="run_logs/action_swap_noisyor_smoke_20260414.log"
SUMMARY_JSON="run_logs/action_swap_noisyor_smoke_20260414.summary.json"
SUMMARY_MD="run_logs/action_swap_noisyor_smoke_20260414.summary.md"
PID_FILE="run_logs/action_swap_noisyor_smoke_20260414.pid"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

PY=".venv-hipporag/bin/python"
SCRIPT="scripts/eval_causal_qwen3.py"
LLM_NAME="qwen3-8b"
LLM_REQUEST_NAME="qwen3-8b-train"
LLM_BASE_URL="http://localhost:8043/v1"
EMBED_NAME="VLLM//mnt/nvme/Qwen3-Embedding-8B"
EMBED_BASE_URL="http://localhost:8018/v1/embeddings"
SAVE_DIR="outputs_step0_general"
DATE_TAG="20260414noisyor"
LIMIT="100"
QA_TOP_K="5"
EXPAND_BASE_K="10"
CE_DEVICE="cuda:0"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"
}

resolve_report_path() {
  local dataset="$1"
  local run_name="$2"
  printf 'outputs_step0_general_%s/eval_reports/%s_%s.json' "$dataset" "$run_name" "$DATE_TAG"
}

run_eval() {
  local dataset="$1"
  local mode="$2"
  local run_name="$3"
  local out_json
  out_json="$(resolve_report_path "$dataset" "$run_name")"

  if [[ -f "$out_json" ]]; then
    log "SKIP dataset=${dataset} run=${run_name} report=${out_json}"
    return 0
  fi

  echo "RUNNING dataset=${dataset} run=${run_name}" > "$STATUS"
  log "START dataset=${dataset} run=${run_name} assemble_mode=${mode}"
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
    --expand_base_k "$EXPAND_BASE_K" \
    --append_max_docs 3 \
    --append_policy bridge \
    --append_random_seed 0 \
    --expand_min_structure_score 0.35 \
    --assemble_mode "$mode" \
    --structure_relation_probe_mode general_factual \
    --qa_top_k "$QA_TOP_K" \
    --ce_device "$CE_DEVICE" \
    --output_json "$out_json" >> "$LOG" 2>&1
  log "DONE dataset=${dataset} run=${run_name} report=${out_json}"
}

build_summary() {
  "$PY" - <<'PY'
import json
from pathlib import Path

DATE_TAG = "20260414noisyor"
LIMIT = 100
QA_TOP_K = 5
DATASETS = ["musique", "2wikimultihopqa"]

existing_ref_candidates = {
    "bridge_append_plus_ce": [
        "width_match_bridge_append_plus_ce_qatopk5_20260409fullfix.json",
        "width_match_bridge_append_plus_ce_qatopk5_20260409smoke.json",
        "width_match_bridge_append_plus_ce_qatopk5_20260409full.json",
    ],
    "action_swap_v0_dryrun": [
        "width_match_bridge_append_plus_action_swap_v0_dryrun_qatopk5_20260413impl.json",
    ],
    "action_swap_v0_judge": [
        "width_match_bridge_append_plus_action_swap_v0_judge_qatopk5_20260413impl.json",
    ],
}

new_refs = {
    "action_swap_noisyor_flat": f"width_match_bridge_append_plus_action_swap_noisyor_flat_qatopk{QA_TOP_K}_{DATE_TAG}.json",
    "action_swap_noisyor_dep": f"width_match_bridge_append_plus_action_swap_noisyor_dep_qatopk{QA_TOP_K}_{DATE_TAG}.json",
}

rows = []
for dataset in DATASETS:
    eval_dir = Path(f"outputs_step0_general_{dataset}/eval_reports")
    resolved_existing_refs = {}
    for run_type, candidates in existing_ref_candidates.items():
        for candidate in candidates:
            candidate_path = eval_dir / candidate
            if candidate_path.exists():
                resolved_existing_refs[run_type] = candidate
                break
    all_refs = {**resolved_existing_refs, **new_refs}
    bridge_f1 = None
    bridge_em = None
    for run_type, filename in all_refs.items():
        report_path = eval_dir / filename
        if not report_path.exists():
            continue
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        method = dict(payload.get("expand_assemble_qa") or {})
        em = method.get("method_EM")
        f1 = method.get("method_F1")
        if em is None or f1 is None:
            continue
        if run_type == "bridge_append_plus_ce":
            bridge_em = float(em)
            bridge_f1 = float(f1)
        rows.append({
            "dataset": dataset,
            "run_type": run_type,
            "em": float(em),
            "f1": float(f1),
            "report": str(report_path),
        })

    if bridge_em is not None and bridge_f1 is not None:
        for row in rows:
            if row["dataset"] != dataset:
                continue
            row["delta_vs_bridge_em"] = round(float(row["em"]) - bridge_em, 4)
            row["delta_vs_bridge_f1"] = round(float(row["f1"]) - bridge_f1, 4)

summary = {"rows": rows}
Path("run_logs/action_swap_noisyor_smoke_20260414.summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

lines = [
    "# Action Swap NoisyOR Smoke 2026-04-14",
    "",
    "| Dataset | Run | EM | F1 | ΔEM vs bridge+CE | ΔF1 vs bridge+CE |",
    "|---|---|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['dataset']} | {row['run_type']} | {row['em']:.4f} | {row['f1']:.4f} | "
        f"{row.get('delta_vs_bridge_em', 0.0):+.4f} | {row.get('delta_vs_bridge_f1', 0.0):+.4f} |"
    )
Path("run_logs/action_swap_noisyor_smoke_20260414.summary.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
PY
}

trap 'code=$?; log "FAILED exit=${code} while running: $(cat "$STATUS" 2>/dev/null || echo unknown)"; build_summary || true; exit $code' ERR

: > "$LOG"
printf '%s\n' "$$" > "$PID_FILE"
echo "RUNNING boot" > "$STATUS"
log "Queue started on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CE_DEVICE=${CE_DEVICE}"

for dataset in musique 2wikimultihopqa; do
  run_eval "$dataset" "action_swap_noisyor_flat" "width_match_bridge_append_plus_action_swap_noisyor_flat_qatopk5"
  build_summary
  run_eval "$dataset" "action_swap_noisyor_dep" "width_match_bridge_append_plus_action_swap_noisyor_dep_qatopk5"
  build_summary
done

echo "DONE" > "$STATUS"
build_summary
log "Queue finished"
