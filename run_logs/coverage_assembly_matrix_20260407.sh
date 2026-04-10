#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

STATUS="run_logs/coverage_assembly_matrix_20260407.status"
LOG="run_logs/coverage_assembly_matrix_20260407.log"
SUMMARY_JSON="run_logs/coverage_assembly_matrix_20260407.summary.json"
SUMMARY_MD="run_logs/coverage_assembly_matrix_20260407.summary.md"
PID_FILE="run_logs/coverage_assembly_matrix_20260407.pid"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

PY=".venv-hipporag/bin/python"
SCRIPT="scripts/eval_causal_qwen3.py"
LLM_NAME="qwen3-8b"
LLM_REQUEST_NAME="qwen3-8b-train"
LLM_BASE_URL="http://localhost:8043/v1"
EMBED_NAME="VLLM//mnt/nvme/Qwen3-Embedding-8B"
EMBED_BASE_URL="http://localhost:8018/v1/embeddings"
SAVE_DIR="outputs_step0_general"

NEW_DATE_TAG="20260407"
BASELINE_DATE_TAG="20260406"
LEGACY_APPEND_DATE_TAG="20260406"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"
}

run_eval() {
  local dataset="$1"
  local qa_top_k="$2"
  local append_max_docs="$3"
  local assemble_mode="$4"
  local expand_base_k="$5"
  local out_json="outputs_step0_general_${dataset}/eval_reports/bridge_append_${assemble_mode}_pool100_base${expand_base_k}_append${append_max_docs}_qatopk${qa_top_k}_${NEW_DATE_TAG}.json"

  if [[ -f "$out_json" ]]; then
    log "SKIP existing dataset=${dataset} assemble=${assemble_mode} append=${append_max_docs} qa_top_k=${qa_top_k} report=${out_json}"
    return 0
  fi

  echo "RUNNING ${dataset} assemble=${assemble_mode} append=${append_max_docs} qa_top_k=${qa_top_k}" > "$STATUS"
  log "START dataset=${dataset} assemble=${assemble_mode} append=${append_max_docs} base=${expand_base_k} qa_top_k=${qa_top_k}"
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
    --expand_base_k "$expand_base_k" \
    --append_max_docs "$append_max_docs" \
    --expand_min_structure_score 0.35 \
    --assemble_mode "$assemble_mode" \
    --coverage_score_variant qe_ce \
    --structure_relation_probe_mode general_factual \
    --qa_top_k "$qa_top_k" \
    --ce_device cuda:0 \
    --output_json "$out_json" >> "$LOG" 2>&1
  log "DONE dataset=${dataset} assemble=${assemble_mode} append=${append_max_docs} qa_top_k=${qa_top_k} report=${out_json}"
}

build_summary() {
  "$PY" - <<'PY'
import json
from pathlib import Path

new_date_tag = "20260407"
baseline_date_tag = "20260406"
legacy_append_date_tag = "20260406"
datasets = ["musique", "hotpotqa", "2wikimultihopqa"]
qa_top_ks = [5, 7, 10]
rows = []


def load_json(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return None, path
    return json.loads(path.read_text()), path


for dataset in datasets:
    for qa_top_k in qa_top_ks:
        baseline, baseline_path = load_json(
            f"outputs_step0_general_{dataset}/eval_reports/causal_eval_{dataset}_100_qwen3-8b_qatopk{qa_top_k}_baseline_{baseline_date_tag}.json"
        )
        if baseline is not None:
            overall = baseline.get("overall_recomputed", {})
            rows.append(
                {
                    "dataset": dataset,
                    "group": "append0_none_baseline",
                    "qa_top_k": qa_top_k,
                    "em": overall.get("ExactMatch"),
                    "f1": overall.get("F1"),
                    "delta_em": 0.0,
                    "report": str(baseline_path),
                }
            )

        append0_ce, append0_ce_path = load_json(
            f"outputs_step0_general_{dataset}/eval_reports/bridge_append_cross_encoder_pool100_base10_append0_qatopk{qa_top_k}_{new_date_tag}.json"
        )
        if append0_ce is not None:
            method = append0_ce.get("expand_assemble_qa", {})
            rows.append(
                {
                    "dataset": dataset,
                    "group": "append0_cross_encoder_base10",
                    "qa_top_k": qa_top_k,
                    "em": method.get("method_EM"),
                    "f1": method.get("method_F1"),
                    "delta_em": method.get("EM_delta"),
                    "report": str(append0_ce_path),
                }
            )

        append0_cov, append0_cov_path = load_json(
            f"outputs_step0_general_{dataset}/eval_reports/bridge_append_coverage_pool100_base10_append0_qatopk{qa_top_k}_{new_date_tag}.json"
        )
        if append0_cov is not None:
            method = append0_cov.get("expand_assemble_qa", {})
            rows.append(
                {
                    "dataset": dataset,
                    "group": "append0_coverage_base10",
                    "qa_top_k": qa_top_k,
                    "em": method.get("method_EM"),
                    "f1": method.get("method_F1"),
                    "delta_em": method.get("EM_delta"),
                    "report": str(append0_cov_path),
                }
            )

        append3_none, append3_none_path = load_json(
            f"outputs_step0_general_{dataset}/eval_reports/bridge_append_none_pool100_base10_append3_qatopk{qa_top_k}_{legacy_append_date_tag}.json"
        )
        if append3_none is not None:
            method = append3_none.get("expand_assemble_qa", {})
            rows.append(
                {
                    "dataset": dataset,
                    "group": "append3_none",
                    "qa_top_k": qa_top_k,
                    "em": method.get("method_EM"),
                    "f1": method.get("method_F1"),
                    "delta_em": method.get("EM_delta"),
                    "report": str(append3_none_path),
                }
            )

        append3_ce, append3_ce_path = load_json(
            f"outputs_step0_general_{dataset}/eval_reports/bridge_append_cross_encoder_pool100_base10_append3_qatopk{qa_top_k}_{legacy_append_date_tag}.json"
        )
        if append3_ce is not None:
            method = append3_ce.get("expand_assemble_qa", {})
            rows.append(
                {
                    "dataset": dataset,
                    "group": "append3_cross_encoder",
                    "qa_top_k": qa_top_k,
                    "em": method.get("method_EM"),
                    "f1": method.get("method_F1"),
                    "delta_em": method.get("EM_delta"),
                    "report": str(append3_ce_path),
                }
            )

        append3_cov, append3_cov_path = load_json(
            f"outputs_step0_general_{dataset}/eval_reports/bridge_append_coverage_pool100_base10_append3_qatopk{qa_top_k}_{new_date_tag}.json"
        )
        if append3_cov is not None:
            method = append3_cov.get("expand_assemble_qa", {})
            rows.append(
                {
                    "dataset": dataset,
                    "group": "append3_coverage",
                    "qa_top_k": qa_top_k,
                    "em": method.get("method_EM"),
                    "f1": method.get("method_F1"),
                    "delta_em": method.get("EM_delta"),
                    "report": str(append3_cov_path),
                }
            )

Path("run_logs/coverage_assembly_matrix_20260407.summary.json").write_text(
    json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

lines = [
    "# Coverage Assembly Matrix",
    "",
    "| Dataset | Group | qa_top_k | EM | F1 | ΔEM vs baseline |",
    "|---|---|---:|---:|---:|---:|",
]
for row in rows:
    em = row.get("em")
    f1 = row.get("f1")
    delta = row.get("delta_em")
    lines.append(
        f"| {row['dataset']} | {row['group']} | {row['qa_top_k']} | "
        f"{'—' if em is None else f'{em:.4f}'} | "
        f"{'—' if f1 is None else f'{f1:.4f}'} | "
        f"{'—' if delta is None else f'{delta:.4f}'} |"
    )
Path("run_logs/coverage_assembly_matrix_20260407.summary.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
PY
}

trap 'code=$?; log "FAILED exit=${code} while running: $(cat "$STATUS" 2>/dev/null || echo unknown)"; build_summary || true; exit $code' ERR

printf '%s\n' "$$" > "$PID_FILE"
: > "$LOG"
echo "RUNNING boot" > "$STATUS"
log "Queue started"
log "Environment CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} HF_ENDPOINT=${HF_ENDPOINT}"

for qa_top_k in 5 7; do
  for dataset in musique hotpotqa 2wikimultihopqa; do
    run_eval "$dataset" "$qa_top_k" 0 cross_encoder 10
    build_summary
    run_eval "$dataset" "$qa_top_k" 0 coverage 10
    build_summary
    run_eval "$dataset" "$qa_top_k" 3 coverage 10
    build_summary
  done
done

for qa_top_k in 10; do
  for dataset in musique hotpotqa 2wikimultihopqa; do
    run_eval "$dataset" "$qa_top_k" 0 cross_encoder 10
    build_summary
    run_eval "$dataset" "$qa_top_k" 0 coverage 10
    build_summary
    run_eval "$dataset" "$qa_top_k" 3 coverage 10
    build_summary
  done
done

echo "DONE" > "$STATUS"
build_summary
log "Queue finished"
