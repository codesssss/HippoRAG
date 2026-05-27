#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
INDEX_ROOT="${ROOT_DIR}/run_logs/evidence_transition_graphragv4_clean_mainline_multi_anchor_strict_musique100_20260511"
RUN_TAG="etv4_clean_vs_no_multi_anchor_full1000_20260511"
GATE_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
CLEAN_ROOT="${ROOT_DIR}/run_logs/etv4_clean_mainline_full1000_reuse_gpt4omini_20260511"
ABL_ROOT="${ROOT_DIR}/run_logs/etv4_no_multi_anchor_precision_full1000_reuse_gpt4omini_20260511"
DATASETS=("2wikimultihopqa" "musique" "hotpotqa")

mkdir -p "${GATE_ROOT}/logs" "${GATE_ROOT}/reports"

status_path="${GATE_ROOT}/launcher.status"
log_path="${GATE_ROOT}/logs/launcher.log"

setup_reuse_indexes() {
  local out_root="$1"
  mkdir -p "${out_root}"
  for dataset in "${DATASETS[@]}"; do
    local src="${INDEX_ROOT}/${dataset}/index"
    local dst_parent="${out_root}/${dataset}"
    local dst="${dst_parent}/index"
    if [[ ! -d "${src}" ]]; then
      echo "[ERROR] missing reusable index: ${src}" >&2
      exit 1
    fi
    mkdir -p "${dst_parent}"
    if [[ -e "${dst}" && ! -L "${dst}" ]]; then
      echo "[ERROR] refusing to replace non-symlink index path: ${dst}" >&2
      exit 1
    fi
    ln -sfn "${src}" "${dst}"
  done
}

run_retrieval() {
  local policy="$1"
  local out_root="$2"
  echo "[RUN] policy=${policy} output=${out_root} start=$(date -Is)"
  (
    cd "${ROOT_DIR}"
    PYTHONPATH=. "${PYTHON_BIN}" \
      evidence_transition_graphragv4_fact_witnessed_sto/run_fresh_e2e.py \
      --datasets 2wikimultihopqa,musique,hotpotqa \
      --output-root "${out_root}" \
      --reuse-current-fresh-index \
      --max-queries 1000 \
      --llm-name gpt-4o-mini \
      --llm-base-url https://yunwu.ai/v1 \
      --embedding-name VLLM/nvidia/NV-Embed-v2 \
      --embedding-base-url http://localhost:8019/v1/embeddings \
      --readout-policy "${policy}" \
      --candidate-pool-k 200 \
      --top-k 5
  )
  echo "[DONE] policy=${policy} output=${out_root} end=$(date -Is)"
}

write_gate_report() {
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/nvme/code/HippoRAG")
clean_root = root / "run_logs/etv4_clean_mainline_full1000_reuse_gpt4omini_20260511"
abl_root = root / "run_logs/etv4_no_multi_anchor_precision_full1000_reuse_gpt4omini_20260511"
out_root = root / "run_logs/etv4_clean_vs_no_multi_anchor_full1000_20260511/reports"
datasets = ["2wikimultihopqa", "musique", "hotpotqa"]
method = "evidence_transition_graphragv4_fact_witnessed_sto"

def load_report(base: Path, dataset: str) -> dict:
    return json.loads((base / dataset / "reports" / f"{dataset}_{method}_retrieval.json").read_text(encoding="utf-8"))

def top5(row: dict) -> tuple[int, ...]:
    return tuple(int(x) for x in (row.get("retrieved_doc_indices_top5") or [])[:5])

def recall(row: dict) -> float:
    return float(row.get("query_grounded_sto_recall_at5") or 0.0)

def all_gold(row: dict) -> bool:
    return bool(row.get("query_grounded_sto_all_gold_at5"))

summary = {
    "clean_root": str(clean_root),
    "ablation_root": str(abl_root),
    "ablation": "no_multi_anchor_precision",
    "datasets": [],
}
md = [
    "# ETv4 No Multi-Anchor Precision Full1000 Retrieval Gate",
    "",
    "| Dataset | Rows | Clean R@5 | No-rule R@5 | Delta R@5 | Clean all-gold@5 | No-rule all-gold@5 | Delta all-gold@5 | Top5 changed | Gold gains | Gold losses | Net gold losses | Decision signal |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
]
for dataset in datasets:
    clean = load_report(clean_root, dataset)
    abl = load_report(abl_root, dataset)
    clean_rows = clean.get("rows", []) or []
    abl_rows = abl.get("rows", []) or []
    if len(clean_rows) != len(abl_rows):
        raise ValueError(f"row count mismatch for {dataset}: {len(clean_rows)} vs {len(abl_rows)}")
    changed = gains = losses = 0
    recall_deltas = []
    changed_examples = []
    for idx, (c, a) in enumerate(zip(clean_rows, abl_rows)):
        c_top = top5(c)
        a_top = top5(a)
        if c_top != a_top:
            changed += 1
            c_rec = recall(c)
            a_rec = recall(a)
            delta = a_rec - c_rec
            recall_deltas.append(delta)
            if delta > 1e-12:
                gains += 1
            elif delta < -1e-12:
                losses += 1
            if len(changed_examples) < 20:
                changed_examples.append({
                    "query_index": c.get("query_index", idx),
                    "question": c.get("question", ""),
                    "clean_top5": list(c_top),
                    "no_rule_top5": list(a_top),
                    "clean_recall_at5": c_rec,
                    "no_rule_recall_at5": a_rec,
                    "delta_recall_at5": delta,
                    "clean_all_gold_at5": all_gold(c),
                    "no_rule_all_gold_at5": all_gold(a),
                })
    clean_metrics = clean.get("metrics", {}) or {}
    abl_metrics = abl.get("metrics", {}) or {}
    clean_r5 = float(clean_metrics.get("r5") or 0.0)
    abl_r5 = float(abl_metrics.get("r5") or 0.0)
    clean_all = float(clean_metrics.get("all_gold_at5") or 0.0)
    abl_all = float(abl_metrics.get("all_gold_at5") or 0.0)
    delta_r5 = abl_r5 - clean_r5
    delta_all = abl_all - clean_all
    net_losses = losses - gains
    decision = "hold"
    if abs(delta_r5) <= 0.002 and abs(delta_all) <= 0.003 and net_losses <= 2:
        decision = "retrieval_tie_delete_candidate"
    elif losses > 2 or delta_r5 < -0.002 or delta_all < -0.003:
        decision = "retrieval_loss_keep_candidate"
    dataset_summary = {
        "dataset": dataset,
        "rows": len(clean_rows),
        "clean_r5": clean_r5,
        "ablation_r5": abl_r5,
        "delta_r5": delta_r5,
        "clean_all_gold_at5": clean_all,
        "ablation_all_gold_at5": abl_all,
        "delta_all_gold_at5": delta_all,
        "top5_changed": changed,
        "gold_gains": gains,
        "gold_losses": losses,
        "net_gold_losses": net_losses,
        "decision_signal": decision,
        "changed_examples": changed_examples,
    }
    summary["datasets"].append(dataset_summary)
    md.append(
        f"| {dataset} | {len(clean_rows)} | {clean_r5:.4f} | {abl_r5:.4f} | {delta_r5:+.4f} | "
        f"{clean_all:.4f} | {abl_all:.4f} | {delta_all:+.4f} | {changed} | {gains} | {losses} | {net_losses} | {decision} |"
    )

out_root.mkdir(parents=True, exist_ok=True)
(out_root / "no_multi_anchor_precision_full1000_gate.json").write_text(
    json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
(out_root / "no_multi_anchor_precision_full1000_gate.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps({row["dataset"]: {k: row[k] for k in ["delta_r5", "delta_all_gold_at5", "top5_changed", "gold_losses", "decision_signal"]} for row in summary["datasets"]}, sort_keys=True))
PY
  )
}

{
  echo "[START] run_tag=${RUN_TAG} start=$(date -Is)"
  setup_reuse_indexes "${CLEAN_ROOT}"
  setup_reuse_indexes "${ABL_ROOT}"
  run_retrieval "clean_mainline" "${CLEAN_ROOT}"
  run_retrieval "no_multi_anchor_precision" "${ABL_ROOT}"
  write_gate_report
  echo "[DONE] run_tag=${RUN_TAG} end=$(date -Is)"
} 2>&1 | tee "${log_path}"

echo "[DONE] run_tag=${RUN_TAG} end=$(date -Is)" > "${status_path}"
