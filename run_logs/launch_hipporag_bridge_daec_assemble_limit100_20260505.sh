#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/hipporag_bridge_daec_assemble_limit100_20260505"
SAVE_DIR="outputs_step0_general_nvembed"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
EVAL_EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"
LLM_MODEL="qwen3-8b-train"
LIMIT=100
POOL_K=100
MATCH_MODE="wiki_title"
CE_MODEL="/mnt/nvme/bge-reranker-v2-m3"
CE_DEVICE="${CE_DEVICE:-cuda:4}"

export HIPPORAG_RERANK_FORCE_NO_THINK=1
mkdir -p "${OUT_DIR}/evals" "${OUT_DIR}/logs" "${OUT_DIR}/status"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_DIR}/logs/launcher.log"
}

dataset_port() {
  case "$1" in
    2wikimultihopqa) echo 8043 ;;
    hotpotqa) echo 8042 ;;
    musique) echo 8041 ;;
    *) return 2 ;;
  esac
}

pool_path() {
  local dataset="$1"
  echo "${ROOT_DIR}/run_logs/hipporag_pool_exports_full1000_20260503/${dataset}_hipporag_pool100.json"
}

variant_selector_args() {
  local variant="$1"
  case "${variant}" in
    bridge_daec)
      printf '%s\n' \
        --setwise_selector bridge_append \
        --setwise_score_mode bridge \
        --setwise_pool_k "${POOL_K}" \
        --setwise_non_anchor_title_dedup true \
        --expand_base_k 10 \
        --append_max_docs 3 \
        --append_policy bridge \
        --append_random_seed 0 \
        --expand_min_structure_score 0.35 \
        --assemble_mode daec_noisyor_llm \
        --structure_relation_probe_mode general_factual \
        --dtc_decomposition_mode llm \
        --dtc_binding_max_candidates 5
      ;;
    bridge_ce)
      printf '%s\n' \
        --setwise_selector bridge_append \
        --setwise_score_mode bridge \
        --setwise_pool_k "${POOL_K}" \
        --setwise_non_anchor_title_dedup true \
        --expand_base_k 10 \
        --append_max_docs 3 \
        --append_policy bridge \
        --append_random_seed 0 \
        --expand_min_structure_score 0.35 \
        --assemble_mode cross_encoder \
        --structure_relation_probe_mode general_factual \
        --ce_model "${CE_MODEL}" \
        --ce_device "${CE_DEVICE}"
      ;;
    *)
      return 2
      ;;
  esac
}

eval_variant() {
  local dataset="$1"
  local port="$2"
  local variant="$3"
  local pool_json output_json log_path status_path cache_path
  local selector_args=()

  pool_json="$(pool_path "${dataset}")"
  output_json="${OUT_DIR}/evals/${dataset}_${variant}_qwen8b_limit100.json"
  log_path="${OUT_DIR}/logs/${dataset}_${variant}.log"
  status_path="${OUT_DIR}/status/${dataset}_${variant}.status"
  cache_path="${OUT_DIR}/evals/${dataset}_shared.binding_cache.json"
  mapfile -t selector_args < <(variant_selector_args "${variant}") || return 2

  if [[ -s "${output_json}" ]]; then
    echo "[SKIP] dataset=${dataset} variant=${variant} output_exists=${output_json} time=$(date -Is)" > "${status_path}"
    log_msg "SKIP eval dataset=${dataset} variant=${variant} output=${output_json}"
    return 0
  fi

  log_msg "START eval dataset=${dataset} variant=${variant} limit=${LIMIT} qa_top_k=5 port=${port}"
  echo "[START] dataset=${dataset} variant=${variant} limit=${LIMIT} qa_top_k=5 port=${port} time=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name "${LLM_MODEL}" \
      --max_retry_attempts 20 \
      --qwen_disable_thinking \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name "${EVAL_EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --external_pool_json "${pool_json}" \
      --external_pool_source_name hipporag_pool100 \
      --external_pool_strict_questions true \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --llm_binding_url "http://localhost:${port}/v1" \
      --llm_binding_model "${LLM_MODEL}" \
      --llm_binding_cache_path "${cache_path}" \
      --llm_binding_title_match_mode "${MATCH_MODE}" \
      --causal_enabled false \
      --causal_engine_version v2 \
      --causal_v2_base_retrieval_mode dense \
      --structure_rerank_enabled false \
      "${selector_args[@]}" \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]] && [[ -s "${output_json}" ]]; then
    echo "[DONE] dataset=${dataset} variant=${variant} time=$(date -Is)" > "${status_path}"
    log_msg "DONE eval dataset=${dataset} variant=${variant} output=${output_json}"
  else
    echo "[FAILED] dataset=${dataset} variant=${variant} code=${code} time=$(date -Is)" > "${status_path}"
    log_msg "FAILED eval dataset=${dataset} variant=${variant} code=${code}"
  fi
  return "${code}"
}

run_dataset() {
  local dataset="$1"
  local port variant
  port="$(dataset_port "${dataset}")" || return 2
  for variant in bridge_daec bridge_ce; do
    eval_variant "${dataset}" "${port}" "${variant}" || return 1
  done
}

summarize_results() {
  "${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
datasets = ["2wikimultihopqa", "hotpotqa", "musique"]
variants = ["bridge_daec", "bridge_ce"]
rows = []

def metric(payload, key, default=0.0):
    try:
        return float(payload.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)

for dataset in datasets:
    for variant in variants:
        path = out_dir / "evals" / f"{dataset}_{variant}_qwen8b_limit100.json"
        if not path.exists():
            rows.append({"dataset": dataset, "variant": variant, "status": "missing"})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        qa = data.get("expand_assemble_qa") or data.get("setwise_selector_qa") or {}
        selector_metrics = qa.get("method_retrieval_metrics") or qa.get("selector_retrieval_metrics") or {}
        summary = qa.get("method_summary") or qa.get("selector_summary") or {}
        overall = data.get("overall_recomputed") or data.get("overall_from_pipeline") or {}
        pool_payload = ((data.get("external_pool") or {}).get("payload_retrieval") or {})
        pool_metrics = pool_payload.get("hipporag_metrics") or pool_payload.get("recomputed_title_recall") or {}
        baseline_em = metric(qa, "baseline_EM", overall.get("ExactMatch", 0.0))
        baseline_f1 = metric(qa, "baseline_F1", overall.get("F1", 0.0))
        method_em = metric(qa, "method_EM", qa.get("selector_EM", 0.0))
        method_f1 = metric(qa, "method_F1", qa.get("selector_F1", 0.0))
        baseline_r5 = metric(overall, "Recall@5")
        method_r5 = metric(selector_metrics, "Recall@5")
        rows.append({
            "dataset": dataset,
            "variant": variant,
            "status": "done",
            "baseline_em": baseline_em,
            "method_em": method_em,
            "delta_em": method_em - baseline_em,
            "baseline_f1": baseline_f1,
            "method_f1": method_f1,
            "delta_f1": method_f1 - baseline_f1,
            "baseline_r5": baseline_r5,
            "method_r5": method_r5,
            "delta_r5": method_r5 - baseline_r5,
            "pool_r5": metric(pool_metrics, "Recall@5"),
            "pool_r100": metric(pool_metrics, "Recall@100"),
            "avg_candidate_set_size": summary.get("avg_candidate_set_size"),
            "avg_appended_doc_count": summary.get("avg_appended_doc_count"),
            "append_stop_reason_counts": json.dumps(summary.get("append_stop_reason_counts") or {}, sort_keys=True),
        })

csv_path = out_dir / "hipporag_bridge_daec_assemble_limit100_results.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "dataset", "variant", "status", "baseline_em", "method_em", "delta_em",
        "baseline_f1", "method_f1", "delta_f1", "baseline_r5", "method_r5",
        "delta_r5", "pool_r5", "pool_r100", "avg_candidate_set_size",
        "avg_appended_doc_count", "append_stop_reason_counts",
    ])
    writer.writeheader()
    writer.writerows(rows)

def fmt(row, key):
    value = row.get(key)
    return "" if value in (None, "") else f"{float(value):.4f}"

lines = [
    "# HippoRAG Pool Bridge Append DAEC Assemble Limit100",
    "",
    "| Dataset | Variant | Base EM | EM | dEM | Base F1 | F1 | dF1 | Base R@5 | R@5 | dR@5 | AppendN |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    if row.get("status") != "done":
        lines.append(f"| {row['dataset']} | {row['variant']} | missing |  |  |  |  |  |  |  |  |  |")
        continue
    lines.append(
        f"| {row['dataset']} | {row['variant']} | {fmt(row, 'baseline_em')} | "
        f"{fmt(row, 'method_em')} | {fmt(row, 'delta_em')} | "
        f"{fmt(row, 'baseline_f1')} | {fmt(row, 'method_f1')} | {fmt(row, 'delta_f1')} | "
        f"{fmt(row, 'baseline_r5')} | {fmt(row, 'method_r5')} | {fmt(row, 'delta_r5')} | "
        f"{fmt(row, 'avg_appended_doc_count')} |"
    )
md_path = out_dir / "summary_hipporag_bridge_daec_assemble_limit100.md"
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {csv_path}")
print(f"wrote {md_path}")
PY
}

main() {
  echo "[START] hipporag_bridge_daec_assemble_limit100 time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
  log_msg "HippoRAG-pool bridge CE vs DAEC-assemble limit100 begin"
  log_msg "Using hipporag legacy_fact_graph pool100, Qwen3-8B reader/decomposer, CE_DEVICE=${CE_DEVICE}"

  local status=0
  run_dataset 2wikimultihopqa || status=1
  run_dataset hotpotqa || status=1
  run_dataset musique || status=1
  summarize_results || status=1

  if [[ "${status}" -eq 0 ]]; then
    echo "[DONE] hipporag_bridge_daec_assemble_limit100 time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
    log_msg "HippoRAG-pool bridge CE vs DAEC-assemble limit100 complete"
  else
    echo "[FAILED] hipporag_bridge_daec_assemble_limit100 status=${status} time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
    log_msg "HippoRAG-pool bridge CE vs DAEC-assemble limit100 finished with failures"
  fi
  return "${status}"
}

main "$@"
