#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/daec_binding_grounded_limit100_20260506"
SAVE_DIR="outputs_step0_general_nvembed"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
EVAL_EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"
LLM_MODEL="qwen3-8b-train"
LIMIT=100
POOL_K=100
MATCH_MODE="wiki_title"
VARIANTS=("base" "typeguard" "softcompat" "grounded" "full")
DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-musique hotpotqa 2wikimultihopqa}"

export HIPPORAG_RERANK_FORCE_NO_THINK=1
mkdir -p "${OUT_DIR}/evals" "${OUT_DIR}/logs" "${OUT_DIR}/status"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_DIR}/logs/launcher.log"
}

dataset_port() {
  case "$1" in
    musique) echo "8041" ;;
    hotpotqa) echo "8042" ;;
    2wikimultihopqa) echo "8043" ;;
    *) return 2 ;;
  esac
}

pool_path() {
  local dataset="$1"
  echo "${ROOT_DIR}/run_logs/hipporag_pool_exports_full1000_20260503/${dataset}_hipporag_pool100.json"
}

variant_args() {
  local variant="$1"
  case "${variant}" in
    base)
      printf '%s\n' \
        --daec_llm_binding_type_filter false \
        --daec_soft_compat_body_weight 0.0 \
        --daec_binding_grounding_enabled false \
        --daec_swap_refinement false
      ;;
    typeguard)
      printf '%s\n' \
        --daec_llm_binding_type_filter true \
        --daec_soft_compat_body_weight 0.0 \
        --daec_binding_grounding_enabled false \
        --daec_swap_refinement false
      ;;
    softcompat)
      printf '%s\n' \
        --daec_llm_binding_type_filter true \
        --daec_soft_compat_body_weight 0.5 \
        --daec_binding_grounding_enabled false \
        --daec_swap_refinement false
      ;;
    grounded)
      printf '%s\n' \
        --daec_llm_binding_type_filter true \
        --daec_soft_compat_body_weight 0.0 \
        --daec_binding_grounding_enabled true \
        --daec_swap_refinement false
      ;;
    full)
      printf '%s\n' \
        --daec_llm_binding_type_filter true \
        --daec_soft_compat_body_weight 0.5 \
        --daec_binding_grounding_enabled true \
        --daec_swap_refinement true \
        --daec_swap_min_gain 0.001
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
  local pool_json output_json cache_path status_path log_path
  local extra_args=()

  pool_json="$(pool_path "${dataset}")"
  output_json="${OUT_DIR}/evals/${dataset}_${variant}_qwen8b_hipporag_pool100_limit100.json"
  cache_path="${OUT_DIR}/evals/${dataset}_qwen8b_hipporag_pool100_shared.binding_cache.json"
  status_path="${OUT_DIR}/status/${dataset}_${variant}.status"
  log_path="${OUT_DIR}/logs/${dataset}_${variant}.log"
  mapfile -t extra_args < <(variant_args "${variant}") || return 2

  if [[ -s "${output_json}" ]]; then
    log_msg "SKIP eval dataset=${dataset} variant=${variant} existing=${output_json}"
    echo "[SKIP] dataset=${dataset} variant=${variant} existing=${output_json} time=$(date -Is)" > "${status_path}"
    return 0
  fi
  if [[ ! -s "${pool_json}" ]]; then
    log_msg "FAILED eval dataset=${dataset} variant=${variant} missing_pool=${pool_json}"
    echo "[FAILED] dataset=${dataset} variant=${variant} missing_pool=${pool_json} time=$(date -Is)" > "${status_path}"
    return 2
  fi

  log_msg "START eval dataset=${dataset} variant=${variant} limit=${LIMIT} port=${port}"
  echo "[START] dataset=${dataset} variant=${variant} limit=${LIMIT} port=${port} time=$(date -Is)" > "${status_path}"
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
      --external_pool_source_name hipporag_legacy_fact_graph_pool100 \
      --external_pool_strict_questions true \
      --setwise_selector daec_noisyor_llm \
      --setwise_pool_k "${POOL_K}" \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --dtc_decomposition_mode llm \
      --dtc_binding_max_candidates 5 \
      --llm_binding_url "http://localhost:${port}/v1" \
      --llm_binding_model "${LLM_MODEL}" \
      --llm_binding_cache_path "${cache_path}" \
      --llm_binding_title_match_mode "${MATCH_MODE}" \
      --causal_enabled false \
      --causal_engine_version legacy \
      --structure_rerank_enabled false \
      "${extra_args[@]}" \
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
  for variant in "${VARIANTS[@]}"; do
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
variants = ["base", "typeguard", "softcompat", "grounded", "full"]
rows = []

def metric(payload, key, default=0.0):
    try:
        return float(payload.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)

def count_trace_diagnostics(query_traces):
    type_rejects = 0
    grounding_changes = 0
    swap_steps = 0
    for query_trace in query_traces:
        trace = query_trace.get("selector_trace") or {}
        for extraction in trace.get("llm_binding_extractions") or []:
            type_rejects += len(extraction.get("type_incompatible_entities") or [])
        binding_objectives = trace.get("binding_objectives") or []
        selected_binding_id = str(trace.get("selected_binding_id") or "")
        if binding_objectives and bool((trace.get("binding_grounding") or {}).get("enabled")):
            best_effective = max(
                binding_objectives,
                key=lambda row: (
                    float(row.get("effective_objective", row.get("objective", 0.0)) or 0.0),
                    float(row.get("objective", 0.0) or 0.0),
                ),
            )
            if str(best_effective.get("binding_id") or "") != selected_binding_id:
                grounding_changes += 1
        swap_steps += len((trace.get("swap_refinement_trace") or {}).get("steps") or [])
    return type_rejects, grounding_changes, swap_steps

for dataset in datasets:
    base_metrics = None
    for variant in variants:
        path = out_dir / "evals" / f"{dataset}_{variant}_qwen8b_hipporag_pool100_limit100.json"
        if not path.exists():
            rows.append({"dataset": dataset, "variant": variant, "status": "missing"})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        qa = data.get("setwise_selector_qa") or {}
        selector_metrics = qa.get("selector_retrieval_metrics") or {}
        query_traces = data.get("setwise_selector_query_traces") or []
        type_rejects, grounding_changes, swap_steps = count_trace_diagnostics(query_traces)
        row = {
            "dataset": dataset,
            "variant": variant,
            "status": "done",
            "em": metric(qa, "selector_EM"),
            "f1": metric(qa, "selector_F1"),
            "r5": metric(selector_metrics, "Recall@5"),
            "base_em": metric(qa, "baseline_EM"),
            "base_f1": metric(qa, "baseline_F1"),
            "type_rejects": int(type_rejects),
            "grounding_changes": int(grounding_changes),
            "swap_steps": int(swap_steps),
        }
        if variant == "base":
            base_metrics = row
        if base_metrics:
            row["delta_em_vs_daec_base"] = row["em"] - base_metrics["em"]
            row["delta_f1_vs_daec_base"] = row["f1"] - base_metrics["f1"]
        else:
            row["delta_em_vs_daec_base"] = 0.0
            row["delta_f1_vs_daec_base"] = 0.0
        rows.append(row)

csv_path = out_dir / "daec_binding_grounded_limit100_results.csv"
fieldnames = [
    "dataset", "variant", "status", "em", "f1", "r5", "base_em", "base_f1",
    "delta_em_vs_daec_base", "delta_f1_vs_daec_base",
    "type_rejects", "grounding_changes", "swap_steps",
]
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})

done_rows = [row for row in rows if row.get("status") == "done"]
md_lines = [
    "# DAEC Binding-Grounded Ablation Limit100",
    "",
    "Pool: HippoRAG aligned legacy_fact_graph pool100. LLM/reader: Qwen3-8B with no_think. Embedding: NV-Embed-v2.",
    "",
    "```text",
    "+------------------+------------+--------+--------+--------+--------+--------+---------+---------+----------+--------+--------+",
    "| Dataset          | Variant    | EM     | F1     | R@5    | dEM    | dF1    | TypeRej | GroundC | SwapStep | BaseEM | BaseF1 |",
    "+------------------+------------+--------+--------+--------+--------+--------+---------+---------+----------+--------+--------+",
]
for row in done_rows:
    md_lines.append(
        "| {dataset:<16} | {variant:<10} | {em:>6.3f} | {f1:>6.3f} | {r5:>6.3f} | {dem:>+6.3f} | {df1:>+6.3f} | {trej:>7d} | {gchg:>7d} | {swp:>8d} | {bem:>6.3f} | {bf1:>6.3f} |".format(
            dataset=str(row["dataset"]),
            variant=str(row["variant"]),
            em=float(row["em"]),
            f1=float(row["f1"]),
            r5=float(row["r5"]),
            dem=float(row["delta_em_vs_daec_base"]),
            df1=float(row["delta_f1_vs_daec_base"]),
            trej=int(row["type_rejects"]),
            gchg=int(row["grounding_changes"]),
            swp=int(row["swap_steps"]),
            bem=float(row["base_em"]),
            bf1=float(row["base_f1"]),
        )
    )
md_lines.extend([
    "+------------------+------------+--------+--------+--------+--------+--------+---------+---------+----------+--------+--------+",
    "```",
    "",
    f"CSV: `{csv_path}`",
])
(out_dir / "summary_daec_binding_grounded_limit100.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print(f"[SUMMARY] wrote {csv_path}")
PY
}

main() {
  echo "[START] daec_binding_grounded_limit100 time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
  log_msg "DAEC binding-grounded limit100 begin"
  log_msg "lanes: musique->8041 hotpotqa->8042 2wikimultihopqa->8043"
  log_msg "datasets: ${DATASETS_OVERRIDE}"
  log_msg "variants: ${VARIANTS[*]}"

  local status=0
  local pids=()
  local pid_names=()
  local dataset
  for dataset in ${DATASETS_OVERRIDE}; do
    (run_dataset "${dataset}") &
    pids+=("$!")
    pid_names+=("${dataset}")
  done
  local idx
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      log_msg "FAILED dataset=${pid_names[$idx]}"
      status=1
    fi
  done

  summarize_results >> "${OUT_DIR}/logs/launcher.log" 2>&1 || status=1

  if [[ "${status}" -eq 0 ]]; then
    echo "[DONE] daec_binding_grounded_limit100 time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
    log_msg "DAEC binding-grounded limit100 complete"
  else
    echo "[FAILED] daec_binding_grounded_limit100 status=${status} time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
    log_msg "DAEC binding-grounded limit100 finished with failures"
  fi
  return "${status}"
}

main "$@"
