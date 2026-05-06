#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/agsto_daec_graphprior_limit100_20260505"
SAVE_DIR="outputs_step0_general_nvembed"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
EMBEDDING_API_MODEL="nvidia/NV-Embed-v2"
EVAL_EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"
LLM_MODEL="qwen3-8b-train"
LIMIT=100
POOL_K=100
MATCH_MODE="wiki_title"
BETAS=("0" "0.005" "0.01" "0.02" "0.05")

export HIPPORAG_RERANK_FORCE_NO_THINK=1
mkdir -p "${OUT_DIR}/pools" "${OUT_DIR}/evals" "${OUT_DIR}/logs" "${OUT_DIR}/status"

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

beta_tag() {
  case "$1" in
    0) echo "b0" ;;
    0.005) echo "b0005" ;;
    0.01) echo "b001" ;;
    0.02) echo "b002" ;;
    0.05) echo "b005" ;;
    *) echo "b${1//./p}" ;;
  esac
}

existing_full_pool_path() {
  local dataset="$1"
  echo "${ROOT_DIR}/run_logs/agsto_qwen8b_nvembed_sync_full1000_20260505/pools/${dataset}_qwen8b_nvembed_sync_pool${POOL_K}_full1000.json"
}

local_pool_path() {
  local dataset="$1"
  echo "${OUT_DIR}/pools/${dataset}_qwen8b_nvembed_sync_pool${POOL_K}_limit100.json"
}

pool_path() {
  local dataset="$1"
  local existing
  existing="$(existing_full_pool_path "${dataset}")"
  if [[ -s "${existing}" ]]; then
    echo "${existing}"
  else
    local_pool_path "${dataset}"
  fi
}

export_pool() {
  local dataset="$1"
  local existing pool_json status_path log_path
  existing="$(existing_full_pool_path "${dataset}")"
  pool_json="$(local_pool_path "${dataset}")"
  status_path="${OUT_DIR}/status/${dataset}_export.status"
  log_path="${OUT_DIR}/logs/${dataset}_export.log"

  if [[ -s "${existing}" ]]; then
    log_msg "SKIP export dataset=${dataset} using_existing_full1000_pool=${existing}"
    echo "[SKIP] dataset=${dataset} stage=export existing_full1000_pool=${existing} time=$(date -Is)" > "${status_path}"
    return 0
  fi
  if [[ -s "${pool_json}" ]]; then
    log_msg "SKIP export dataset=${dataset} existing=${pool_json}"
    echo "[SKIP] dataset=${dataset} stage=export existing=${pool_json} time=$(date -Is)" > "${status_path}"
    return 0
  fi

  log_msg "START export dataset=${dataset} limit=${LIMIT}"
  echo "[START] dataset=${dataset} stage=export limit=${LIMIT} time=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/export_agsto_pool.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --pool_k "${POOL_K}" \
      --openie_template "outputs_step0_general_nvembed_{dataset}/openie_results_ner_qwen3-8b.json" \
      --retrieval_top_k 20 \
      --candidate_limit 120 \
      --proposal_candidate_depth 10 \
      --support_proposal_depth 6 \
      --beam_size 12 \
      --native_dense_anchor true \
      --native_dense_anchor_top_k 20 \
      --chunk_embedding_template "outputs_step0_general_nvembed_{dataset}/qwen3-8b_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --embedding_model "${EMBEDDING_API_MODEL}" \
      --embedding_batch_size 8 \
      --dense_query_instruction_mode raw \
      --semantic_residual_weight 12.0 \
      --output_json "${pool_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]] && [[ -s "${pool_json}" ]]; then
    echo "[DONE] dataset=${dataset} stage=export time=$(date -Is)" > "${status_path}"
    log_msg "DONE export dataset=${dataset} output=${pool_json}"
  else
    echo "[FAILED] dataset=${dataset} stage=export code=${code} time=$(date -Is)" > "${status_path}"
    log_msg "FAILED export dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

eval_beta() {
  local dataset="$1"
  local port="$2"
  local beta="$3"
  local tag pool_json output_json cache_path status_path log_path
  tag="$(beta_tag "${beta}")"
  pool_json="$(pool_path "${dataset}")"
  output_json="${OUT_DIR}/evals/${dataset}_qwen8b_nvembed_graphprior_${tag}_limit100.json"
  cache_path="${OUT_DIR}/evals/${dataset}_qwen8b_nvembed_graphprior_shared.binding_cache.json"
  status_path="${OUT_DIR}/status/${dataset}_graphprior_${tag}.status"
  log_path="${OUT_DIR}/logs/${dataset}_graphprior_${tag}.log"

  if [[ -s "${output_json}" ]]; then
    log_msg "SKIP eval dataset=${dataset} beta=${beta} existing=${output_json}"
    echo "[SKIP] dataset=${dataset} stage=eval beta=${beta} existing=${output_json} time=$(date -Is)" > "${status_path}"
    return 0
  fi
  if [[ ! -s "${pool_json}" ]]; then
    log_msg "FAILED eval dataset=${dataset} beta=${beta} missing_pool=${pool_json}"
    echo "[FAILED] dataset=${dataset} stage=eval beta=${beta} missing_pool=${pool_json} time=$(date -Is)" > "${status_path}"
    return 2
  fi

  log_msg "START eval dataset=${dataset} beta=${beta} limit=${LIMIT} port=${port}"
  echo "[START] dataset=${dataset} stage=eval beta=${beta} limit=${LIMIT} port=${port} time=$(date -Is)" > "${status_path}"
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
      --external_pool_source_name "agsto_qwen8b_nvembed_sync_pool${POOL_K}_graphprior" \
      --external_pool_strict_questions true \
      --setwise_selector daec_noisyor_llm_agsto \
      --setwise_pool_k "${POOL_K}" \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --dtc_decomposition_mode llm \
      --dtc_binding_max_candidates 5 \
      --llm_binding_url "http://localhost:${port}/v1" \
      --llm_binding_model "${LLM_MODEL}" \
      --llm_binding_cache_path "${cache_path}" \
      --llm_binding_title_match_mode "${MATCH_MODE}" \
      --daec_graph_prior_beta "${beta}" \
      --daec_graph_prior_w_selected 1.0 \
      --daec_graph_prior_w_anchor 0.3 \
      --daec_graph_prior_w_rank 0.2 \
      --causal_enabled false \
      --causal_engine_version v2 \
      --causal_v2_base_retrieval_mode dense \
      --structure_rerank_enabled false \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]] && [[ -s "${output_json}" ]]; then
    echo "[DONE] dataset=${dataset} stage=eval beta=${beta} time=$(date -Is)" > "${status_path}"
    log_msg "DONE eval dataset=${dataset} beta=${beta} output=${output_json}"
  else
    echo "[FAILED] dataset=${dataset} stage=eval beta=${beta} code=${code} time=$(date -Is)" > "${status_path}"
    log_msg "FAILED eval dataset=${dataset} beta=${beta} code=${code}"
  fi
  return "${code}"
}

run_dataset() {
  local dataset="$1"
  local port beta
  port="$(dataset_port "${dataset}")" || return 2
  export_pool "${dataset}" || return 1
  for beta in "${BETAS[@]}"; do
    eval_beta "${dataset}" "${port}" "${beta}" || return 1
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
betas = ["0", "0.005", "0.01", "0.02", "0.05"]
tags = {"0": "b0", "0.005": "b0005", "0.01": "b001", "0.02": "b002", "0.05": "b005"}
rows = []

def metric(payload, key, default=0.0):
    try:
        return float(payload.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)

for dataset in datasets:
    for beta in betas:
        path = out_dir / "evals" / f"{dataset}_qwen8b_nvembed_graphprior_{tags[beta]}_limit100.json"
        if not path.exists():
            rows.append({
                "dataset": dataset,
                "beta": beta,
                "status": "missing",
            })
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        qa = data.get("setwise_selector_qa") or {}
        selector_metrics = qa.get("selector_retrieval_metrics") or {}
        pool_metrics = data.get("overall_recomputed") or {}
        query_traces = data.get("setwise_selector_query_traces") or []
        anchor_query_count = 0
        anchor_replaced_count = 0
        for row in query_traces:
            trace = row.get("selector_trace") or {}
            prior = trace.get("agsto_graph_prior") or {}
            anchor_positions = list(prior.get("selected_positions") or [])[:2]
            if not anchor_positions:
                continue
            final_positions = set(trace.get("final_front_pool_positions") or trace.get("selected_positions") or [])
            anchor_query_count += 1
            if any(int(pos) not in final_positions for pos in anchor_positions):
                anchor_replaced_count += 1
        rows.append({
            "dataset": dataset,
            "beta": beta,
            "status": "done",
            "selector_em": metric(qa, "selector_EM"),
            "selector_f1": metric(qa, "selector_F1"),
            "selector_r5": metric(selector_metrics, "Recall@5"),
            "pool_r100": metric(pool_metrics, "Recall@100"),
            "baseline_em": metric(qa, "baseline_EM"),
            "baseline_f1": metric(qa, "baseline_F1"),
            "anchor_replacement_rate": (
                anchor_replaced_count / max(anchor_query_count, 1)
                if anchor_query_count else 0.0
            ),
            "anchor_query_count": anchor_query_count,
        })

csv_path = out_dir / "agsto_graphprior_limit100_results.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    fieldnames = [
        "dataset", "beta", "status", "selector_em", "selector_f1",
        "selector_r5", "pool_r100", "baseline_em", "baseline_f1",
        "anchor_replacement_rate", "anchor_query_count",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

done_rows = [row for row in rows if row.get("status") == "done"]
md_lines = [
    "# AG-STO 8B + DAEC Graph Prior Limit100",
    "",
    "Selector: `daec_noisyor_llm_agsto`; beta=0 is the compatibility point for pure `daec_noisyor_llm` selection.",
    "",
    "```text",
    "+------------------+--------+--------+--------+--------+---------+--------+--------+---------+",
    "| Dataset          | beta   | EM     | F1     | R@5    | PoolR100| BaseEM | BaseF1 | AnchRep |",
    "+------------------+--------+--------+--------+--------+---------+--------+--------+---------+",
]
for row in done_rows:
    md_lines.append(
        "| {dataset:<16} | {beta:<6} | {em:>6.3f} | {f1:>6.3f} | {r5:>6.3f} | {r100:>7.3f} | {bem:>6.3f} | {bf1:>6.3f} | {arep:>7.3f} |".format(
            dataset=str(row["dataset"]),
            beta=str(row["beta"]),
            em=float(row["selector_em"]),
            f1=float(row["selector_f1"]),
            r5=float(row["selector_r5"]),
            r100=float(row["pool_r100"]),
            bem=float(row["baseline_em"]),
            bf1=float(row["baseline_f1"]),
            arep=float(row["anchor_replacement_rate"]),
        )
    )
md_lines.extend([
    "+------------------+--------+--------+--------+--------+---------+--------+--------+---------+",
    "```",
    "",
    f"CSV: `{csv_path}`",
])
(out_dir / "summary_graphprior_limit100.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print(f"[SUMMARY] wrote {csv_path}")
PY
}

main() {
  echo "[START] agsto_daec_graphprior_limit100 time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
  log_msg "AG-STO Qwen3-8B + DAEC graph-prior limit100 begin"
  log_msg "lanes: musique->8041 hotpotqa->8042 2wikimultihopqa->8043"
  log_msg "betas: ${BETAS[*]}"

  (run_dataset musique) &
  local pid_musique=$!
  (run_dataset hotpotqa) &
  local pid_hotpotqa=$!
  (run_dataset 2wikimultihopqa) &
  local pid_2wiki=$!

  local status=0
  if ! wait "${pid_musique}"; then
    log_msg "FAILED dataset=musique"
    status=1
  fi
  if ! wait "${pid_hotpotqa}"; then
    log_msg "FAILED dataset=hotpotqa"
    status=1
  fi
  if ! wait "${pid_2wiki}"; then
    log_msg "FAILED dataset=2wikimultihopqa"
    status=1
  fi

  summarize_results >> "${OUT_DIR}/logs/launcher.log" 2>&1 || status=1

  if [[ "${status}" -eq 0 ]]; then
    echo "[DONE] agsto_daec_graphprior_limit100 time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
    log_msg "AG-STO Qwen3-8B + DAEC graph-prior limit100 complete"
  else
    echo "[FAILED] agsto_daec_graphprior_limit100 status=${status} time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
    log_msg "AG-STO Qwen3-8B + DAEC graph-prior limit100 finished with failures"
  fi
  return "${status}"
}

main "$@"
