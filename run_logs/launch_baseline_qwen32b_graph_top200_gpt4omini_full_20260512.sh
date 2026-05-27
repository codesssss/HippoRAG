#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
RUN_TAG="baseline_qwen32b_graph_top200_gpt4omini_full_20260512"
OUT_DIR="${ROOT_DIR}/run_logs/${RUN_TAG}"

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
POOL_K="${POOL_K:-200}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-200}"
QA_TOP_K="${QA_TOP_K:-5}"

GRAPH_LLM_NAME="${GRAPH_LLM_NAME:-qwen3-32b-judge}"
GRAPH_LLM_BASE_URL="${GRAPH_LLM_BASE_URL:-http://localhost:8045/v1}"
READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
OPENIE_MAX_NEW_TOKENS="${OPENIE_MAX_NEW_TOKENS:-2048}"

HIPPO_SAVE_DIR="${HIPPO_SAVE_DIR:-${ROOT_DIR}/outputs_hipporag_qwen32b_nothink_top200_20260512}"
PROPRAG_SAVE_DIR="${PROPRAG_SAVE_DIR:-/mnt/nvme/code/PropRAG/outputs_proprag_qwen32b_nothink_top200_20260512}"

mkdir -p \
  "${OUT_DIR}/logs" \
  "${OUT_DIR}/status" \
  "${OUT_DIR}/pools/hipporag" \
  "${OUT_DIR}/pools/proprag" \
  "${OUT_DIR}/reader_inputs/hipporag" \
  "${OUT_DIR}/reader_inputs/proprag" \
  "${OUT_DIR}/reader_qa/hipporag" \
  "${OUT_DIR}/reader_qa/proprag" \
  "${OUT_DIR}/reader_runtime"

export HIPPORAG_RERANK_FORCE_NO_THINK=1
export TOKENIZERS_PARALLELISM=false

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_DIR}/logs/launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${OUT_DIR}/status/${name}.status"
}

check_endpoint() {
  local name="$1"
  local url="$2"
  if curl -fsS --max-time 10 "${url}" >/dev/null; then
    log_msg "OK endpoint ${name}: ${url}"
    return 0
  fi
  log_msg "FAILED endpoint ${name}: ${url}"
  return 1
}

preflight() {
  local rc=0
  check_endpoint "qwen32b" "${GRAPH_LLM_BASE_URL%/}/models" || rc=1
  check_endpoint "embedding" "${EMBEDDING_BASE_URL%/embeddings}/models" || rc=1
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    log_msg "FAILED missing OPENAI_API_KEY for GPT-4o-mini reader"
    rc=1
  fi
  return "${rc}"
}

pool_path() {
  local method="$1"
  local dataset="$2"
  echo "${OUT_DIR}/pools/${method}/${dataset}_${method}_qwen32b_nothink_pool${POOL_K}.json"
}

reader_input_path() {
  local method="$1"
  local dataset="$2"
  echo "${OUT_DIR}/reader_inputs/${method}/${dataset}_${method}_qwen32b_nothink_pool${POOL_K}_reader_input.json"
}

reader_report_path() {
  local method="$1"
  local dataset="$2"
  echo "${OUT_DIR}/reader_qa/${method}/${dataset}_${method}_qwen32b_nothink_top${POOL_K}_gpt4omini_reader_top${QA_TOP_K}.json"
}

reader_report_md_path() {
  local method="$1"
  local dataset="$2"
  echo "${OUT_DIR}/reader_qa/${method}/${dataset}_${method}_qwen32b_nothink_top${POOL_K}_gpt4omini_reader_top${QA_TOP_K}.md"
}

export_hipporag_dataset() {
  local dataset="$1"
  local output_json
  output_json="$(pool_path hipporag "${dataset}")"
  local log_path="${OUT_DIR}/logs/export_hipporag_${dataset}.log"
  if [[ -s "${output_json}" ]]; then
    write_status "export_hipporag_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP HippoRAG Qwen32B graph export dataset=${dataset}"
    return 0
  fi
  write_status "export_hipporag_${dataset}" "START dataset=${dataset} output=${output_json}"
  log_msg "START HippoRAG Qwen32B graph export dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" scripts/export_hipporag_pool.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --pool_k "${POOL_K}" \
      --save_dir "${HIPPO_SAVE_DIR}" \
      --llm_name "${GRAPH_LLM_NAME}" \
      --llm_request_name "${GRAPH_LLM_NAME}" \
      --llm_base_url "${GRAPH_LLM_BASE_URL}" \
      --embedding_name "${EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
      --max_new_tokens "${OPENIE_MAX_NEW_TOKENS}" \
      --retrieval_top_k "${RETRIEVAL_TOP_K}" \
      --force_index_from_scratch true \
      --force_openie_from_scratch true \
      --openie_mode online \
      --qwen_disable_thinking \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "export_hipporag_${dataset}" "DONE output=${output_json}"
    log_msg "DONE HippoRAG Qwen32B graph export dataset=${dataset}"
  else
    write_status "export_hipporag_${dataset}" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED HippoRAG Qwen32B graph export dataset=${dataset} code=${code}"
    return "${code}"
  fi
}

export_proprag_dataset() {
  local dataset="$1"
  local output_json
  output_json="$(pool_path proprag "${dataset}")"
  local log_path="${OUT_DIR}/logs/export_proprag_${dataset}.log"
  if [[ -s "${output_json}" ]]; then
    write_status "export_proprag_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP PropRAG Qwen32B graph export dataset=${dataset}"
    return 0
  fi
  write_status "export_proprag_${dataset}" "START dataset=${dataset} output=${output_json}"
  log_msg "START PropRAG Qwen32B graph export dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" scripts/export_proprag_pool.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --pool_k "${POOL_K}" \
      --llm_name "${GRAPH_LLM_NAME}" \
      --llm_base_url "${GRAPH_LLM_BASE_URL}" \
      --embedding_name "${EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
      --save_dir "${PROPRAG_SAVE_DIR}" \
      --retrieval_top_k "${RETRIEVAL_TOP_K}" \
      --qa_top_k "${QA_TOP_K}" \
      --max_new_tokens "${OPENIE_MAX_NEW_TOKENS}" \
      --reuse_preextracted_openie false \
      --openie_llm_name "${GRAPH_LLM_NAME}" \
      --force_index_from_scratch true \
      --force_openie_from_scratch true \
      --openie_mode online \
      --use_propositions true \
      --use_beam_search true \
      --beam_width 4 \
      --max_path_length 3 \
      --second_stage_filter_k 40 \
      --sim_threshold 0.75 \
      --qwen_disable_thinking \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "export_proprag_${dataset}" "DONE output=${output_json}"
    log_msg "DONE PropRAG Qwen32B graph export dataset=${dataset}"
  else
    write_status "export_proprag_${dataset}" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED PropRAG Qwen32B graph export dataset=${dataset} code=${code}"
    return "${code}"
  fi
}

wrap_reader_input() {
  local method="$1"
  local dataset="$2"
  local pool_json input_json log_path
  pool_json="$(pool_path "${method}" "${dataset}")"
  input_json="$(reader_input_path "${method}" "${dataset}")"
  log_path="${OUT_DIR}/logs/wrap_${method}_${dataset}.log"
  if [[ -s "${input_json}" ]]; then
    write_status "wrap_${method}_${dataset}" "SKIP existing input=${input_json}"
    return 0
  fi
  write_status "wrap_${method}_${dataset}" "START pool=${pool_json}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" scripts/make_external_pool_reader_inputs.py \
      --pool-json "${pool_json}" \
      --method-name "${method}_qwen32b_top${POOL_K}" \
      --output-json "${input_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${input_json}" ]]; then
    write_status "wrap_${method}_${dataset}" "DONE input=${input_json}"
  else
    write_status "wrap_${method}_${dataset}" "FAILED code=${code} log=${log_path}"
    return "${code}"
  fi
}

run_reader_dataset() {
  local method="$1"
  local dataset="$2"
  local input_json output_json output_md log_path
  input_json="$(reader_input_path "${method}" "${dataset}")"
  output_json="$(reader_report_path "${method}" "${dataset}")"
  output_md="$(reader_report_md_path "${method}" "${dataset}")"
  log_path="${OUT_DIR}/logs/reader_${method}_${dataset}.log"
  if [[ -s "${output_json}" ]]; then
    write_status "reader_${method}_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP reader method=${method} dataset=${dataset}"
    return 0
  fi
  write_status "reader_${method}_${dataset}" "START input=${input_json} output=${output_json}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py \
      --input-files "${input_json}" \
      --method-name "${method}_qwen32b_top${POOL_K}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --doc-source external_pool_topk_docs \
      --save-dir "${OUT_DIR}/reader_runtime/${method}/${dataset}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens none \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${output_json}" \
      --output-md "${output_md}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "reader_${method}_${dataset}" "DONE output=${output_json}"
    log_msg "DONE reader method=${method} dataset=${dataset}"
  else
    write_status "reader_${method}_${dataset}" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED reader method=${method} dataset=${dataset} code=${code}"
    return "${code}"
  fi
}

run_reader_method_parallel() {
  local method="$1"
  local pids=()
  local dataset
  for dataset in ${DATASETS}; do
    run_reader_dataset "${method}" "${dataset}" &
    pids+=("$!")
  done
  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  return "${rc}"
}

write_summary() {
  local summary_json="${OUT_DIR}/summary.json"
  local summary_md="${OUT_DIR}/summary.md"
  env RUN_OUT_DIR="${OUT_DIR}" POOL_K="${POOL_K}" QA_TOP_K="${QA_TOP_K}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["RUN_OUT_DIR"])
rows = []
for method in ("hipporag", "proprag"):
    for path in sorted((out_dir / "reader_qa" / method).glob("*.json")):
        payload = json.loads(path.read_text())
        for dataset in payload.get("datasets", []):
            for method_name, method_payload in (dataset.get("methods") or {}).items():
                metrics = method_payload.get("metrics") or {}
                rows.append({
                    "dataset": dataset.get("dataset"),
                    "method": method_name,
                    "count": metrics.get("count"),
                    "Recall@5": metrics.get("Recall@5") or metrics.get("r5"),
                    "Recall@20": metrics.get("Recall@20"),
                    "Recall@100": metrics.get("Recall@100"),
                    "Recall@200": metrics.get("Recall@200"),
                    "EM": metrics.get("ExactMatch"),
                    "F1": metrics.get("F1"),
                    "reader_docs": metrics.get("mean_reader_docs"),
                    "report": str(path),
                })
summary_json = out_dir / "summary.json"
summary_json.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n")
lines = [
    "# Qwen32B Baseline Graph Top200 + GPT-4o-mini Reader Top5",
    "",
    "| Dataset | Method | Count | R@5 | R@20 | R@100 | R@200 | EM | F1 |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    def fmt(value):
        return "" if value is None else f"{float(value):.4f}"
    lines.append(
        f"| {row['dataset']} | {row['method']} | {int(row['count'] or 0)} | "
        f"{fmt(row['Recall@5'])} | {fmt(row['Recall@20'])} | {fmt(row['Recall@100'])} | "
        f"{fmt(row['Recall@200'])} | {fmt(row['EM'])} | {fmt(row['F1'])} |"
    )
(out_dir / "summary.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"summary_json": str(summary_json), "rows": len(rows)}, indent=2))
PY
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: HippoRAG/PropRAG native graph build with ${GRAPH_LLM_NAME} /no_think, pool=${POOL_K}, reader=${READER_LLM_NAME}, reader_top=${QA_TOP_K}"
  preflight || {
    write_status "launcher" "FAILED preflight"
    return 1
  }

  local dataset
  for dataset in ${DATASETS}; do
    export_hipporag_dataset "${dataset}" || {
      write_status "launcher" "FAILED export_hipporag dataset=${dataset}"
      return 1
    }
    wrap_reader_input hipporag "${dataset}" || return 1
  done

  for dataset in ${DATASETS}; do
    export_proprag_dataset "${dataset}" || {
      write_status "launcher" "FAILED export_proprag dataset=${dataset}"
      return 1
    }
    wrap_reader_input proprag "${dataset}" || return 1
  done

  run_reader_method_parallel hipporag || {
    write_status "launcher" "FAILED reader hipporag"
    return 1
  }
  run_reader_method_parallel proprag || {
    write_status "launcher" "FAILED reader proprag"
    return 1
  }
  write_summary
  write_status "launcher" "DONE run_tag=${RUN_TAG}"
  log_msg "DONE ${RUN_TAG}"
}

main "$@"
