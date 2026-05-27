#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
RUN_TAG="hipporag_qwen32b_valid_graph_top200_full1000_20260513_r2"
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

HIPPO_SAVE_DIR="${HIPPO_SAVE_DIR:-${ROOT_DIR}/outputs_hipporag_qwen32b_valid_graph_top200_20260513_r2}"
KEY_TMUX_SESSION="${KEY_TMUX_SESSION:-baseline_qwen32b_top200_20260512}"

mkdir -p \
  "${OUT_DIR}/logs" \
  "${OUT_DIR}/status" \
  "${OUT_DIR}/pools/hipporag" \
  "${OUT_DIR}/reader_inputs/hipporag" \
  "${OUT_DIR}/reader_qa/hipporag" \
  "${OUT_DIR}/reader_runtime/hipporag"

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

hydrate_openai_key() {
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    return 0
  fi
  local key_line
  key_line="$(tmux show-environment -t "${KEY_TMUX_SESSION}" OPENAI_API_KEY 2>/dev/null || true)"
  if [[ "${key_line}" == OPENAI_API_KEY=* ]]; then
    export OPENAI_API_KEY="${key_line#OPENAI_API_KEY=}"
  fi
}

preflight() {
  local rc=0
  check_endpoint "qwen32b" "${GRAPH_LLM_BASE_URL%/}/models" || rc=1
  check_endpoint "embedding" "${EMBEDDING_BASE_URL%/embeddings}/models" || rc=1
  hydrate_openai_key
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    log_msg "FAILED missing OPENAI_API_KEY for reader; graph exports need local endpoints only, but full E2E reader cannot run."
    rc=1
  else
    log_msg "OK reader key available"
  fi
  return "${rc}"
}

embedding_slug() {
  local value="$1"
  printf '%s' "${value//\//_}"
}

artifact_root() {
  local dataset="$1"
  printf '%s_%s/%s_%s' "${HIPPO_SAVE_DIR}" "${dataset}" "${GRAPH_LLM_NAME}" "$(embedding_slug "${EMBEDDING_NAME}")"
}

pool_path() {
  local dataset="$1"
  printf '%s/pools/hipporag/%s_hipporag_qwen32b_valid_graph_pool%s.json' "${OUT_DIR}" "${dataset}" "${POOL_K}"
}

reader_input_path() {
  local dataset="$1"
  printf '%s/reader_inputs/hipporag/%s_hipporag_qwen32b_valid_graph_pool%s_reader_input.json' "${OUT_DIR}" "${dataset}" "${POOL_K}"
}

reader_report_path() {
  local dataset="$1"
  printf '%s/reader_qa/hipporag/%s_hipporag_qwen32b_valid_graph_top%s_gpt4omini_reader_top%s.json' "${OUT_DIR}" "${dataset}" "${POOL_K}" "${QA_TOP_K}"
}

reader_report_md_path() {
  local dataset="$1"
  printf '%s/reader_qa/hipporag/%s_hipporag_qwen32b_valid_graph_top%s_gpt4omini_reader_top%s.md' "${OUT_DIR}" "${dataset}" "${POOL_K}" "${QA_TOP_K}"
}

verify_hipporag_export() {
  local dataset="$1"
  local log_path="$2"
  local output_json="$3"
  local model_root
  model_root="$(artifact_root "${dataset}")"

  if [[ ! -s "${output_json}" ]]; then
    log_msg "FAILED verify dataset=${dataset}: missing pool JSON"
    return 1
  fi
  if grep -q "Falling back to dense ranking" "${log_path}"; then
    log_msg "FAILED verify dataset=${dataset}: dense fallback warning found"
    return 1
  fi
  if [[ ! -s "${model_root}/graph.pickle" ]]; then
    log_msg "FAILED verify dataset=${dataset}: missing graph.pickle under ${model_root}"
    return 1
  fi
  if [[ ! -d "${model_root}/entity_embeddings" || ! -d "${model_root}/fact_embeddings" ]]; then
    log_msg "FAILED verify dataset=${dataset}: missing entity/fact embedding stores"
    return 1
  fi
  if ! compgen -G "${HIPPO_SAVE_DIR}_${dataset}/openie_results_ner_*.json" >/dev/null; then
    log_msg "FAILED verify dataset=${dataset}: missing OpenIE results"
    return 1
  fi
  return 0
}

export_hipporag_dataset() {
  local dataset="$1"
  local output_json log_path
  output_json="$(pool_path "${dataset}")"
  log_path="${OUT_DIR}/logs/export_hipporag_${dataset}.log"
  write_status "export_hipporag_${dataset}" "START dataset=${dataset} output=${output_json}"
  log_msg "START HippoRAG valid graph export dataset=${dataset}"
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
  if [[ "${code}" -eq 0 ]] && verify_hipporag_export "${dataset}" "${log_path}" "${output_json}"; then
    write_status "export_hipporag_${dataset}" "DONE output=${output_json}"
    log_msg "DONE HippoRAG valid graph export dataset=${dataset}"
  else
    write_status "export_hipporag_${dataset}" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED HippoRAG valid graph export dataset=${dataset} code=${code}"
    return 1
  fi
}

run_exports_parallel() {
  local pids=()
  local dataset
  for dataset in ${DATASETS}; do
    export_hipporag_dataset "${dataset}" &
    pids+=("$!")
  done
  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  return "${rc}"
}

wrap_reader_input() {
  local dataset="$1"
  local pool_json input_json log_path
  pool_json="$(pool_path "${dataset}")"
  input_json="$(reader_input_path "${dataset}")"
  log_path="${OUT_DIR}/logs/wrap_hipporag_${dataset}.log"
  write_status "wrap_hipporag_${dataset}" "START pool=${pool_json}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" scripts/make_external_pool_reader_inputs.py \
      --pool-json "${pool_json}" \
      --method-name "hipporag_qwen32b_valid_graph_top${POOL_K}" \
      --output-json "${input_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${input_json}" ]]; then
    write_status "wrap_hipporag_${dataset}" "DONE input=${input_json}"
  else
    write_status "wrap_hipporag_${dataset}" "FAILED code=${code} log=${log_path}"
    return "${code}"
  fi
}

run_reader_dataset() {
  local dataset="$1"
  local input_json output_json output_md log_path
  input_json="$(reader_input_path "${dataset}")"
  output_json="$(reader_report_path "${dataset}")"
  output_md="$(reader_report_md_path "${dataset}")"
  log_path="${OUT_DIR}/logs/reader_hipporag_${dataset}.log"
  write_status "reader_hipporag_${dataset}" "START input=${input_json} output=${output_json}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py \
      --input-files "${input_json}" \
      --method-name "hipporag_qwen32b_valid_graph_top${POOL_K}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --doc-source external_pool_topk_docs \
      --save-dir "${OUT_DIR}/reader_runtime/hipporag/${dataset}" \
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
    write_status "reader_hipporag_${dataset}" "DONE output=${output_json}"
    log_msg "DONE reader dataset=${dataset}"
  else
    write_status "reader_hipporag_${dataset}" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED reader dataset=${dataset} code=${code}"
    return "${code}"
  fi
}

run_readers_parallel() {
  local pids=()
  local dataset
  for dataset in ${DATASETS}; do
    run_reader_dataset "${dataset}" &
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
  env RUN_OUT_DIR="${OUT_DIR}" POOL_K="${POOL_K}" QA_TOP_K="${QA_TOP_K}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["RUN_OUT_DIR"])
rows = []
for path in sorted((out_dir / "reader_qa" / "hipporag").glob("*.json")):
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
                "AllGold@5": metrics.get("AllGold@5") or metrics.get("all_gold_at_5"),
                "ExactMatch": metrics.get("ExactMatch"),
                "F1": metrics.get("F1"),
                "report": str(path),
            })

summary_json = out_dir / "summary.json"
summary_json.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n")
lines = [
    "# HippoRAG Qwen32B Valid Graph Top200 + GPT-4o-mini Reader Top5",
    "",
    "| Dataset | Method | Count | R@5 | R@20 | R@100 | R@200 | All@5 | EM | F1 |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
def fmt(value):
    return "" if value is None else f"{float(value):.4f}"
for row in rows:
    lines.append(
        f"| {row['dataset']} | {row['method']} | {int(row['count'] or 0)} | "
        f"{fmt(row['Recall@5'])} | {fmt(row['Recall@20'])} | {fmt(row['Recall@100'])} | "
        f"{fmt(row['Recall@200'])} | {fmt(row['AllGold@5'])} | {fmt(row['ExactMatch'])} | {fmt(row['F1'])} |"
    )
(out_dir / "summary.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"summary_json": str(summary_json), "rows": len(rows)}, indent=2))
PY
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: HippoRAG legacy graph build with ${GRAPH_LLM_NAME} /no_think, pool=${POOL_K}, reader=${READER_LLM_NAME}, reader_top=${QA_TOP_K}"
  preflight || {
    write_status "launcher" "FAILED preflight"
    return 1
  }

  run_exports_parallel || {
    write_status "launcher" "FAILED export stage"
    return 1
  }

  local dataset
  for dataset in ${DATASETS}; do
    wrap_reader_input "${dataset}" || {
      write_status "launcher" "FAILED reader input dataset=${dataset}"
      return 1
    }
  done

  run_readers_parallel || {
    write_status "launcher" "FAILED reader stage"
    return 1
  }

  write_summary
  write_status "launcher" "DONE run_tag=${RUN_TAG}"
  log_msg "DONE ${RUN_TAG}"
}

main "$@"
