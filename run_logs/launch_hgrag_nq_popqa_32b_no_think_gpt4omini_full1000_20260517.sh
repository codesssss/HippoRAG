#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/mnt/nvme/code/HippoRAG}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
HGRAG_ROOT="${HGRAG_ROOT:-/mnt/nvme/code/HGRAG}"
RUN_TAG="${RUN_TAG:-hgrag_nq_popqa_32b_no_think_gpt4omini_full1000_20260517}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/run_logs/${RUN_TAG}}"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

DATASETS="${DATASETS:-nq_rear popqa}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
QA_TOP_K="${QA_TOP_K:-5}"
RECALL_TOP_K="${RECALL_TOP_K:-20}"
E2E_TOP_K="${E2E_TOP_K:-20}"
ENT_TOPK="${ENT_TOPK:-1}"
BETA="${BETA:-0.5}"
STEP="${STEP:-2}"
HGRAG_MAX_WORKERS="${HGRAG_MAX_WORKERS:-4}"
HGRAG_EMBEDDING_BATCH_SIZE="${HGRAG_EMBEDDING_BATCH_SIZE:-16}"
HGRAG_NER_MAX_NEW_TOKENS="${HGRAG_NER_MAX_NEW_TOKENS:-512}"

GRAPH_LLM_NAME="${GRAPH_LLM_NAME:-qwen3-32b-judge}"
QWEN32_URL_A="${QWEN32_URL_A:-http://localhost:8045/v1}"
QWEN32_URL_B="${QWEN32_URL_B:-http://localhost:8046/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
HGRAG_EMBEDDING_NAME="${HGRAG_EMBEDDING_NAME:-nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
READER_MAX_NEW_TOKENS="${READER_MAX_NEW_TOKENS:-none}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/status" "${OUT_ROOT}/retrieval" \
  "${OUT_ROOT}/reader_inputs/hgrag" "${OUT_ROOT}/reader_qa/hgrag" \
  "${OUT_ROOT}/reader_runtime/hgrag"

if [[ -f "${SECRET_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRET_ENV}"
  set +a
fi

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/logs/launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee "${OUT_ROOT}/status/${name}.status"
}

check_endpoint() {
  local name="$1"
  local url="$2"
  if curl -fsS --max-time 10 "${url}" >/dev/null 2>&1; then
    log_msg "OK endpoint ${name}: ${url}"
    return 0
  fi
  log_msg "FAILED endpoint ${name}: ${url}"
  return 1
}

display_dataset_name() {
  case "$1" in
    nq|nq_rear) printf 'nq\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

evidence_dataset_name() {
  case "$1" in
    nq) printf 'nq_rear\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

qwen_url_for_dataset() {
  case "$(display_dataset_name "$1")" in
    nq) printf '%s\n' "${QWEN32_URL_A}" ;;
    popqa) printf '%s\n' "${QWEN32_URL_B}" ;;
    *) printf '%s\n' "${QWEN32_URL_A}" ;;
  esac
}

hgrag_retrieval_path() {
  local dataset="$1"
  local display
  display="$(display_dataset_name "${dataset}")"
  printf '%s/retrieval/%s_hgrag_qwen32b_no_think_retrieval.json\n' "${OUT_ROOT}" "${display}"
}

hgrag_reader_input_path() {
  local dataset="$1"
  local display
  display="$(display_dataset_name "${dataset}")"
  printf '%s/reader_inputs/hgrag/%s_hgrag_qwen32b_no_think_reader_input.json\n' "${OUT_ROOT}" "${display}"
}

hgrag_reader_report_path() {
  local dataset="$1"
  local display
  display="$(display_dataset_name "${dataset}")"
  printf '%s/reader_qa/hgrag/%s_hgrag_qwen32b_no_think_gpt4omini_reader.json\n' "${OUT_ROOT}" "${display}"
}

hgrag_reader_report_md_path() {
  local dataset="$1"
  local display
  display="$(display_dataset_name "${dataset}")"
  printf '%s/reader_qa/hgrag/%s_hgrag_qwen32b_no_think_gpt4omini_reader.md\n' "${OUT_ROOT}" "${display}"
}

is_hgrag_retrieval_done() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  "${PYTHON_BIN}" - "${path}" "${MAX_QUERIES}" <<'PY'
import json
import sys
path = sys.argv[1]
expected = int(sys.argv[2])
try:
    payload = json.load(open(path, encoding="utf-8"))
except Exception:
    sys.exit(1)
ok = (
    payload.get("method") == "hgrag"
    and int(payload.get("num_queries") or 0) == expected
    and bool(payload.get("overall_retrieval_result"))
    and payload.get("overall_qa_results", {}).get("skipped") is True
)
sys.exit(0 if ok else 1)
PY
}

run_logged() {
  local desc="$1"
  local status_name="$2"
  local log_path="$3"
  shift 3
  write_status "${status_name}" "START ${desc} log=${log_path}"
  log_msg "START ${desc}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "$@"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    write_status "${status_name}" "DONE ${desc}"
    log_msg "DONE ${desc}"
  else
    write_status "${status_name}" "FAILED ${desc} code=${code} log=${log_path}"
    log_msg "FAILED ${desc} code=${code} log=${log_path}"
  fi
  return "${code}"
}

run_hgrag_retrieval_dataset() {
  local dataset="$1"
  local display llm_base_url output_json log_path tag
  dataset="$(evidence_dataset_name "${dataset}")"
  display="$(display_dataset_name "${dataset}")"
  llm_base_url="$(qwen_url_for_dataset "${dataset}")"
  output_json="$(hgrag_retrieval_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/hgrag_retrieval_${display}.log"
  tag="aligned_limit${MAX_QUERIES}_${GRAPH_LLM_NAME}_nothink_gpt4omini_protocol_20260517"

  if is_hgrag_retrieval_done "${output_json}"; then
    write_status "hgrag_retrieval_${display}" "SKIP existing output=${output_json}"
    return 0
  fi

  run_logged "hgrag retrieval dataset=${display} evidence_dataset=${dataset}" \
    "hgrag_retrieval_${display}" "${log_path}" \
    env OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" "${PYTHON_BIN}" scripts/run_hgrag_aligned.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --hgrag_root "${HGRAG_ROOT}" \
      --output_json "${output_json}" \
      --output_method_name "${tag}" \
      --llm_name "${GRAPH_LLM_NAME}" \
      --llm_base_url "${llm_base_url}" \
      --embedding_name "${HGRAG_EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --qa_top_k "${QA_TOP_K}" \
      --recall_top_k "${RECALL_TOP_K}" \
      --e2e_top_k "${E2E_TOP_K}" \
      --ent_topk "${ENT_TOPK}" \
      --beta "${BETA}" \
      --step "${STEP}" \
      --embedding_batch_size "${HGRAG_EMBEDDING_BATCH_SIZE}" \
      --max_workers "${HGRAG_MAX_WORKERS}" \
      --hgraph_device cpu \
      --ner_max_new_tokens "${HGRAG_NER_MAX_NEW_TOKENS}" \
      --skip_qa
}

wrap_hgrag_reader_input() {
  local dataset="$1"
  local display source_json input_json log_path
  dataset="$(evidence_dataset_name "${dataset}")"
  display="$(display_dataset_name "${dataset}")"
  source_json="$(hgrag_retrieval_path "${dataset}")"
  input_json="$(hgrag_reader_input_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/wrap_hgrag_${display}.log"

  if [[ -s "${input_json}" ]]; then
    write_status "wrap_hgrag_${display}" "SKIP existing input=${input_json}"
    return 0
  fi

  write_status "wrap_hgrag_${display}" "START source=${source_json}"
env SOURCE_JSON="${source_json}" OUTPUT_JSON="${input_json}" DISPLAY_DATASET="${display}" "${PYTHON_BIN}" - <<'PY' > "${log_path}" 2>&1
import ast
import json
import os
from pathlib import Path

source = Path(os.environ["SOURCE_JSON"])
output = Path(os.environ["OUTPUT_JSON"])
display_dataset = os.environ["DISPLAY_DATASET"]
payload = json.loads(source.read_text(encoding="utf-8"))

def parse_answer_alias_values(value):
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                return parse_answer_alias_values(ast.literal_eval(cleaned))
            except (ValueError, SyntaxError):
                return [cleaned]
        return [cleaned]
    if isinstance(value, (list, tuple, set)):
        aliases = []
        for item in value:
            aliases.extend(parse_answer_alias_values(item))
        return aliases
    return [str(value)]

def normalize_gold_answers(value):
    return sorted({str(item).strip() for item in parse_answer_alias_values(value) if str(item).strip()})

examples = []
for idx, record in enumerate(payload.get("records", []) or []):
    docs = list(record.get("retrieved_docs", []) or [])[:5]
    examples.append({
        "query_index": int(record.get("query_idx", idx)),
        "question": str(record.get("question") or ""),
        "gold_answers": normalize_gold_answers(record.get("gold_answers", [])),
        "gold_doc_ids": list(record.get("gold_doc_ids", []) or []),
        "docs": docs,
        "retrieved_doc_ids": list(record.get("retrieved_doc_ids", []) or [])[:5],
        "retrieval_trace": {
            "source_output_json": str(source.resolve()),
            "source_method": "hgrag",
            "uses_saved_hgrag_docs": True,
            "built_in_hgrag_qa_skipped": payload.get("overall_qa_results", {}).get("skipped") is True,
        },
    })
metrics = payload.get("overall_retrieval_result", {}) or {}
output_payload = {
    "format": "hgrag_saved_docs_reader_input_v1",
    "dataset": display_dataset,
    "method": "hgrag",
    "source": str(source.resolve()),
    "limit": payload.get("limit"),
    "num_queries": len(examples),
    "overall_recomputed": metrics,
    "retrieval": {"recomputed_title_recall": metrics},
    "config": payload.get("config", {}),
    "examples": examples,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output_json": str(output), "count": len(examples)}, indent=2))
PY
  local code=$?
  if [[ "${code}" -eq 0 && -s "${input_json}" ]]; then
    write_status "wrap_hgrag_${display}" "DONE input=${input_json}"
  else
    write_status "wrap_hgrag_${display}" "FAILED code=${code} log=${log_path}"
  fi
  return "${code}"
}

run_hgrag_reader_dataset() {
  local dataset="$1"
  local display input_json output_json output_md log_path
  dataset="$(evidence_dataset_name "${dataset}")"
  display="$(display_dataset_name "${dataset}")"
  input_json="$(hgrag_reader_input_path "${dataset}")"
  output_json="$(hgrag_reader_report_path "${dataset}")"
  output_md="$(hgrag_reader_report_md_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/reader_hgrag_${display}.log"

  if [[ -s "${output_json}" ]]; then
    write_status "reader_hgrag_${display}" "SKIP existing output=${output_json}"
    return 0
  fi

  run_logged "reader hgrag dataset=${display}" "reader_hgrag_${display}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py \
      --input-files "${input_json}" \
      --method-name "hgrag_qwen32b_no_think_top${QA_TOP_K}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --doc-source saved_docs \
      --save-dir "${OUT_ROOT}/reader_runtime/hgrag/${display}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens "${READER_MAX_NEW_TOKENS}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${output_json}" \
      --output-md "${output_md}"
}

run_lane() {
  local dataset="$1"
  local display
  dataset="$(evidence_dataset_name "${dataset}")"
  display="$(display_dataset_name "${dataset}")"
  write_status "lane_${display}" "START endpoint=$(qwen_url_for_dataset "${dataset}")"
  run_hgrag_retrieval_dataset "${dataset}" || return 1
  wrap_hgrag_reader_input "${dataset}" || return 1
  run_hgrag_reader_dataset "${dataset}" || return 1
  write_status "lane_${display}" "DONE"
}

write_summary() {
  env OUT_ROOT="${OUT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
import os

out = Path(os.environ["OUT_ROOT"])
rows = []
for path in sorted((out / "reader_qa" / "hgrag").glob("*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for dataset in payload.get("datasets", []) or []:
        label = dataset.get("dataset")
        for method_name, method_payload in (dataset.get("methods") or {}).items():
            metrics = method_payload.get("metrics") or {}
            rows.append({
                "dataset": label,
                "method": method_name,
                "count": metrics.get("count"),
                "Recall@5": metrics.get("Recall@5") or metrics.get("r5"),
                "Recall@20": metrics.get("Recall@20"),
                "ExactMatch": metrics.get("ExactMatch"),
                "F1": metrics.get("F1"),
                "mean_reader_docs": metrics.get("mean_reader_docs"),
                "source_json": str(path),
            })
summary = {"rows": rows}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lines = ["| Dataset | Method | Count | R@5 | R@20 | EM | F1 | Reader docs |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
for row in rows:
    def fmt(value):
        return "" if value is None else f"{float(value):.4f}"
    lines.append(
        f"| {row['dataset']} | {row['method']} | {int(row.get('count') or 0)} | "
        f"{fmt(row.get('Recall@5'))} | {fmt(row.get('Recall@20'))} | "
        f"{fmt(row.get('ExactMatch'))} | {fmt(row.get('F1'))} | {fmt(row.get('mean_reader_docs'))} |"
    )
(out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG} datasets=${DATASETS}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: HGRAG retrieval uses ${GRAPH_LLM_NAME} no_think; reader uses ${READER_LLM_NAME}; datasets=${DATASETS}"

  local preflight=0
  check_endpoint "qwen32_a" "${QWEN32_URL_A}/models" || preflight=1
  check_endpoint "qwen32_b" "${QWEN32_URL_B}/models" || preflight=1
  check_endpoint "embedding" "http://localhost:8019/v1/models" || preflight=1
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    log_msg "FAILED missing OPENAI_API_KEY after loading ${SECRET_ENV}"
    preflight=1
  fi
  if [[ "${preflight}" -ne 0 ]]; then
    write_status "launcher" "FAILED preflight"
    return 1
  fi

  local pids=()
  local dataset
  for dataset in ${DATASETS}; do
    run_lane "${dataset}" &
    pids+=("$!")
    log_msg "LAUNCHED HGRAG lane dataset=${dataset} pid=${pids[-1]}"
  done

  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done

  if [[ "${rc}" -eq 0 ]]; then
    write_summary | tee -a "${OUT_ROOT}/logs/launcher.log"
    write_status "launcher" "DONE run_tag=${RUN_TAG}"
    log_msg "DONE ${RUN_TAG}"
  else
    write_status "launcher" "FAILED run_tag=${RUN_TAG} rc=${rc}"
    log_msg "FAILED ${RUN_TAG} rc=${rc}"
  fi
  return "${rc}"
}

main "$@"
