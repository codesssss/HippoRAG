#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
RUN_TAG="${RUN_TAG:-neocor_nq_popqa_limit100_32b_no_think_gpt4omini_20260517}"
OUT_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

DATASETS="${DATASETS:-nq_rear popqa}"
MAX_QUERIES="${MAX_QUERIES:-100}"
QA_TOP_K="${QA_TOP_K:-5}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-100}"
OPENIE_MAX_NEW_TOKENS="${OPENIE_MAX_NEW_TOKENS:-2048}"

GRAPH_LLM_NAME="${GRAPH_LLM_NAME:-qwen3-32b-judge}"
QWEN32_URL_A="${QWEN32_URL_A:-http://localhost:8045/v1}"
QWEN32_URL_B="${QWEN32_URL_B:-http://localhost:8046/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
READER_MAX_NEW_TOKENS="${READER_MAX_NEW_TOKENS:-none}"

NEOCORRAG_ROOT="${NEOCORRAG_ROOT:-/mnt/nvme/code/NeocorRAG}"
RERETRIEVAL_LLM_NAME="${RERETRIEVAL_LLM_NAME:-/mnt/nvme/Qwen3-32B}"
NEOCORRAG_CUDA_VISIBLE_DEVICES="${NEOCORRAG_CUDA_VISIBLE_DEVICES:-4}"

if [[ -s "${SECRET_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRET_ENV}"
  set +a
fi

mkdir -p \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/status" \
  "${OUT_ROOT}/neocorrag" \
  "${OUT_ROOT}/reader_inputs/neocorrag" \
  "${OUT_ROOT}/reader_qa/neocorrag" \
  "${OUT_ROOT}/reader_runtime/neocorrag"

export HIPPORAG_RERANK_FORCE_NO_THINK=1
export TOKENIZERS_PARALLELISM=false

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/logs/launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${OUT_ROOT}/status/${name}.status"
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
    nq_rear) printf 'nq\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

qwen_url_for_dataset() {
  case "$1" in
    nq|nq_rear) printf '%s\n' "${QWEN32_URL_A}" ;;
    popqa) printf '%s\n' "${QWEN32_URL_B}" ;;
    *) printf '%s\n' "${QWEN32_URL_A}" ;;
  esac
}

neocor_output_path() {
  local dataset="$1"
  local display_dataset
  display_dataset="$(display_dataset_name "${dataset}")"
  printf '%s/neocorrag/%s_neocorrag_k3_qwen32b_no_think_limit%s.json' \
    "${OUT_ROOT}" "${display_dataset}" "${MAX_QUERIES}"
}

neocor_reader_input_path() {
  local dataset="$1"
  local display_dataset
  display_dataset="$(display_dataset_name "${dataset}")"
  printf '%s/reader_inputs/neocorrag/%s_neocorrag_qwen32b_no_think_top%s_limit%s_reader_input.json' \
    "${OUT_ROOT}" "${display_dataset}" "${QA_TOP_K}" "${MAX_QUERIES}"
}

neocor_reader_report_path() {
  local dataset="$1"
  local display_dataset
  display_dataset="$(display_dataset_name "${dataset}")"
  printf '%s/reader_qa/neocorrag/%s_neocorrag_qwen32b_no_think_gpt4omini_reader_top%s_limit%s.json' \
    "${OUT_ROOT}" "${display_dataset}" "${QA_TOP_K}" "${MAX_QUERIES}"
}

neocor_reader_report_md_path() {
  local dataset="$1"
  local display_dataset
  display_dataset="$(display_dataset_name "${dataset}")"
  printf '%s/reader_qa/neocorrag/%s_neocorrag_qwen32b_no_think_gpt4omini_reader_top%s_limit%s.md' \
    "${OUT_ROOT}" "${display_dataset}" "${QA_TOP_K}" "${MAX_QUERIES}"
}

run_logged() {
  local label="$1"
  local status_name="$2"
  local log_path="$3"
  shift 3

  write_status "${status_name}" "START ${label} log=${log_path}"
  (
    printf '[START] %s time=%s\n' "${label}" "$(date -Is)"
    cd "${ROOT_DIR}" || exit 1
    "$@"
    rc=$?
    printf '[EXIT] %s rc=%s time=%s\n' "${label}" "${rc}" "$(date -Is)"
    exit "${rc}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    write_status "${status_name}" "DONE ${label}"
  else
    write_status "${status_name}" "FAILED code=${code} ${label} log=${log_path}"
  fi
  return "${code}"
}

run_neocor_retrieval_dataset() {
  local dataset="$1"
  local output_json log_path llm_base_url
  output_json="$(neocor_output_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/neocor_${dataset}.log"
  llm_base_url="$(qwen_url_for_dataset "${dataset}")"
  if [[ -s "${output_json}" ]]; then
    write_status "neocor_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  run_logged "neocor retrieval dataset=${dataset}" "neocor_${dataset}" "${log_path}" \
    env PYTHONPATH=. CUDA_VISIBLE_DEVICES="${NEOCORRAG_CUDA_VISIBLE_DEVICES}" \
      "${PYTHON_BIN}" scripts/run_neocorrag_aligned.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --neocorrag_root "${NEOCORRAG_ROOT}" \
      --output_json "${output_json}" \
      --output_method_name "aligned_limit${MAX_QUERIES}_${GRAPH_LLM_NAME}_beam_k3_ret100_reretrieval32b_nvembedv2_nq_popqa_20260517" \
      --llm_name "${GRAPH_LLM_NAME}" \
      --llm_base_url "${llm_base_url}" \
      --graph_llm_name "${GRAPH_LLM_NAME}" \
      --graph_llm_base_url "${llm_base_url}" \
      --embedding_name nvidia/NV-Embed-v2 \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --reretrieval_llm_name "${RERETRIEVAL_LLM_NAME}" \
      --reretrieval_embedding_name nvidia/NV-Embed-v2 \
      --generation_mode beam \
      --k 3 \
      --qa_top_k "${QA_TOP_K}" \
      --retrieval_top_k "${RETRIEVAL_TOP_K}" \
      --embedding_batch_size 4 \
      --max_new_tokens "${OPENIE_MAX_NEW_TOKENS}"
}

wrap_neocor_reader_input() {
  local dataset="$1"
  local source_json input_json log_path
  source_json="$(neocor_output_path "${dataset}")"
  input_json="$(neocor_reader_input_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/wrap_neocor_${dataset}.log"
  if [[ -s "${input_json}" ]]; then
    write_status "wrap_neocor_${dataset}" "SKIP existing input=${input_json}"
    return 0
  fi
  write_status "wrap_neocor_${dataset}" "START source=${source_json}"
  env SOURCE_JSON="${source_json}" OUTPUT_JSON="${input_json}" "${PYTHON_BIN}" - <<'PY' > "${log_path}" 2>&1
import ast
import json
import os
from pathlib import Path

source = Path(os.environ["SOURCE_JSON"])
output = Path(os.environ["OUTPUT_JSON"])
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
    docs = list(record.get("docs", []) or [])[:5]
    examples.append({
        "query_index": int(record.get("query_idx", idx)),
        "question": str(record.get("question") or ""),
        "gold_answers": normalize_gold_answers(record.get("gold_answers", [])),
        "gold_docs": list(record.get("gold_docs", []) or []),
        "docs": docs,
        "retrieved_doc_ids": list(range(len(docs))),
        "retrieval_trace": {
            "source_output_json": str(source.resolve()),
            "source_method": "neocorrag",
            "uses_saved_neocorrag_docs": True,
            "source_reader": payload.get("config", {}).get("llm_name"),
        },
    })
metrics = payload.get("overall_retrieval_result", {}) or {}
output_payload = {
    "format": "neocorrag_saved_docs_reader_input_v1",
    "dataset": payload.get("dataset"),
    "method": "neocorrag",
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
    write_status "wrap_neocor_${dataset}" "DONE input=${input_json}"
  else
    write_status "wrap_neocor_${dataset}" "FAILED code=${code} log=${log_path}"
  fi
  return "${code}"
}

run_neocor_reader_dataset() {
  local dataset="$1"
  local input_json output_json output_md log_path
  input_json="$(neocor_reader_input_path "${dataset}")"
  output_json="$(neocor_reader_report_path "${dataset}")"
  output_md="$(neocor_reader_report_md_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/reader_neocor_${dataset}.log"
  if [[ -s "${output_json}" ]]; then
    write_status "reader_neocor_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  run_logged "reader neocor dataset=${dataset}" "reader_neocor_${dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py \
      --input-files "${input_json}" \
      --method-name "neocorrag_qwen32b_no_think_top${QA_TOP_K}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --doc-source saved_docs \
      --save-dir "${OUT_ROOT}/reader_runtime/neocorrag/$(display_dataset_name "${dataset}")" \
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

write_summary() {
  env OUT_ROOT="${OUT_ROOT}" "${PYTHON_BIN}" - <<'PY' > "${OUT_ROOT}/summary.tsv"
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
print("dataset\tmethod\tcount\tR@5\tR@20\tEM\tF1\tfile")
for path in sorted((root / "reader_qa" / "neocorrag").glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    for ds in data.get("datasets", []):
        label = ds.get("dataset")
        if label == "nq_rear":
            label = "nq"
        for name, payload in (ds.get("methods") or {}).items():
            m = payload.get("metrics") or {}
            print(
                "\t".join(
                    str(x)
                    for x in [
                        label,
                        name,
                        m.get("count", ""),
                        m.get("Recall@5", m.get("r5", "")),
                        m.get("Recall@20", ""),
                        m.get("ExactMatch", ""),
                        m.get("F1", ""),
                        path,
                    ]
                )
            )
PY
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: NeocorRAG limit=${MAX_QUERIES}; Qwen3-32B no_think non-reader; GPT-4o-mini reader; datasets=${DATASETS}"

  local preflight=0
  check_endpoint "qwen32_a" "${QWEN32_URL_A%/}/models" || preflight=1
  check_endpoint "qwen32_b" "${QWEN32_URL_B%/}/models" || preflight=1
  check_endpoint "embedding" "${EMBEDDING_BASE_URL%/embeddings}/models" || preflight=1
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    log_msg "FAILED missing OPENAI_API_KEY for GPT-4o-mini reader; expected SECRET_ENV=${SECRET_ENV}"
    preflight=1
  fi
  if [[ "${preflight}" -ne 0 ]]; then
    write_status "launcher" "FAILED preflight"
    return 1
  fi

  local dataset
  for dataset in ${DATASETS}; do
    run_neocor_retrieval_dataset "${dataset}" || {
      write_status "launcher" "FAILED neocor retrieval dataset=${dataset}"
      return 1
    }
    wrap_neocor_reader_input "${dataset}" || {
      write_status "launcher" "FAILED wrap dataset=${dataset}"
      return 1
    }
    run_neocor_reader_dataset "${dataset}" || {
      write_status "launcher" "FAILED reader dataset=${dataset}"
      return 1
    }
  done

  write_summary
  write_status "launcher" "DONE run_tag=${RUN_TAG}"
  log_msg "DONE ${RUN_TAG}"
}

main "$@"
