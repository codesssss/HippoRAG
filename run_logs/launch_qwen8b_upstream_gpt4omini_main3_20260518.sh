#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/mnt/nvme/code/HippoRAG}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
RUN_TAG="${RUN_TAG:-qwen8b_upstream_gpt4omini_main3_20260518}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/run_logs/${RUN_TAG}}"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
METHODS="${METHODS:-evidenceflow_etv3 hgrag neocor}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
QA_TOP_K="${QA_TOP_K:-5}"
READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
READER_MAX_NEW_TOKENS="${READER_MAX_NEW_TOKENS:-none}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
PARALLEL_METHODS="${PARALLEL_METHODS:-0}"
SKIP_QA="${SKIP_QA:-0}"

EVIDENCEFLOW_8B_ROOT="${EVIDENCEFLOW_8B_ROOT:-${ROOT_DIR}/run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_full_20260510}"
HGRAG_8B_ROOT="${HGRAG_8B_ROOT:-${ROOT_DIR}/run_logs/hgrag_aligned_full1000_20260429}"
NEOCOR_8B_ROOT="${NEOCOR_8B_ROOT:-${ROOT_DIR}/run_logs/neocorrag_aligned_full1000_20260429}"
mkdir -p \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/status" \
  "${OUT_ROOT}/reader_inputs/hgrag" \
  "${OUT_ROOT}/reader_inputs/neocor" \
  "${OUT_ROOT}/reader_qa/evidenceflow_etv3" \
  "${OUT_ROOT}/reader_qa/hgrag" \
  "${OUT_ROOT}/reader_qa/neocor" \
  "${OUT_ROOT}/reader_runtime/evidenceflow_etv3" \
  "${OUT_ROOT}/reader_runtime/hgrag" \
  "${OUT_ROOT}/reader_runtime/neocor"

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

check_reader_key() {
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    return 0
  fi
  log_msg "FAILED missing OPENAI_API_KEY; expected SECRET_ENV=${SECRET_ENV}"
  return 1
}

evidenceflow_retrieval_path() {
  local dataset="$1"
  printf '%s/runs/%s/%s/reports/%s_evidence_transition_graphragv3_variable_flow_retrieval.json' \
    "${EVIDENCEFLOW_8B_ROOT}" "${dataset}" "${dataset}" "${dataset}"
}

hgrag_source_path() {
  local dataset="$1"
  printf '%s/%s_hgrag.json' "${HGRAG_8B_ROOT}" "${dataset}"
}

neocor_source_path() {
  local dataset="$1"
  printf '%s/%s_neocorrag.json' "${NEOCOR_8B_ROOT}" "${dataset}"
}

saved_docs_input_path() {
  local method="$1"
  local dataset="$2"
  printf '%s/reader_inputs/%s/%s_%s_qwen8b_no_think_reader_input.json' \
    "${OUT_ROOT}" "${method}" "${dataset}" "${method}"
}

reader_json_path() {
  local method="$1"
  local dataset="$2"
  printf '%s/reader_qa/%s/%s_%s_qwen8b_no_think_gpt4omini_reader_top%s_limit%s.json' \
    "${OUT_ROOT}" "${method}" "${dataset}" "${method}" "${QA_TOP_K}" "${MAX_QUERIES}"
}

reader_md_path() {
  local method="$1"
  local dataset="$2"
  printf '%s/reader_qa/%s/%s_%s_qwen8b_no_think_gpt4omini_reader_top%s_limit%s.md' \
    "${OUT_ROOT}" "${method}" "${dataset}" "${method}" "${QA_TOP_K}" "${MAX_QUERIES}"
}

wrap_saved_docs_input() {
  local method="$1"
  local dataset="$2"
  local source_json output_json
  case "${method}" in
    hgrag) source_json="$(hgrag_source_path "${dataset}")" ;;
    neocor) source_json="$(neocor_source_path "${dataset}")" ;;
    *)
      log_msg "FAILED wrap unsupported method=${method}"
      return 2
      ;;
  esac
  output_json="$(saved_docs_input_path "${method}" "${dataset}")"
  if [[ ! -s "${source_json}" ]]; then
    write_status "wrap_${method}_${dataset}" "FAILED missing_source=${source_json}"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "wrap_${method}_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  write_status "wrap_${method}_${dataset}" "START source=${source_json} output=${output_json}"
  SOURCE_JSON="${source_json}" OUTPUT_JSON="${output_json}" METHOD="${method}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
from typing import Any, Mapping


def normalize_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(normalize_answers(item))
        return sorted({item.strip() for item in out if item.strip()})
    return [str(value)]


source = Path(os.environ["SOURCE_JSON"]).expanduser()
output = Path(os.environ["OUTPUT_JSON"]).expanduser()
method = os.environ["METHOD"]
payload = json.loads(source.read_text(encoding="utf-8"))
if not isinstance(payload, Mapping):
    raise TypeError(f"Expected object payload in {source}")

examples = []
for offset, record in enumerate(payload.get("records", []) or []):
    if not isinstance(record, Mapping):
        continue
    query_idx = int(record.get("query_idx", offset))
    if method == "hgrag":
        docs = list(record.get("retrieved_docs", []) or [])
        retrieved_doc_ids = list(record.get("retrieved_doc_ids", []) or [])
        gold_doc_ids = list(record.get("gold_doc_ids", []) or [])
    elif method == "neocor":
        docs = list(record.get("docs", []) or [])
        retrieved_doc_ids = list(record.get("retrieved_doc_ids", []) or [])
        gold_doc_ids = list(record.get("gold_doc_ids", []) or [])
    else:
        raise ValueError(method)
    examples.append(
        {
            "query_index": query_idx,
            "question": str(record.get("question") or ""),
            "gold_answers": normalize_answers(record.get("gold_answers", [])),
            "gold_doc_ids": gold_doc_ids,
            "docs": docs[:5],
            "retrieved_doc_ids": retrieved_doc_ids[:5],
            "retrieval_trace": {
                "source_output_json": str(source.resolve()),
                "source_method": method,
                "upstream_llm": "qwen3-8b-train",
                "upstream_no_think": True,
                "does_not_modify_retrieval": True,
            },
        }
    )

metrics = payload.get("overall_retrieval_result", {}) or {}
output_payload = {
    "format": "saved_docs_reader_input_v1",
    "dataset": str(payload.get("dataset") or ""),
    "method": f"{method}_qwen8b_no_think",
    "source": str(source.resolve()),
    "limit": payload.get("limit"),
    "num_queries": len(examples),
    "overall_recomputed": metrics,
    "retrieval": {"recomputed_title_recall": metrics},
    "config": {
        **dict(payload.get("config", {}) or {}),
        "reader_replay_protocol": "qwen8b_upstream_gpt4omini",
        "source_method": method,
    },
    "examples": examples,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output_json": str(output), "count": len(examples)}, indent=2))
PY
  local rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    write_status "wrap_${method}_${dataset}" "FAILED rc=${rc}"
    return "${rc}"
  fi
  write_status "wrap_${method}_${dataset}" "DONE output=${output_json}"
}

run_evidenceflow_reader_dataset() {
  local dataset="$1"
  local retrieval_json output_json output_md log_path skip_arg=()
  retrieval_json="$(evidenceflow_retrieval_path "${dataset}")"
  output_json="$(reader_json_path evidenceflow_etv3 "${dataset}")"
  output_md="$(reader_md_path evidenceflow_etv3 "${dataset}")"
  log_path="${OUT_ROOT}/logs/reader_evidenceflow_etv3_${dataset}.log"
  if [[ ! -s "${retrieval_json}" ]]; then
    write_status "reader_evidenceflow_etv3_${dataset}" "FAILED missing_retrieval=${retrieval_json}"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "reader_evidenceflow_etv3_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  if [[ "${SKIP_QA}" == "1" ]]; then
    skip_arg=(--skip-qa)
  fi
  write_status "reader_evidenceflow_etv3_${dataset}" "START retrieval=${retrieval_json} output=${output_json}"
  log_msg "START EvidenceFlow-ETv3 qwen8b upstream GPT-4o-mini reader dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. TOKENIZERS_PARALLELISM=false "${PYTHON_BIN}" \
      evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_reader_qa.py \
      --retrieval-reports "${retrieval_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${OUT_ROOT}/reader_runtime/evidenceflow_etv3/${dataset}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens "${READER_MAX_NEW_TOKENS}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      "${skip_arg[@]}" \
      --output-json "${output_json}" \
      --output-md "${output_md}"
  ) > "${log_path}" 2>&1
  local rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    write_status "reader_evidenceflow_etv3_${dataset}" "FAILED rc=${rc} log=${log_path}"
    return "${rc}"
  fi
  write_status "reader_evidenceflow_etv3_${dataset}" "DONE output=${output_json}"
}

run_saved_docs_reader_dataset() {
  local method="$1"
  local dataset="$2"
  local input_json output_json output_md log_path skip_arg=()
  case "${method}" in
    hgrag|neocor)
      wrap_saved_docs_input "${method}" "${dataset}" || return $?
      input_json="$(saved_docs_input_path "${method}" "${dataset}")"
      ;;
    *)
      log_msg "FAILED saved-docs reader unsupported method=${method}"
      return 2
      ;;
  esac
  if [[ ! -s "${input_json}" ]]; then
    write_status "reader_${method}_${dataset}" "FAILED missing_input=${input_json}"
    return 2
  fi
  output_json="$(reader_json_path "${method}" "${dataset}")"
  output_md="$(reader_md_path "${method}" "${dataset}")"
  log_path="${OUT_ROOT}/logs/reader_${method}_${dataset}.log"
  if [[ -s "${output_json}" ]]; then
    write_status "reader_${method}_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  if [[ "${SKIP_QA}" == "1" ]]; then
    skip_arg=(--skip-qa)
  fi
  write_status "reader_${method}_${dataset}" "START input=${input_json} output=${output_json}"
  log_msg "START ${method} qwen8b upstream GPT-4o-mini reader dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. TOKENIZERS_PARALLELISM=false "${PYTHON_BIN}" \
      evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py \
      --input-files "${input_json}" \
      --method-name "${method}_qwen8b_no_think_top${QA_TOP_K}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --doc-source saved_docs \
      --save-dir "${OUT_ROOT}/reader_runtime/${method}/${dataset}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens "${READER_MAX_NEW_TOKENS}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      "${skip_arg[@]}" \
      --output-json "${output_json}" \
      --output-md "${output_md}"
  ) > "${log_path}" 2>&1
  local rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    write_status "reader_${method}_${dataset}" "FAILED rc=${rc} log=${log_path}"
    return "${rc}"
  fi
  write_status "reader_${method}_${dataset}" "DONE output=${output_json}"
}

run_method_dataset() {
  local method="$1"
  local dataset="$2"
  case "${method}" in
    evidenceflow_etv3) run_evidenceflow_reader_dataset "${dataset}" ;;
    hgrag|neocor) run_saved_docs_reader_dataset "${method}" "${dataset}" ;;
    *)
      log_msg "FAILED unsupported method=${method}"
      return 2
      ;;
  esac
}

write_summary() {
  OUT_ROOT="${OUT_ROOT}" PYTHON_BIN="${PYTHON_BIN}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT_ROOT"])
rows = []
for path in sorted((out / "reader_qa").glob("*/*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for dataset in payload.get("datasets", []) or []:
        label = dataset.get("dataset")
        for method, method_payload in (dataset.get("methods") or {}).items():
            metrics = method_payload.get("metrics") or {}
            rows.append(
                {
                    "dataset": label,
                    "method": method,
                    "count": metrics.get("count"),
                    "Recall@5": metrics.get("Recall@5") or metrics.get("r5"),
                    "Recall@20": metrics.get("Recall@20"),
                    "ExactMatch": metrics.get("ExactMatch"),
                    "F1": metrics.get("F1"),
                    "mean_reader_docs": metrics.get("mean_reader_docs"),
                    "source_json": str(path),
                }
            )

def fmt(value):
    return "" if value is None else f"{float(value):.4f}"

(out / "summary.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lines = [
    "| Dataset | Method | Count | R@5 | R@20 | EM | F1 | Reader docs |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(
        f"| {row['dataset']} | {row['method']} | {int(row.get('count') or 0)} | "
        f"{fmt(row.get('Recall@5'))} | {fmt(row.get('Recall@20'))} | "
        f"{fmt(row.get('ExactMatch'))} | {fmt(row.get('F1'))} | {fmt(row.get('mean_reader_docs'))} |"
    )
(out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))
PY
}

main() {
  local rc=0
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: Qwen3-8B no-think upstream assets; GPT-4o-mini reader; datasets=${DATASETS}; methods=${METHODS}; max_queries=${MAX_QUERIES}; skip_qa=${SKIP_QA}"
  if [[ "${SKIP_QA}" != "1" ]]; then
    check_reader_key || return 1
  fi

  if [[ "${PARALLEL_METHODS}" == "1" ]]; then
    local pids=()
    local method dataset
    for method in ${METHODS}; do
      for dataset in ${DATASETS}; do
        (
          run_method_dataset "${method}" "${dataset}"
        ) > "${OUT_ROOT}/logs/lane_${method}_${dataset}.log" 2>&1 &
        pids+=("$!")
        log_msg "LAUNCHED method=${method} dataset=${dataset} pid=$!"
      done
    done
    local pid
    for pid in "${pids[@]}"; do
      wait "${pid}" || rc=1
    done
  else
    local method dataset
    for method in ${METHODS}; do
      for dataset in ${DATASETS}; do
        run_method_dataset "${method}" "${dataset}" || rc=1
        if [[ "${rc}" -ne 0 ]]; then
          break
        fi
      done
      if [[ "${rc}" -ne 0 ]]; then
        break
      fi
    done
  fi

  write_summary || rc=1
  if [[ "${rc}" -ne 0 ]]; then
    write_status launcher "FAILED rc=${rc}"
    log_msg "FAILED ${RUN_TAG} rc=${rc}"
    return "${rc}"
  fi
  write_status launcher "DONE run_tag=${RUN_TAG}"
  log_msg "DONE ${RUN_TAG}"
}

main "$@"
