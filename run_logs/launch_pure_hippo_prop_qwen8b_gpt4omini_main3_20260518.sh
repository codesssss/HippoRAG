#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/mnt/nvme/code/HippoRAG}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
RUN_TAG="${RUN_TAG:-pure_hippo_prop_qwen8b_gpt4omini_main3_20260518}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/run_logs/${RUN_TAG}}"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
METHODS="${METHODS:-hipporag proprag}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
QA_TOP_K="${QA_TOP_K:-5}"
POOL_K="${POOL_K:-100}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"

READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
READER_MAX_NEW_TOKENS="${READER_MAX_NEW_TOKENS:-none}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

HIPPO_POOL_8B_ROOT="${HIPPO_POOL_8B_ROOT:-${ROOT_DIR}/run_logs/hipporag_pool_exports_full1000_20260503}"
PROP_POOL_8B_ROOT="${PROP_POOL_8B_ROOT:-${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/status"

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

pool_path() {
  local method="$1"
  local dataset="$2"
  case "${method}" in
    hipporag)
      printf '%s/%s_hipporag_pool%s.json' "${HIPPO_POOL_8B_ROOT}" "${dataset}" "${POOL_K}"
      ;;
    proprag)
      printf '%s/%s_pool%s.json' "${PROP_POOL_8B_ROOT}" "${dataset}" "${POOL_K}"
      ;;
    *)
      return 2
      ;;
  esac
}

reader_input_path() {
  local method="$1"
  local dataset="$2"
  printf '%s/reader_inputs/%s/%s_%s_qwen8b_no_think_pool%s_reader_input.json' \
    "${OUT_ROOT}" "${method}" "${dataset}" "${method}" "${POOL_K}"
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

method_display_name() {
  local method="$1"
  case "${method}" in
    hipporag) printf 'hipporag_qwen8b_no_think_top%s' "${QA_TOP_K}" ;;
    proprag) printf 'proprag_qwen8b_no_think_top%s' "${QA_TOP_K}" ;;
    *) printf '%s_qwen8b_no_think_top%s' "${method}" "${QA_TOP_K}" ;;
  esac
}

preflight() {
  local rc=0
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    log_msg "FAILED missing OPENAI_API_KEY; expected SECRET_ENV=${SECRET_ENV}"
    rc=1
  fi
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    log_msg "FAILED missing executable PYTHON_BIN=${PYTHON_BIN}"
    rc=1
  fi
  local method dataset p
  for method in ${METHODS}; do
    for dataset in ${DATASETS}; do
      p="$(pool_path "${method}" "${dataset}")" || {
        log_msg "FAILED unknown method=${method}"
        rc=1
        continue
      }
      if [[ ! -s "${p}" ]]; then
        log_msg "FAILED missing pure ${method} pool dataset=${dataset} path=${p}"
        rc=1
      fi
    done
  done
  return "${rc}"
}

wrap_reader_input() {
  local method="$1"
  local dataset="$2"
  local pool_json input_json log_path method_name
  pool_json="$(pool_path "${method}" "${dataset}")"
  input_json="$(reader_input_path "${method}" "${dataset}")"
  log_path="${OUT_ROOT}/logs/wrap_${method}_${dataset}.log"
  method_name="$(method_display_name "${method}")"

  mkdir -p "$(dirname "${input_json}")"
  if [[ -s "${input_json}" ]]; then
    write_status "wrap_${method}_${dataset}" "SKIP existing input=${input_json}"
    return 0
  fi
  write_status "wrap_${method}_${dataset}" "START pool=${pool_json} output=${input_json}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" scripts/make_external_pool_reader_inputs.py \
      --pool-json "${pool_json}" \
      --method-name "${method_name}" \
      --output-json "${input_json}"
  ) > "${log_path}" 2>&1
  local rc=$?
  if [[ "${rc}" -eq 0 && -s "${input_json}" ]]; then
    write_status "wrap_${method}_${dataset}" "DONE input=${input_json}"
    return 0
  fi
  write_status "wrap_${method}_${dataset}" "FAILED rc=${rc} log=${log_path}"
  return "${rc}"
}

run_reader_dataset() {
  local method="$1"
  local dataset="$2"
  local input_json output_json output_md log_path method_name
  input_json="$(reader_input_path "${method}" "${dataset}")"
  output_json="$(reader_json_path "${method}" "${dataset}")"
  output_md="$(reader_md_path "${method}" "${dataset}")"
  log_path="${OUT_ROOT}/logs/reader_${method}_${dataset}.log"
  method_name="$(method_display_name "${method}")"

  mkdir -p "$(dirname "${output_json}")" "${OUT_ROOT}/reader_runtime/${method}/${dataset}"
  if [[ -s "${output_json}" ]]; then
    write_status "reader_${method}_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  write_status "reader_${method}_${dataset}" "START input=${input_json} output=${output_json}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. TOKENIZERS_PARALLELISM=false "${PYTHON_BIN}" \
      evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py \
      --input-files "${input_json}" \
      --method-name "${method_name}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --doc-source external_pool_topk_docs \
      --save-dir "${OUT_ROOT}/reader_runtime/${method}/${dataset}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens "${READER_MAX_NEW_TOKENS}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${output_json}" \
      --output-md "${output_md}"
  ) > "${log_path}" 2>&1
  local rc=$?
  if [[ "${rc}" -eq 0 && -s "${output_json}" ]]; then
    write_status "reader_${method}_${dataset}" "DONE output=${output_json}"
    return 0
  fi
  write_status "reader_${method}_${dataset}" "FAILED rc=${rc} log=${log_path}"
  return "${rc}"
}

run_one() {
  local method="$1"
  local dataset="$2"
  wrap_reader_input "${method}" "${dataset}" && run_reader_dataset "${method}" "${dataset}"
}

run_all_parallel() {
  local active=0
  local rc=0
  local method dataset
  for method in ${METHODS}; do
    for dataset in ${DATASETS}; do
      run_one "${method}" "${dataset}" &
      active=$((active + 1))
      if [[ "${active}" -ge "${PARALLEL_JOBS}" ]]; then
        wait -n || rc=1
        active=$((active - 1))
      fi
    done
  done
  while [[ "${active}" -gt 0 ]]; do
    wait -n || rc=1
    active=$((active - 1))
  done
  return "${rc}"
}

write_summary() {
  env RUN_OUT_ROOT="${OUT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["RUN_OUT_ROOT"])
rows = []
for path in sorted((out / "reader_qa").glob("*/*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for dataset in payload.get("datasets", []) or []:
        for method_name, method_payload in (dataset.get("methods") or {}).items():
            metrics = method_payload.get("metrics") or {}
            rows.append(
                {
                    "dataset": dataset.get("dataset"),
                    "method": method_name,
                    "count": metrics.get("count"),
                    "Recall@5": metrics.get("Recall@5") or metrics.get("r5"),
                    "Recall@20": metrics.get("Recall@20"),
                    "EM": metrics.get("ExactMatch"),
                    "F1": metrics.get("F1"),
                    "mean_reader_docs": metrics.get("mean_reader_docs"),
                    "report": str(path),
                }
            )

def fmt(value):
    if value is None:
        return ""
    return f"{float(value):.4f}"

(out / "summary.json").write_text(
    json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
lines = [
    "# Pure HippoRAG / PropRAG Qwen3-8B no-think + GPT-4o-mini Reader",
    "",
    "Reader is fixed to GPT-4o-mini; upstream retrieval pools are pure HippoRAG/PropRAG exports, not DAEC selector outputs.",
    "",
    "| Dataset | Method | Count | R@5 | R@20 | EM | F1 | Mean reader docs |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(
        f"| {row['dataset']} | {row['method']} | {int(row['count'] or 0)} | "
        f"{fmt(row['Recall@5'])} | {fmt(row['Recall@20'])} | "
        f"{fmt(row['EM'])} | {fmt(row['F1'])} | {fmt(row['mean_reader_docs'])} |"
    )
(out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"summary": str(out / "summary.md"), "rows": len(rows)}, indent=2))
PY
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: pure HippoRAG/PropRAG Qwen3-8B no-think pool exports -> GPT-4o-mini reader top-${QA_TOP_K}; no DAEC selector."
  preflight || {
    write_status "launcher" "FAILED preflight"
    return 1
  }
  run_all_parallel || {
    write_status "launcher" "FAILED reader_replay"
    write_summary || true
    return 1
  }
  write_summary
  write_status "launcher" "DONE run_tag=${RUN_TAG}"
  log_msg "DONE ${RUN_TAG}"
}

main "$@"
