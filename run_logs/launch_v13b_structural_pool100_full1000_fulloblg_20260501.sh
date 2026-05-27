#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
OUT_DIR="${ROOT_DIR}/run_logs/v13b_structural_pool100_full1000_fulloblg_20260501"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
LIMIT="${LIMIT:-1000}"
DENSE_LIMIT="${DENSE_LIMIT:-100}"
MAX_ENDPOINT_DOC_DEGREE="${MAX_ENDPOINT_DOC_DEGREE:-30}"

mkdir -p "${OUT_DIR}/dense_source" "${OUT_DIR}/query_obligation_runtime" "${OUT_DIR}/qa_runtime" "${OUT_DIR}/failure_reports"

dataset_port() {
  case "$1" in
    2wikimultihopqa) echo 8041 ;;
    hotpotqa) echo 8042 ;;
    musique) echo 8043 ;;
    *) return 2 ;;
  esac
}

openie_path() {
  echo "${ROOT_DIR}/outputs_step0_general_nvembed_${1}/openie_results_ner_qwen3-8b.json"
}

is_rows_done_json() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  "${PYTHON_BIN}" - "${path}" "${LIMIT}" <<'PY'
import json
import sys

path, limit = sys.argv[1], int(sys.argv[2])
try:
    data = json.load(open(path))
except Exception:
    sys.exit(1)
datasets = data.get("datasets", [])
ok = bool(datasets)
for dataset in datasets:
    if len(dataset.get("rows", []) or []) < limit:
        ok = False
sys.exit(0 if ok else 1)
PY
}

is_cache_done_json() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  "${PYTHON_BIN}" - "${path}" "${LIMIT}" <<'PY'
import json
import sys

path, limit = sys.argv[1], int(sys.argv[2])
try:
    data = json.load(open(path))
except Exception:
    sys.exit(1)
ok = data.get("source") == "llm_query_obligation" and bool(data.get("datasets"))
for dataset in data.get("datasets", []) or []:
    rows = dataset.get("rows", []) or []
    metrics = dataset.get("metrics", {}) or {}
    if len(rows) < limit or int(metrics.get("rows", 0) or 0) < limit:
        ok = False
sys.exit(0 if ok else 1)
PY
}

is_selector_done_json() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  "${PYTHON_BIN}" - "${path}" "${LIMIT}" <<'PY'
import json
import sys

path, limit = sys.argv[1], int(sys.argv[2])
try:
    data = json.load(open(path))
except Exception:
    sys.exit(1)
ok = bool(data.get("datasets"))
for dataset in data.get("datasets", []) or []:
    if len(dataset.get("rows", []) or []) < limit:
        ok = False
    if "obligation_closed_sto_local_ppr_r5" not in (dataset.get("metrics", {}) or {}):
        ok = False
sys.exit(0 if ok else 1)
PY
}

is_qa_done_json() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  "${PYTHON_BIN}" - "${path}" "${LIMIT}" <<'PY'
import json
import sys

path, limit = sys.argv[1], int(sys.argv[2])
try:
    data = json.load(open(path))
except Exception:
    sys.exit(1)
ok = bool(data.get("datasets"))
for dataset in data.get("datasets", []) or []:
    result = (dataset.get("methods", {}) or {}).get("obligation_closed_sto_local_ppr", {}) or {}
    metrics = result.get("metrics", {}) or {}
    if int(metrics.get("count", metrics.get("num_queries", 0)) or 0) < limit:
        ok = False
    if "ExactMatch" not in metrics and "em" not in metrics:
        ok = False
    if "F1" not in metrics and "f1" not in metrics:
        ok = False
sys.exit(0 if ok else 1)
PY
}

run_dataset() {
  local dataset="$1"
  local port
  port="$(dataset_port "${dataset}")" || return 2

  local dense_source_json="${OUT_DIR}/dense_source/${dataset}_source_report.json"
  local cache_json="${OUT_DIR}/${dataset}_query_obligation_cache_full1000.json"
  local source_json="${OUT_DIR}/${dataset}_source_report.json"
  local selector_json="${OUT_DIR}/${dataset}_v13b_selector.json"
  local selector_md="${OUT_DIR}/${dataset}_v13b_selector.md"
  local qa_json="${OUT_DIR}/${dataset}_v13b_qa.json"
  local qa_md="${OUT_DIR}/${dataset}_v13b_qa.md"
  local failure_json="${OUT_DIR}/failure_reports/${dataset}_grounding_failures.json"
  local failure_md="${OUT_DIR}/failure_reports/${dataset}_grounding_failures.md"
  local status_path="${OUT_DIR}/${dataset}.status"

  echo "[START] dataset=${dataset} stage=dense_source limit=${LIMIT} start=$(date -Is)" | tee "${status_path}"
  if ! is_rows_done_json "${dense_source_json}"; then
    (
      cd "${ROOT_DIR}" || exit 1
      "${PYTHON_BIN}" scripts/v13b_make_source_report.py \
        --datasets "${dataset}" \
        --max-queries "${LIMIT}" \
        --candidate-limit "${DENSE_LIMIT}" \
        --output-json "${dense_source_json}"
    ) > "${OUT_DIR}/${dataset}_dense_source_report.log" 2>&1 || return 1
  fi

  echo "[START] dataset=${dataset} stage=query_obligation_cache limit=${LIMIT} port=${port} start=$(date -Is)" | tee "${status_path}"
  if ! is_cache_done_json "${cache_json}"; then
    (
      cd "${ROOT_DIR}" || exit 1
      "${PYTHON_BIN}" build_query_obligation_cache.py \
        --report "${dense_source_json}" \
        --source llm_query_obligation \
        --datasets "${dataset}" \
        --max-queries "${LIMIT}" \
        --llm-name qwen3-8b-train \
        --llm-base-url "http://localhost:${port}/v1" \
        --save-dir "${OUT_DIR}/query_obligation_runtime/${dataset}" \
        --max-new-tokens 512 \
        --max-retry-attempts 20 \
        --qwen-disable-thinking \
        --output-json "${cache_json}"
    ) > "${OUT_DIR}/${dataset}_query_obligation_cache.log" 2>&1 || return 1
  fi

  echo "[START] dataset=${dataset} stage=structural_source limit=${LIMIT} start=$(date -Is)" | tee "${status_path}"
  if ! is_rows_done_json "${source_json}"; then
    (
      cd "${ROOT_DIR}" || exit 1
      "${PYTHON_BIN}" scripts/build_v13b_structural_source_report.py \
        --datasets "${dataset}" \
        --max-queries "${LIMIT}" \
        --dense-limit "${DENSE_LIMIT}" \
        --max-endpoint-doc-degree "${MAX_ENDPOINT_DOC_DEGREE}" \
        --query-obligation-cache-template "${OUT_DIR}/{dataset}_query_obligation_cache_full1000.json" \
        --output-json "${source_json}"
    ) > "${OUT_DIR}/${dataset}_source_report.log" 2>&1 || return 1
  fi

  echo "[START] dataset=${dataset} stage=selector limit=${LIMIT} start=$(date -Is)" | tee "${status_path}"
  if ! is_selector_done_json "${selector_json}"; then
    (
      cd "${ROOT_DIR}" || exit 1
      "${PYTHON_BIN}" evaluate_obligation_closed_sto_local_ppr.py \
        --report "${source_json}" \
        --max-queries "${LIMIT}" \
        --evidence-set-size 5 \
        --max-path-edges 3 \
        --max-endpoint-degree 30 \
        --alpha 0.2 \
        --residual-epsilon 1e-6 \
        --query-obligation-cache-json "${cache_json}" \
        --allow-support-obligation-grounding \
        --allow-source-span-obligation-grounding \
        --allow-program-evidence-assembler \
        --use-typed-retrieval-critical-obligations \
        --output-json "${selector_json}" \
        --output-md "${selector_md}"
    ) > "${OUT_DIR}/${dataset}_v13b_selector.log" 2>&1 || return 1
  fi

  echo "[START] dataset=${dataset} stage=qa limit=${LIMIT} port=${port} start=$(date -Is)" | tee "${status_path}"
  if ! is_qa_done_json "${qa_json}"; then
    (
      cd "${ROOT_DIR}" || exit 1
      "${PYTHON_BIN}" run_transition_top5_qa.py \
        --transition-report "${selector_json}" \
        --datasets "${dataset}" \
        --methods obligation_closed_sto_local_ppr \
        --max-queries "${LIMIT}" \
        --qa-top-k 5 \
        --save-dir "${OUT_DIR}/qa_runtime/${dataset}" \
        --llm-name qwen3-8b-train \
        --llm-base-url "http://localhost:${port}/v1" \
        --qwen-disable-thinking \
        --max-new-tokens 400 \
        --embedding-name nvidia/NV-Embed-v2 \
        --embedding-base-url "${EMBEDDING_BASE_URL}" \
        --embedding-batch-size 4 \
        --openie-mode offline \
        --output-json "${qa_json}" \
        --output-md "${qa_md}"
    ) > "${OUT_DIR}/${dataset}_v13b_qa.log" 2>&1 || return 1
  fi

  echo "[START] dataset=${dataset} stage=grounding_failure_analysis limit=${LIMIT} start=$(date -Is)" | tee "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" analyze_v13b_grounding_failures.py \
      --source-report "${source_json}" \
      --selector-json "${selector_json}" \
      --query-obligation-cache "${cache_json}" \
      --openie-json "$(openie_path "${dataset}")" \
      --dataset "${dataset}" \
      --output-json "${failure_json}" \
      --output-md "${failure_md}"
  ) > "${OUT_DIR}/${dataset}_grounding_failure_analysis.log" 2>&1 || return 1

  echo "[DONE] dataset=${dataset} end=$(date -Is) source=${source_json} selector=${selector_json} qa=${qa_json} failures=${failure_json}" | tee "${status_path}"
}

echo "[START] v13b structural pool100 full1000 full-obligation parallel $(date -Is)" | tee "${OUT_DIR}/launcher.status"
pids=()
for dataset in 2wikimultihopqa hotpotqa musique; do
  (
    run_dataset "${dataset}"
  ) > "${OUT_DIR}/${dataset}_driver.log" 2>&1 &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  wait "${pid}" || rc=1
done

if [[ "${rc}" -eq 0 ]]; then
  echo "[DONE] v13b structural pool100 full1000 full-obligation parallel $(date -Is)" | tee "${OUT_DIR}/launcher.status"
else
  echo "[FAILED] v13b structural pool100 full1000 full-obligation parallel rc=${rc} $(date -Is)" | tee "${OUT_DIR}/launcher.status"
fi
exit "${rc}"
