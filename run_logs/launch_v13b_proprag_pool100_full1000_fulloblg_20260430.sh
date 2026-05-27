#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
SOURCE_DIR="${SOURCE_DIR:-${ROOT_DIR}/run_logs/v13b_proprag_pool100_full1000_20260430}"
OUT_DIR="${ROOT_DIR}/run_logs/v13b_proprag_pool100_full1000_fulloblg_20260430"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
LIMIT=1000

mkdir -p "${OUT_DIR}/query_obligation_runtime" "${OUT_DIR}/qa_runtime"

dataset_port() {
  case "$1" in
    2wikimultihopqa) echo 8041 ;;
    hotpotqa) echo 8042 ;;
    musique) echo 8043 ;;
    *) return 2 ;;
  esac
}

is_source_done_json() {
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
    rows = dataset.get("rows", [])
    metrics = dataset.get("metrics", {})
    if len(rows) < limit:
        ok = False
    if int(metrics.get("missing_gold_doc_index_rows", -1)) != 0:
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
for dataset in data.get("datasets", []):
    rows = dataset.get("rows", [])
    metrics = dataset.get("metrics", {})
    if len(rows) < limit:
        ok = False
    if int(metrics.get("rows", 0) or 0) < limit:
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
for dataset in data.get("datasets", []):
    rows = dataset.get("rows", [])
    metrics = dataset.get("metrics", {})
    if len(rows) < limit:
        ok = False
    if "obligation_closed_sto_local_ppr_r5" not in metrics:
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
for dataset in data.get("datasets", []):
    methods = dataset.get("methods", {})
    result = methods.get("obligation_closed_sto_local_ppr", {})
    metrics = result.get("metrics", {})
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

  local source_json="${OUT_DIR}/${dataset}_source_report.json"
  local source_seed_json="${SOURCE_DIR}/${dataset}_source_report.json"
  local cache_json="${OUT_DIR}/${dataset}_query_obligation_cache_full1000.json"
  local selector_json="${OUT_DIR}/${dataset}_v13b_selector.json"
  local selector_md="${OUT_DIR}/${dataset}_v13b_selector.md"
  local qa_json="${OUT_DIR}/${dataset}_v13b_qa.json"
  local qa_md="${OUT_DIR}/${dataset}_v13b_qa.md"
  local source_log="${OUT_DIR}/${dataset}_source_report.log"
  local cache_log="${OUT_DIR}/${dataset}_query_obligation_cache.log"
  local selector_log="${OUT_DIR}/${dataset}_v13b_selector.log"
  local qa_log="${OUT_DIR}/${dataset}_v13b_qa.log"
  local status_path="${OUT_DIR}/${dataset}.status"

  echo "[START] dataset=${dataset} stage=source_report limit=${LIMIT} start=$(date -Is)" | tee "${status_path}"
  if ! is_source_done_json "${source_json}"; then
    if is_source_done_json "${source_seed_json}"; then
      cp "${source_seed_json}" "${source_json}"
    else
      (
        cd "${ROOT_DIR}" || exit 1
        "${PYTHON_BIN}" scripts/v13b_make_source_report.py \
          --datasets "${dataset}" \
          --max-queries "${LIMIT}" \
          --candidate-limit 100 \
          --output-json "${source_json}"
      ) > "${source_log}" 2>&1 || {
        echo "[FAILED] dataset=${dataset} stage=source_report end=$(date -Is) log=${source_log}" | tee "${status_path}"
        return 1
      }
    fi
  fi

  echo "[START] dataset=${dataset} stage=query_obligation_cache limit=${LIMIT} port=${port} start=$(date -Is)" | tee "${status_path}"
  if ! is_cache_done_json "${cache_json}"; then
    (
      cd "${ROOT_DIR}" || exit 1
      "${PYTHON_BIN}" build_query_obligation_cache.py \
        --report "${source_json}" \
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
    ) > "${cache_log}" 2>&1 || {
      echo "[FAILED] dataset=${dataset} stage=query_obligation_cache end=$(date -Is) log=${cache_log}" | tee "${status_path}"
      return 1
    }
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
    ) > "${selector_log}" 2>&1 || {
      echo "[FAILED] dataset=${dataset} stage=selector end=$(date -Is) log=${selector_log}" | tee "${status_path}"
      return 1
    }
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
    ) > "${qa_log}" 2>&1 || {
      echo "[FAILED] dataset=${dataset} stage=qa end=$(date -Is) log=${qa_log}" | tee "${status_path}"
      return 1
    }
  fi

  echo "[DONE] dataset=${dataset} end=$(date -Is) cache=${cache_json} selector=${selector_json} qa=${qa_json}" | tee "${status_path}"
}

echo "[START] v13b proprag pool100 full1000 full-obligation parallel $(date -Is)" | tee "${OUT_DIR}/launcher.status"
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
  echo "[DONE] v13b proprag pool100 full1000 full-obligation parallel $(date -Is)" | tee "${OUT_DIR}/launcher.status"
else
  echo "[FAILED] v13b proprag pool100 full1000 full-obligation parallel rc=${rc} $(date -Is)" | tee "${OUT_DIR}/launcher.status"
fi
exit "${rc}"
