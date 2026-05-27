#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
OUT_DIR="${ROOT_DIR}/run_logs/v13b_proprag_pool100_limit100_20260430"
QUERY_OBLIGATION_CACHE="${QUERY_OBLIGATION_CACHE:-/mnt/nvme/zly/HippoRAG/outputs_anchor_guided_evidence_20260426/reports/query_obligation_cache_llm_100x3_20260429.json}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"

mkdir -p "${OUT_DIR}/qa_runtime"

dataset_port() {
  case "$1" in
    2wikimultihopqa) echo 8041 ;;
    hotpotqa) echo 8042 ;;
    musique) echo 8043 ;;
    *) return 2 ;;
  esac
}

is_selector_done_json() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  "${PYTHON_BIN}" - "${path}" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
ok = bool(data.get("datasets"))
for dataset in data.get("datasets", []):
    rows = dataset.get("rows", [])
    metrics = dataset.get("metrics", {})
    if len(rows) < 100:
        ok = False
    if "obligation_closed_sto_local_ppr_r5" not in metrics:
        ok = False
sys.exit(0 if ok else 1)
PY
}

is_qa_done_json() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  "${PYTHON_BIN}" - "${path}" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
ok = bool(data.get("datasets"))
for dataset in data.get("datasets", []):
    methods = dataset.get("methods", {})
    result = methods.get("obligation_closed_sto_local_ppr", {})
    metrics = result.get("metrics", {})
    if int(metrics.get("num_queries", 0) or 0) < 100:
        ok = False
    if "em" not in metrics or "f1" not in metrics:
        ok = False
sys.exit(0 if ok else 1)
PY
}

run_dataset() {
  local dataset="$1"
  local port
  port="$(dataset_port "${dataset}")" || return 2

  local source_json="${OUT_DIR}/${dataset}_source_report.json"
  local selector_json="${OUT_DIR}/${dataset}_v13b_selector.json"
  local selector_md="${OUT_DIR}/${dataset}_v13b_selector.md"
  local qa_json="${OUT_DIR}/${dataset}_v13b_qa.json"
  local qa_md="${OUT_DIR}/${dataset}_v13b_qa.md"
  local source_log="${OUT_DIR}/${dataset}_source_report.log"
  local selector_log="${OUT_DIR}/${dataset}_v13b_selector.log"
  local qa_log="${OUT_DIR}/${dataset}_v13b_qa.log"
  local status_path="${OUT_DIR}/${dataset}.status"

  echo "[START] dataset=${dataset} stage=source_report start=$(date -Is)" | tee "${status_path}"
  if [[ ! -s "${source_json}" ]]; then
    (
      cd "${ROOT_DIR}" || exit 1
      "${PYTHON_BIN}" scripts/v13b_make_source_report.py \
        --datasets "${dataset}" \
        --max-queries 100 \
        --candidate-limit 100 \
        --output-json "${source_json}"
    ) > "${source_log}" 2>&1 || {
      echo "[FAILED] dataset=${dataset} stage=source_report end=$(date -Is) log=${source_log}" | tee "${status_path}"
      return 1
    }
  fi

  echo "[START] dataset=${dataset} stage=selector start=$(date -Is)" | tee "${status_path}"
  if ! is_selector_done_json "${selector_json}"; then
    (
      cd "${ROOT_DIR}" || exit 1
      "${PYTHON_BIN}" evaluate_obligation_closed_sto_local_ppr.py \
        --report "${source_json}" \
        --max-queries 100 \
        --evidence-set-size 5 \
        --max-path-edges 3 \
        --max-endpoint-degree 30 \
        --alpha 0.2 \
        --residual-epsilon 1e-6 \
        --query-obligation-cache-json "${QUERY_OBLIGATION_CACHE}" \
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

  echo "[START] dataset=${dataset} stage=qa port=${port} start=$(date -Is)" | tee "${status_path}"
  if ! is_qa_done_json "${qa_json}"; then
    (
      cd "${ROOT_DIR}" || exit 1
      "${PYTHON_BIN}" run_transition_top5_qa.py \
        --transition-report "${selector_json}" \
        --datasets "${dataset}" \
        --methods obligation_closed_sto_local_ppr \
        --max-queries 100 \
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

  echo "[DONE] dataset=${dataset} end=$(date -Is) selector=${selector_json} qa=${qa_json}" | tee "${status_path}"
}

echo "[START] v13b proprag pool100 limit100 $(date -Is)" | tee "${OUT_DIR}/launcher.status"
rc=0
for dataset in 2wikimultihopqa hotpotqa musique; do
  run_dataset "${dataset}" || {
    rc=1
    break
  }
done

if [[ "${rc}" -eq 0 ]]; then
  echo "[DONE] v13b proprag pool100 limit100 $(date -Is)" | tee "${OUT_DIR}/launcher.status"
else
  echo "[FAILED] v13b proprag pool100 limit100 rc=${rc} $(date -Is)" | tee "${OUT_DIR}/launcher.status"
fi
exit "${rc}"
