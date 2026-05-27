#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/run_logs/pathrag_qwen32b_no_think_gpt4omini_full1000_20260520}"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

LLM_NAME="${LLM_NAME:-qwen3-32b-judge}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-nvidia/NV-Embed-v2}"
LIMIT="${LIMIT:-1000}"
QA_TOP_K="${QA_TOP_K:-5}"
PATHRAG_TOP_K="${PATHRAG_TOP_K:-40}"
LLM_MAX_ASYNC="${LLM_MAX_ASYNC:-16}"
EMBEDDING_MAX_ASYNC="${EMBEDDING_MAX_ASYNC:-1}"
EMBEDDING_BATCH_NUM="${EMBEDDING_BATCH_NUM:-4}"
ENTITY_EXTRACT_MAX_GLEANING="${ENTITY_EXTRACT_MAX_GLEANING:-1}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa musique nq_rear popqa}"
LANE_A_URL="${LANE_A_URL:-http://localhost:8045/v1}"
LANE_B_URL="${LANE_B_URL:-http://localhost:8046/v1}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/reader_outputs" "${OUT_ROOT}/reader_runtime"

run_one() {
  local dataset="$1"
  local llm_base_url="$2"
  local log_path="${OUT_ROOT}/logs/${dataset}.log"
  local reader_input="${OUT_ROOT}/reader_inputs/${dataset}_pathrag_qwen32b_no_think_top${QA_TOP_K}_limit${LIMIT}_reader_input.json"
  local reader_json="${OUT_ROOT}/reader_outputs/${dataset}_pathrag_gpt4omini_top${QA_TOP_K}_limit${LIMIT}.json"
  local reader_md="${OUT_ROOT}/reader_outputs/${dataset}_pathrag_gpt4omini_top${QA_TOP_K}_limit${LIMIT}.md"

  {
    echo "[pathrag] dataset=${dataset} llm_base_url=${llm_base_url} start=$(date -Is)"
    if [[ -f "${reader_input}" && "${FORCE_RETRIEVAL:-0}" != "1" ]]; then
      echo "[pathrag] dataset=${dataset} retrieval exists, skip indexing/retrieval: ${reader_input}"
    else
      "${PYTHON_BIN}" scripts/run_pathrag_aligned.py \
        --dataset "${dataset}" \
        --limit "${LIMIT}" \
        --output-root "${OUT_ROOT}" \
        --llm-base-url "${llm_base_url}" \
        --llm-name "${LLM_NAME}" \
        --llm-max-async "${LLM_MAX_ASYNC}" \
        --embedding-base-url "${EMBEDDING_BASE_URL}" \
        --embedding-name "${EMBEDDING_NAME}" \
        --embedding-max-async "${EMBEDDING_MAX_ASYNC}" \
        --embedding-batch-num "${EMBEDDING_BATCH_NUM}" \
        --entity-extract-max-gleaning "${ENTITY_EXTRACT_MAX_GLEANING}" \
        --pathrag-top-k "${PATHRAG_TOP_K}" \
        --qa-top-k "${QA_TOP_K}" || {
          echo "[pathrag] dataset=${dataset} retrieval failed rc=$? end=$(date -Is)"
          return 1
        }
    fi

    if [[ -f "${SECRET_ENV}" ]]; then
      set -a
      # shellcheck disable=SC1090
      source "${SECRET_ENV}"
      set +a
    fi

    echo "[pathrag-reader] dataset=${dataset} start=$(date -Is)"
    if [[ -f "${reader_json}" && "${FORCE_READER:-0}" != "1" ]]; then
      echo "[pathrag-reader] dataset=${dataset} reader exists, skip: ${reader_json}"
    else
      env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py \
        --input-files "${reader_input}" \
        --method-name pathrag \
        --max-queries "${LIMIT}" \
        --qa-top-k "${QA_TOP_K}" \
        --doc-source saved_docs \
        --save-dir "${OUT_ROOT}/reader_runtime/${dataset}" \
        --llm-name "${READER_LLM_NAME:-gpt-4o-mini}" \
        --llm-base-url "${READER_LLM_BASE_URL:-}" \
        --output-json "${reader_json}" \
        --output-md "${reader_md}" || {
          echo "[pathrag-reader] dataset=${dataset} failed rc=$? end=$(date -Is)"
          return 1
        }
    fi
    echo "[pathrag] dataset=${dataset} done=$(date -Is)"
  } 2>&1 | tee -a "${log_path}"
}

run_wave() {
  local left_dataset="$1"
  local right_dataset="${2:-}"
  local left_pid=""
  local right_pid=""

  run_one "${left_dataset}" "${LANE_A_URL}" &
  left_pid="$!"
  if [[ -n "${right_dataset}" ]]; then
    run_one "${right_dataset}" "${LANE_B_URL}" &
    right_pid="$!"
  fi
  local rc=0
  wait "${left_pid}" || rc=1
  if [[ -n "${right_pid}" ]]; then
    wait "${right_pid}" || rc=1
  fi
  return "${rc}"
}

run_dataset_list() {
  local datasets=("$@")
  local i=0
  local rc=0
  while [[ "${i}" -lt "${#datasets[@]}" ]]; do
    if [[ $((i + 1)) -lt "${#datasets[@]}" ]]; then
      run_wave "${datasets[$i]}" "${datasets[$((i + 1))]}" || rc=1
      i=$((i + 2))
    else
      run_wave "${datasets[$i]}" || rc=1
      i=$((i + 1))
    fi
  done
  return "${rc}"
}

read -r -a DATASET_ARRAY <<< "${DATASETS}"
echo "[pathrag-launch] out_root=${OUT_ROOT} datasets=${DATASETS} limit=${LIMIT} gleaning=${ENTITY_EXTRACT_MAX_GLEANING} start=$(date -Is)"
run_dataset_list "${DATASET_ARRAY[@]}" || true
"${PYTHON_BIN}" scripts/summarize_pathrag_aligned.py --run-root "${OUT_ROOT}" || true
echo "[pathrag-launch] done=$(date -Is)"
