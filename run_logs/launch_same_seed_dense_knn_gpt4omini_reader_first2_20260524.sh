#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"

RUN_TAG="${RUN_TAG:-evidencelink_same_seed_dense_control_full1000_20260524}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/run_logs/${RUN_TAG}}"
READER_QA_ROOT="${READER_QA_ROOT:-${OUT_ROOT}/reader_qa}"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"
ETV4_ROOT="${ETV4_ROOT:-${ROOT_DIR}/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512}"

if [[ -s "${SECRET_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRET_ENV}"
  set +a
fi

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
POOL_K="${POOL_K:-100}"
READER_BUDGET_K="${READER_BUDGET_K:-5}"
PREFIX_BUDGET_M="${PREFIX_BUDGET_M:-4}"
RESIDUAL_BUDGET="$((READER_BUDGET_K - PREFIX_BUDGET_M))"
POLICY="same_seed_edge_count_matched_dense_doc_knn"

READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-none}"
QA_TOP_K="${QA_TOP_K:-5}"

EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

mkdir -p \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/status" \
  "${READER_QA_ROOT}/inputs" \
  "${READER_QA_ROOT}/reader_runtime" \
  "${READER_QA_ROOT}/reports"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/logs/04_reader_qa_first2_launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${OUT_ROOT}/status/${name}.status"
}

pcec_report_path() {
  local dataset="$1"
  printf '%s/pcec/%s/evals/%s_%s_pcec_native_pool_prefix%s_residual%s_pool%s_limit%s.json' \
    "${OUT_ROOT}" "${POLICY}" "${dataset}" "${POLICY}" "${PREFIX_BUDGET_M}" \
    "${RESIDUAL_BUDGET}" "${POOL_K}" "${MAX_QUERIES}"
}

reader_input_report_path() {
  local dataset="$1"
  printf '%s/inputs/%s_same_seed_dense_doc_knn_reader_input_full1000.json' \
    "${READER_QA_ROOT}" "${dataset}"
}

metadata_report_path() {
  local dataset="$1"
  printf '%s/%s/reports/%s_minimal_per_query_fresh_v4_fact_witnessed_sto.json' \
    "${ETV4_ROOT}" "${dataset}" "${dataset}"
}

openie_results_path() {
  local dataset="$1"
  printf '%s/%s/index/openie_results_ner_qwen3-32b-judge.json' \
    "${ETV4_ROOT}" "${dataset}"
}

retrieval_report_path() {
  local dataset="$1"
  printf '%s/%s/reports/%s_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json' \
    "${ETV4_ROOT}" "${dataset}" "${dataset}"
}

reader_output_json() {
  local dataset="$1"
  printf '%s/reports/%s_same_seed_dense_doc_knn_gpt4omini_reader_qa_full1000.json' \
    "${READER_QA_ROOT}" "${dataset}"
}

reader_output_md() {
  local dataset="$1"
  printf '%s/reports/%s_same_seed_dense_doc_knn_gpt4omini_reader_qa_full1000.md' \
    "${READER_QA_ROOT}" "${dataset}"
}

run_reader_qa_dataset() {
  local dataset="$1"
  local source_retrieval_json retrieval_json metadata_json openie_json etv4_retrieval_json output_json output_md log_path
  source_retrieval_json="$(pcec_report_path "${dataset}")"
  retrieval_json="$(reader_input_report_path "${dataset}")"
  metadata_json="$(metadata_report_path "${dataset}")"
  openie_json="$(openie_results_path "${dataset}")"
  etv4_retrieval_json="$(retrieval_report_path "${dataset}")"
  output_json="$(reader_output_json "${dataset}")"
  output_md="$(reader_output_md "${dataset}")"
  log_path="${OUT_ROOT}/logs/04_reader_qa_${dataset}.log"

  if [[ ! -s "${source_retrieval_json}" ]]; then
    write_status "04_reader_qa_${dataset}" "FAILED missing_retrieval=${source_retrieval_json}"
    log_msg "FAILED reader QA dataset=${dataset} missing retrieval=${source_retrieval_json}"
    return 2
  fi
  if [[ ! -s "${metadata_json}" || ! -s "${openie_json}" ]]; then
    write_status "04_reader_qa_${dataset}" "FAILED missing_metadata=${metadata_json} openie=${openie_json}"
    log_msg "FAILED reader QA dataset=${dataset} missing metadata/openie"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "04_reader_qa_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP reader QA dataset=${dataset} existing output=${output_json}"
    return 0
  fi

  write_status "04_reader_qa_${dataset}" "START dataset=${dataset} model=${READER_LLM_NAME}"
  log_msg "START reader QA dataset=${dataset} model=${READER_LLM_NAME}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
metadata_path = Path(sys.argv[3])
openie_path = Path(sys.argv[4])
retrieval_path = Path(sys.argv[5])

payload = json.loads(source_path.read_text(encoding="utf-8"))
payload["input_report"] = str(metadata_path)
payload["openie_path"] = str(openie_path)
payload["retrieval_report"] = str(retrieval_path)
payload.setdefault("reader_input_patch", {})["reason"] = (
    "same-seed control PCEC report omits metadata/openie paths; "
    "reader QA requires canonical ETv4 metadata and corpus passages"
)
output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
' "${source_retrieval_json}" "${retrieval_json}" "${metadata_json}" "${openie_json}" "${etv4_retrieval_json}"
    "${PYTHON_BIN}" evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_reader_qa.py \
      --retrieval-reports "${retrieval_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${READER_QA_ROOT}/reader_runtime/${dataset}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${output_json}" \
      --output-md "${output_md}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "04_reader_qa_${dataset}" "DONE output=${output_json}"
    log_msg "DONE reader QA dataset=${dataset}"
  else
    write_status "04_reader_qa_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED reader QA dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

main() {
  write_status "04_reader_qa_first2_launcher" "START datasets=${DATASETS}"
  log_msg "BEGIN same-seed dense KNN GPT-4o-mini reader; datasets=${DATASETS}"
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    write_status "04_reader_qa_first2_launcher" "FAILED missing OPENAI_API_KEY"
    log_msg "FAILED missing OPENAI_API_KEY; expected SECRET_ENV=${SECRET_ENV}"
    return 1
  fi

  local pids=()
  local dataset pid status=0
  for dataset in ${DATASETS}; do
    run_reader_qa_dataset "${dataset}" &
    pid="$!"
    pids+=("${pid}")
    log_msg "LAUNCHED reader worker dataset=${dataset} pid=${pid}"
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done

  if [[ "${status}" -eq 0 ]]; then
    write_status "04_reader_qa_first2_launcher" "DONE datasets=${DATASETS}"
    log_msg "DONE same-seed dense KNN GPT-4o-mini reader first2"
  else
    write_status "04_reader_qa_first2_launcher" "FAILED datasets=${DATASETS}"
    log_msg "FAILED same-seed dense KNN GPT-4o-mini reader first2"
  fi
  return "${status}"
}

main "$@"
