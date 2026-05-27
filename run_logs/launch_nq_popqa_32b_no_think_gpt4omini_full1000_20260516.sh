#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
RUN_TAG="${RUN_TAG:-nq_popqa_32b_no_think_gpt4omini_full1000_20260516}"
OUT_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

DATASETS="${DATASETS:-nq popqa}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
BASELINE_POOL_K="${BASELINE_POOL_K:-200}"
EVIDENCE_POOL_K="${EVIDENCE_POOL_K:-100}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-200}"
QA_TOP_K="${QA_TOP_K:-5}"
READER_BUDGET_K="${READER_BUDGET_K:-5}"
PREFIX_BUDGET_M="${PREFIX_BUDGET_M:-4}"
ET_CANDIDATE_POOL_K="${ET_CANDIDATE_POOL_K:-200}"
OPENIE_MAX_NEW_TOKENS="${OPENIE_MAX_NEW_TOKENS:-2048}"
MAX_RETRY_ATTEMPTS="${MAX_RETRY_ATTEMPTS:-20}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"

GRAPH_LLM_NAME="${GRAPH_LLM_NAME:-qwen3-32b-judge}"
QWEN32_URL_A="${QWEN32_URL_A:-http://localhost:8045/v1}"
QWEN32_URL_B="${QWEN32_URL_B:-http://localhost:8046/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EXPANDER_EMBEDDING_NAME="${EXPANDER_EMBEDDING_NAME:-nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
READER_MAX_NEW_TOKENS="${READER_MAX_NEW_TOKENS:-none}"

HIPPO_SAVE_PREFIX="${HIPPO_SAVE_PREFIX:-${ROOT_DIR}/outputs_hipporag_qwen32b_nothink_nq_popqa_20260516}"
PROPRAG_SAVE_DIR="${PROPRAG_SAVE_DIR:-/mnt/nvme/code/PropRAG/outputs_proprag_qwen32b_nothink_nq_popqa_20260516}"
ETV4_ROOT="${ETV4_ROOT:-${OUT_ROOT}/etv4_clean_mainline}"
NEOCORRAG_ROOT="${NEOCORRAG_ROOT:-/mnt/nvme/code/NeocorRAG}"
RERETRIEVAL_LLM_NAME="${RERETRIEVAL_LLM_NAME:-/mnt/nvme/Qwen3-32B}"
NEOCORRAG_CUDA_VISIBLE_DEVICES="${NEOCORRAG_CUDA_VISIBLE_DEVICES:-4}"
RUN_NEOCOR="${RUN_NEOCOR:-1}"

if [[ -s "${SECRET_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRET_ENV}"
  set +a
fi

mkdir -p \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/status" \
  "${OUT_ROOT}/pools/dense" \
  "${OUT_ROOT}/pools/hipporag" \
  "${OUT_ROOT}/pools/proprag" \
  "${OUT_ROOT}/reader_inputs/dense" \
  "${OUT_ROOT}/reader_inputs/hipporag" \
  "${OUT_ROOT}/reader_inputs/proprag" \
  "${OUT_ROOT}/reader_inputs/neocorrag" \
  "${OUT_ROOT}/reader_qa/dense" \
  "${OUT_ROOT}/reader_qa/hipporag" \
  "${OUT_ROOT}/reader_qa/proprag" \
  "${OUT_ROOT}/reader_qa/evidenceflow" \
  "${OUT_ROOT}/reader_qa/neocorrag" \
  "${OUT_ROOT}/reader_runtime" \
  "${OUT_ROOT}/evidenceflow/pools" \
  "${OUT_ROOT}/evidenceflow/dbec/evals" \
  "${OUT_ROOT}/evidenceflow/pcec/evals" \
  "${OUT_ROOT}/evidenceflow/pcec/runtime" \
  "${OUT_ROOT}/neocorrag"

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

qwen_url_for_dataset() {
  case "$1" in
    nq|nq_rear) printf '%s\n' "${QWEN32_URL_A}" ;;
    popqa) printf '%s\n' "${QWEN32_URL_B}" ;;
    *) printf '%s\n' "${QWEN32_URL_A}" ;;
  esac
}

evidence_dataset_name() {
  case "$1" in
    nq) printf 'nq_rear\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

display_dataset_name() {
  case "$1" in
    nq_rear) printf 'nq\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

baseline_pool_path() {
  local method="$1"
  local dataset="$2"
  printf '%s/pools/%s/%s_%s_qwen32b_no_think_pool%s.json' \
    "${OUT_ROOT}" "${method}" "${dataset}" "${method}" "${BASELINE_POOL_K}"
}

baseline_reader_input_path() {
  local method="$1"
  local dataset="$2"
  printf '%s/reader_inputs/%s/%s_%s_qwen32b_no_think_pool%s_reader_input.json' \
    "${OUT_ROOT}" "${method}" "${dataset}" "${method}" "${BASELINE_POOL_K}"
}

baseline_reader_report_path() {
  local method="$1"
  local dataset="$2"
  printf '%s/reader_qa/%s/%s_%s_qwen32b_no_think_pool%s_gpt4omini_reader_top%s.json' \
    "${OUT_ROOT}" "${method}" "${dataset}" "${method}" "${BASELINE_POOL_K}" "${QA_TOP_K}"
}

baseline_reader_report_md_path() {
  local method="$1"
  local dataset="$2"
  printf '%s/reader_qa/%s/%s_%s_qwen32b_no_think_pool%s_gpt4omini_reader_top%s.md' \
    "${OUT_ROOT}" "${method}" "${dataset}" "${method}" "${BASELINE_POOL_K}" "${QA_TOP_K}"
}

etv4_retrieval_report_path() {
  local dataset="$1"
  printf '%s/%s/reports/%s_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json' \
    "${ETV4_ROOT}" "${dataset}" "${dataset}"
}

etv4_openie_path() {
  local dataset="$1"
  printf '%s/%s/index/openie_results_ner_%s.json' "${ETV4_ROOT}" "${dataset}" "${GRAPH_LLM_NAME}"
}

evidence_pool_path() {
  local dataset="$1"
  printf '%s/evidenceflow/pools/%s_etv4_pool%s_limit%s.json' \
    "${OUT_ROOT}" "${dataset}" "${EVIDENCE_POOL_K}" "${MAX_QUERIES}"
}

dbec_report_path() {
  local dataset="$1"
  printf '%s/evidenceflow/dbec/evals/%s_etv4_pool%s_dbec_qwen32b_stable_limit%s.json' \
    "${OUT_ROOT}" "${dataset}" "${EVIDENCE_POOL_K}" "${MAX_QUERIES}"
}

dbec_binding_cache_path() {
  local dataset="$1"
  printf '%s/evidenceflow/dbec/evals/%s_etv4_pool%s_dbec_qwen32b.binding_cache.json' \
    "${OUT_ROOT}" "${dataset}" "${EVIDENCE_POOL_K}"
}

pcec_report_path() {
  local dataset="$1"
  printf '%s/evidenceflow/pcec/evals/%s_pcec_native_pool_prefix%s_residual%s_pool%s_limit%s.json' \
    "${OUT_ROOT}" "${dataset}" "${PREFIX_BUDGET_M}" \
    "$((READER_BUDGET_K - PREFIX_BUDGET_M))" "${EVIDENCE_POOL_K}" "${MAX_QUERIES}"
}

evidence_reader_report_path() {
  local display_dataset="$1"
  printf '%s/reader_qa/evidenceflow/%s_evidenceflow_qwen32b_no_think_gpt4omini_reader_top%s_full1000.json' \
    "${OUT_ROOT}" "${display_dataset}" "${QA_TOP_K}"
}

evidence_reader_report_md_path() {
  local display_dataset="$1"
  printf '%s/reader_qa/evidenceflow/%s_evidenceflow_qwen32b_no_think_gpt4omini_reader_top%s_full1000.md' \
    "${OUT_ROOT}" "${display_dataset}" "${QA_TOP_K}"
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

export_hipporag_dataset() {
  local dataset="$1"
  local llm_base_url="$2"
  local output_json log_path
  output_json="$(baseline_pool_path hipporag "${dataset}")"
  log_path="${OUT_ROOT}/logs/export_hipporag_${dataset}.log"
  if [[ -s "${output_json}" ]]; then
    write_status "export_hipporag_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  run_logged "export_hipporag dataset=${dataset}" "export_hipporag_${dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" scripts/export_hipporag_pool.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --pool_k "${BASELINE_POOL_K}" \
      --save_dir "${HIPPO_SAVE_PREFIX}" \
      --llm_name "${GRAPH_LLM_NAME}" \
      --llm_request_name "${GRAPH_LLM_NAME}" \
      --llm_base_url "${llm_base_url}" \
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
}

export_dense_dataset() {
  local dataset="$1"
  local output_json log_path
  output_json="$(baseline_pool_path dense "${dataset}")"
  log_path="${OUT_ROOT}/logs/export_dense_${dataset}.log"
  if [[ -s "${output_json}" ]]; then
    write_status "export_dense_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  run_logged "export_dense dataset=${dataset}" "export_dense_${dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" scripts/export_dense_pool.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --pool_k "${BASELINE_POOL_K}" \
      --save_dir "${HIPPO_SAVE_PREFIX}" \
      --llm_name "${GRAPH_LLM_NAME}" \
      --embedding_name "${EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --embedding_batch_size 32 \
      --output_json "${output_json}"
}

export_proprag_dataset() {
  local dataset="$1"
  local llm_base_url="$2"
  local output_json log_path
  output_json="$(baseline_pool_path proprag "${dataset}")"
  log_path="${OUT_ROOT}/logs/export_proprag_${dataset}.log"
  if [[ -s "${output_json}" ]]; then
    write_status "export_proprag_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  run_logged "export_proprag dataset=${dataset}" "export_proprag_${dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" scripts/export_proprag_pool.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --pool_k "${BASELINE_POOL_K}" \
      --llm_name "${GRAPH_LLM_NAME}" \
      --llm_base_url "${llm_base_url}" \
      --embedding_name "${EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
      --save_dir "${PROPRAG_SAVE_DIR}" \
      --retrieval_top_k "${RETRIEVAL_TOP_K}" \
      --qa_top_k "${QA_TOP_K}" \
      --max_new_tokens "${OPENIE_MAX_NEW_TOKENS}" \
      --reuse_preextracted_openie false \
      --openie_llm_name "${GRAPH_LLM_NAME}" \
      --force_index_from_scratch true \
      --force_openie_from_scratch true \
      --openie_mode online \
      --use_propositions true \
      --use_beam_search true \
      --beam_width 4 \
      --max_path_length 3 \
      --second_stage_filter_k 40 \
      --sim_threshold 0.75 \
      --qwen_disable_thinking \
      --output_json "${output_json}"
}

wrap_pool_reader_input() {
  local method="$1"
  local dataset="$2"
  local pool_json input_json log_path method_name
  pool_json="$(baseline_pool_path "${method}" "${dataset}")"
  input_json="$(baseline_reader_input_path "${method}" "${dataset}")"
  log_path="${OUT_ROOT}/logs/wrap_${method}_${dataset}.log"
  method_name="${method}_qwen32b_no_think_top${BASELINE_POOL_K}"
  if [[ "${method}" == "dense" ]]; then
    method_name="dense_entry"
  fi
  if [[ -s "${input_json}" ]]; then
    write_status "wrap_${method}_${dataset}" "SKIP existing input=${input_json}"
    return 0
  fi
  run_logged "wrap_reader_input method=${method} dataset=${dataset}" "wrap_${method}_${dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" scripts/make_external_pool_reader_inputs.py \
      --pool-json "${pool_json}" \
      --method-name "${method_name}" \
      --output-json "${input_json}"
}

run_baseline_reader_dataset() {
  local method="$1"
  local dataset="$2"
  local input_json output_json output_md log_path method_name
  input_json="$(baseline_reader_input_path "${method}" "${dataset}")"
  output_json="$(baseline_reader_report_path "${method}" "${dataset}")"
  output_md="$(baseline_reader_report_md_path "${method}" "${dataset}")"
  log_path="${OUT_ROOT}/logs/reader_${method}_${dataset}.log"
  method_name="${method}_qwen32b_no_think_top${BASELINE_POOL_K}"
  if [[ "${method}" == "dense" ]]; then
    method_name="dense_entry"
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "reader_${method}_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  run_logged "reader method=${method} dataset=${dataset}" "reader_${method}_${dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py \
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
}

run_etv4_fresh_dataset() {
  local dataset="$1"
  local llm_base_url="$2"
  local output_json log_path
  output_json="$(etv4_retrieval_report_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/etv4_fresh_${dataset}.log"
  if [[ -s "${output_json}" ]]; then
    write_status "etv4_fresh_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  run_logged "etv4_fresh dataset=${dataset}" "etv4_fresh_${dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_fresh_e2e.py \
      --datasets "${dataset}" \
      --output-root "${ETV4_ROOT}" \
      --max-queries "${MAX_QUERIES}" \
      --llm-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${llm_base_url}" \
      --max-new-tokens "${OPENIE_MAX_NEW_TOKENS}" \
      --max-retry-attempts "${MAX_RETRY_ATTEMPTS}" \
      --qwen-disable-thinking \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --readout-policy clean_mainline \
      --candidate-pool-k "${RETRIEVAL_TOP_K}" \
      --dense-root-count 20 \
      --top-k "${QA_TOP_K}"
}

export_evidence_pool_dataset() {
  local dataset="$1"
  local output_json log_path
  output_json="$(evidence_pool_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/export_evidence_pool_${dataset}.log"
  if [[ -s "${output_json}" ]]; then
    write_status "export_evidence_pool_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  run_logged "export_evidence_pool dataset=${dataset}" "export_evidence_pool_${dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" scripts/export_evidence_transition_pool.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --pool_k "${EVIDENCE_POOL_K}" \
      --data_root reproduce/dataset \
      --retrieval_report "$(etv4_retrieval_report_path "${dataset}")" \
      --openie_results "$(etv4_openie_path "${dataset}")" \
      --output_json "${output_json}"
}

run_dbec_dataset() {
  local dataset="$1"
  local llm_base_url="$2"
  local output_json binding_json log_path
  output_json="$(dbec_report_path "${dataset}")"
  binding_json="$(dbec_binding_cache_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/dbec32_${dataset}.log"
  if [[ -s "${output_json}" && -s "${binding_json}" ]]; then
    write_status "dbec32_${dataset}" "SKIP existing report=${output_json} binding=${binding_json}"
    return 0
  fi
  run_logged "dbec32 dataset=${dataset}" "dbec32_${dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv3_dbec_latest/run_eval.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --pool-k "${EVIDENCE_POOL_K}" \
      --setwise-pool-k "${EVIDENCE_POOL_K}" \
      --qa-top-k "${QA_TOP_K}" \
      --pool-json "$(evidence_pool_path "${dataset}")" \
      --output-root "${OUT_ROOT}/evidenceflow/dbec" \
      --output-json "${output_json}" \
      --binding-cache-path "${binding_json}" \
      --save-dir outputs_step0_general_nvembed \
      --llm-name "${GRAPH_LLM_NAME}" \
      --llm-request-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${llm_base_url}" \
      --llm-binding-url "${llm_base_url}" \
      --llm-binding-model "${GRAPH_LLM_NAME}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --qwen-disable-thinking
}

run_pcec_dataset() {
  local dataset="$1"
  local llm_base_url="$2"
  local output_json log_path
  output_json="$(pcec_report_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/pcec_${dataset}.log"
  if [[ -s "${output_json}" ]]; then
    write_status "pcec_${dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  run_logged "pcec dataset=${dataset}" "pcec_${dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_composition/run_native_pool.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --max-queries "${MAX_QUERIES}" \
      --pool-k "${EVIDENCE_POOL_K}" \
      --reader-budget-k "${READER_BUDGET_K}" \
      --prefix-budget-m "${PREFIX_BUDGET_M}" \
      --pool-json "$(evidence_pool_path "${dataset}")" \
      --requirement-report "$(dbec_report_path "${dataset}")" \
      --binding-cache-path "$(dbec_binding_cache_path "${dataset}")" \
      --output-json "${output_json}" \
      --output-root "${OUT_ROOT}/evidenceflow/pcec" \
      --data-root reproduce/dataset \
      --save-dir "${OUT_ROOT}/evidenceflow/pcec/runtime/${dataset}" \
      --llm-name "${GRAPH_LLM_NAME}" \
      --llm-request-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${llm_base_url}" \
      --llm-binding-model "${GRAPH_LLM_NAME}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --qwen-disable-thinking
}

run_evidence_reader_dataset() {
  local dataset="$1"
  local display_dataset output_json output_md log_path
  display_dataset="$(display_dataset_name "${dataset}")"
  output_json="$(evidence_reader_report_path "${display_dataset}")"
  output_md="$(evidence_reader_report_md_path "${display_dataset}")"
  log_path="${OUT_ROOT}/logs/reader_evidenceflow_${display_dataset}.log"
  if [[ -s "${output_json}" ]]; then
    write_status "reader_evidenceflow_${display_dataset}" "SKIP existing output=${output_json}"
    return 0
  fi
  run_logged "reader evidenceflow dataset=${dataset}" "reader_evidenceflow_${display_dataset}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_reader_qa.py \
      --retrieval-reports "$(pcec_report_path "${dataset}")" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${OUT_ROOT}/reader_runtime/evidenceflow/${display_dataset}" \
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

run_non_neocor_lane() {
  local dataset="$1"
  local llm_base_url evidence_dataset display_dataset
  llm_base_url="$(qwen_url_for_dataset "${dataset}")"
  evidence_dataset="$(evidence_dataset_name "${dataset}")"
  display_dataset="$(display_dataset_name "${evidence_dataset}")"

  write_status "lane_${display_dataset}" "START endpoint=${llm_base_url}"
  log_msg "START lane dataset=${dataset} evidence_dataset=${evidence_dataset} endpoint=${llm_base_url}"

  export_hipporag_dataset "${dataset}" "${llm_base_url}" || return 1
  export_dense_dataset "${dataset}" || return 1
  export_proprag_dataset "${dataset}" "${llm_base_url}" || return 1

  for method in dense hipporag proprag; do
    wrap_pool_reader_input "${method}" "${dataset}" || return 1
    run_baseline_reader_dataset "${method}" "${dataset}" || return 1
  done

  run_etv4_fresh_dataset "${evidence_dataset}" "${llm_base_url}" || return 1
  export_evidence_pool_dataset "${evidence_dataset}" || return 1
  run_dbec_dataset "${evidence_dataset}" "${llm_base_url}" || return 1
  run_pcec_dataset "${evidence_dataset}" "${llm_base_url}" || return 1
  run_evidence_reader_dataset "${evidence_dataset}" || return 1

  write_status "lane_${display_dataset}" "DONE endpoint=${llm_base_url}"
  log_msg "DONE lane dataset=${dataset}"
}

neocor_output_path() {
  local dataset="$1"
  local display_dataset
  display_dataset="$(display_dataset_name "${dataset}")"
  printf '%s/neocorrag/%s_neocorrag_k3_qwen32b_no_think.json' "${OUT_ROOT}" "${display_dataset}"
}

neocor_reader_input_path() {
  local dataset="$1"
  local display_dataset
  display_dataset="$(display_dataset_name "${dataset}")"
  printf '%s/reader_inputs/neocorrag/%s_neocorrag_qwen32b_no_think_top%s_reader_input.json' \
    "${OUT_ROOT}" "${display_dataset}" "${QA_TOP_K}"
}

neocor_reader_report_path() {
  local dataset="$1"
  local display_dataset
  display_dataset="$(display_dataset_name "${dataset}")"
  printf '%s/reader_qa/neocorrag/%s_neocorrag_qwen32b_no_think_gpt4omini_reader_top%s_full1000.json' \
    "${OUT_ROOT}" "${display_dataset}" "${QA_TOP_K}"
}

neocor_reader_report_md_path() {
  local dataset="$1"
  local display_dataset
  display_dataset="$(display_dataset_name "${dataset}")"
  printf '%s/reader_qa/neocorrag/%s_neocorrag_qwen32b_no_think_gpt4omini_reader_top%s_full1000.md' \
    "${OUT_ROOT}" "${display_dataset}" "${QA_TOP_K}"
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
    env PYTHONPATH=. CUDA_VISIBLE_DEVICES="${NEOCORRAG_CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" scripts/run_neocorrag_aligned.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --neocorrag_root "${NEOCORRAG_ROOT}" \
      --output_json "${output_json}" \
      --output_method_name "aligned_limit${MAX_QUERIES}_${GRAPH_LLM_NAME}_beam_k3_ret100_reretrieval32b_nvembedv2_nq_popqa_20260516" \
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
      --retrieval_top_k 100 \
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

run_neocor_all() {
  local dataset evidence_dataset
  if [[ "${RUN_NEOCOR}" != "1" ]]; then
    log_msg "SKIP NeocorRAG because RUN_NEOCOR=${RUN_NEOCOR}"
    return 0
  fi
  for dataset in ${DATASETS}; do
    evidence_dataset="$(evidence_dataset_name "${dataset}")"
    run_neocor_retrieval_dataset "${evidence_dataset}" || return 1
    wrap_neocor_reader_input "${evidence_dataset}" || return 1
    run_neocor_reader_dataset "${evidence_dataset}" || return 1
  done
}

write_summary() {
  env OUT_ROOT="${OUT_ROOT}" PYTHON_BIN="${PYTHON_BIN}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT_ROOT"])
rows = []
for method in ("dense", "hipporag", "proprag", "evidenceflow", "neocorrag"):
    root = out / "reader_qa" / method
    if not root.exists():
        continue
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for dataset in payload.get("datasets", []) or []:
            label = dataset.get("dataset")
            if label == "nq_rear":
                label = "nq"
            for method_name, method_payload in (dataset.get("methods") or {}).items():
                metrics = method_payload.get("metrics") or {}
                rows.append({
                    "dataset": label,
                    "method_group": method,
                    "method": method_name,
                    "count": metrics.get("count"),
                    "Recall@5": metrics.get("Recall@5") or metrics.get("r5"),
                    "Recall@20": metrics.get("Recall@20"),
                    "Recall@100": metrics.get("Recall@100"),
                    "Recall@200": metrics.get("Recall@200"),
                    "EM": metrics.get("ExactMatch"),
                    "F1": metrics.get("F1"),
                    "reader_docs": metrics.get("mean_reader_docs"),
                    "report": str(path),
                })
summary = {"rows": rows}
(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
lines = [
    "# NQ/PopQA 32B no_think + GPT-4o-mini Summary",
    "",
    "| Dataset | Method group | Method | Count | R@5 | R@20 | R@100 | R@200 | EM | F1 | Reader docs |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
def fmt(value):
    return "" if value is None else f"{float(value):.4f}"
for row in rows:
    lines.append(
        f"| {row['dataset']} | {row['method_group']} | {row['method']} | {int(row['count'] or 0)} | "
        f"{fmt(row['Recall@5'])} | {fmt(row['Recall@20'])} | {fmt(row['Recall@100'])} | {fmt(row['Recall@200'])} | "
        f"{fmt(row['EM'])} | {fmt(row['F1'])} | {fmt(row['reader_docs'])} |"
    )
(out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"summary": str(out / "summary.json"), "rows": len(rows)}, indent=2))
PY
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: Qwen3-32B no_think for non-reader work; GPT-4o-mini reader; datasets=${DATASETS}; Neocor last=${RUN_NEOCOR}"

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

  local pids=()
  local dataset pid rc=0
  for dataset in ${DATASETS}; do
    run_non_neocor_lane "${dataset}" > "${OUT_ROOT}/logs/lane_${dataset}.log" 2>&1 &
    pids+=("$!")
    log_msg "LAUNCHED lane dataset=${dataset} pid=$!"
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  if [[ "${rc}" -ne 0 ]]; then
    write_status "launcher" "FAILED non_neocor rc=${rc}"
    log_msg "FAILED non-Neocor lanes rc=${rc}"
    return 1
  fi

  run_neocor_all || {
    write_status "launcher" "FAILED neocor"
    return 1
  }

  write_summary
  write_status "launcher" "DONE run_tag=${RUN_TAG}"
  log_msg "DONE ${RUN_TAG}"
}

main "$@"
