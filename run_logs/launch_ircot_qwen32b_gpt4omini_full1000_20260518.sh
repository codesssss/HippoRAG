#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/mnt/nvme/code/HippoRAG}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
RUN_TAG="${RUN_TAG:-ircot_qwen32b_no_think_gpt4omini_full1000_20260518}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/run_logs/${RUN_TAG}}"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
MAX_ITER="${MAX_ITER:-3}"
TOP_K_PER_ITER="${TOP_K_PER_ITER:-5}"
QA_TOP_K="${QA_TOP_K:-5}"
FINAL_DOC_ORDER="${FINAL_DOC_ORDER:-round_robin}"

GRAPH_LLM_NAME="${GRAPH_LLM_NAME:-qwen3-32b-judge}"
QWEN32_URL_A="${QWEN32_URL_A:-http://localhost:8045/v1}"
QWEN32_URL_B="${QWEN32_URL_B:-http://localhost:8046/v1}"
HIPPO_SAVE_DIR="${HIPPO_SAVE_DIR:-${ROOT_DIR}/outputs_hipporag_qwen32b_valid_graph_top200_20260513_r2}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
READER_MAX_NEW_TOKENS="${READER_MAX_NEW_TOKENS:-none}"

mkdir -p \
  "${OUT_DIR}/logs" \
  "${OUT_DIR}/status" \
  "${OUT_DIR}/reader_inputs/ircot" \
  "${OUT_DIR}/reader_qa/ircot" \
  "${OUT_DIR}/reader_runtime/ircot"

if [[ -f "${SECRET_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRET_ENV}"
  set +a
fi

export HIPPORAG_RERANK_FORCE_NO_THINK=1
export TOKENIZERS_PARALLELISM=false

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_DIR}/logs/launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee "${OUT_DIR}/status/${name}.status"
}

qwen_url_for_lane() {
  case "$1" in
    lane_b) printf '%s\n' "${QWEN32_URL_B}" ;;
    *) printf '%s\n' "${QWEN32_URL_A}" ;;
  esac
}

reader_input_path() {
  local dataset="$1"
  printf '%s/reader_inputs/ircot/%s_ircot_qwen32b_no_think_reader_input_limit%s.json\n' \
    "${OUT_DIR}" "${dataset}" "${MAX_QUERIES}"
}

reader_report_path() {
  local dataset="$1"
  printf '%s/reader_qa/ircot/%s_ircot_qwen32b_no_think_gpt4omini_reader_top%s_limit%s.json\n' \
    "${OUT_DIR}" "${dataset}" "${QA_TOP_K}" "${MAX_QUERIES}"
}

reader_report_md_path() {
  local dataset="$1"
  printf '%s/reader_qa/ircot/%s_ircot_qwen32b_no_think_gpt4omini_reader_top%s_limit%s.md\n' \
    "${OUT_DIR}" "${dataset}" "${QA_TOP_K}" "${MAX_QUERIES}"
}

check_endpoint() {
  local name="$1"
  local url="$2"
  if curl -fsS --max-time 10 "${url%/}/models" >/dev/null 2>&1; then
    log_msg "OK endpoint ${name}: ${url}"
    return 0
  fi
  write_status "endpoint_${name}" "FAILED url=${url}"
  log_msg "FAILED endpoint ${name}: ${url}"
  return 1
}

preflight() {
  local rc=0
  check_endpoint qwen32_a "${QWEN32_URL_A}" || rc=1
  check_endpoint qwen32_b "${QWEN32_URL_B}" || rc=1
  if ! curl -fsS --max-time 10 "${EMBEDDING_BASE_URL%/}/models" >/dev/null 2>&1; then
    log_msg "WARN embedding endpoint did not answer /models: ${EMBEDDING_BASE_URL}"
  else
    log_msg "OK endpoint embedding: ${EMBEDDING_BASE_URL}"
  fi
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    write_status "launcher" "FAILED missing OPENAI_API_KEY expected_secret_env=${SECRET_ENV}"
    log_msg "FAILED missing OPENAI_API_KEY"
    rc=1
  fi
  return "${rc}"
}

run_retrieval_dataset() {
  local dataset="$1"
  local lane="$2"
  local qwen_url output_json output_md log_path
  qwen_url="$(qwen_url_for_lane "${lane}")"
  output_json="$(reader_input_path "${dataset}")"
  output_md="${output_json%.json}.md"
  log_path="${OUT_DIR}/logs/retrieval_ircot_${dataset}.log"

  if [[ -s "${output_json}" ]]; then
    write_status "retrieval_ircot_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP IRCoT retrieval dataset=${dataset}"
    return 0
  fi

  write_status "retrieval_ircot_${dataset}" "START lane=${lane} qwen=${qwen_url} output=${output_json}"
  log_msg "START IRCoT retrieval dataset=${dataset} lane=${lane} qwen=${qwen_url}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. TOKENIZERS_PARALLELISM=false HIPPORAG_RERANK_FORCE_NO_THINK=1 \
      "${PYTHON_BIN}" scripts/bsgs_run_ircot_baseline.py \
        --dataset "${dataset}" \
        --limit "${MAX_QUERIES}" \
        --max_iter "${MAX_ITER}" \
        --top_k_per_iter "${TOP_K_PER_ITER}" \
        --qa_top_k "${QA_TOP_K}" \
        --final_doc_order "${FINAL_DOC_ORDER}" \
        --qa_doc_max_chars 2048 \
        --save_dir "${HIPPO_SAVE_DIR}" \
        --llm_base_url "${qwen_url}" \
        --llm_name "${GRAPH_LLM_NAME}" \
        --llm_request_name "${GRAPH_LLM_NAME}" \
        --embedding_name "${EMBEDDING_NAME}" \
        --embedding_base_url "${EMBEDDING_BASE_URL}" \
        --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
        --max_retry_attempts 20 \
        --openie_mode online \
        --skip_qa \
        --report_json "${output_json}" \
        --report_md "${output_md}"
  ) > "${log_path}" 2>&1
  local code=$?

  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "retrieval_ircot_${dataset}" "DONE output=${output_json}"
    log_msg "DONE IRCoT retrieval dataset=${dataset}"
  else
    write_status "retrieval_ircot_${dataset}" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED IRCoT retrieval dataset=${dataset} code=${code}"
    return "${code}"
  fi
}

run_reader_dataset() {
  local dataset="$1"
  local input_json output_json output_md log_path
  input_json="$(reader_input_path "${dataset}")"
  output_json="$(reader_report_path "${dataset}")"
  output_md="$(reader_report_md_path "${dataset}")"
  log_path="${OUT_DIR}/logs/reader_ircot_${dataset}.log"

  if [[ -s "${output_json}" ]]; then
    write_status "reader_ircot_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP IRCoT reader dataset=${dataset}"
    return 0
  fi
  if [[ ! -s "${input_json}" ]]; then
    write_status "reader_ircot_${dataset}" "FAILED missing input=${input_json}"
    log_msg "FAILED IRCoT reader dataset=${dataset}: missing input=${input_json}"
    return 2
  fi

  write_status "reader_ircot_${dataset}" "START input=${input_json} output=${output_json}"
  log_msg "START IRCoT GPT-4o-mini reader dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. TOKENIZERS_PARALLELISM=false \
      "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py \
        --input-files "${input_json}" \
        --method-name "ircot_qwen32b_no_think_top${QA_TOP_K}" \
        --max-queries "${MAX_QUERIES}" \
        --qa-top-k "${QA_TOP_K}" \
        --doc-source saved_docs \
        --save-dir "${OUT_DIR}/reader_runtime/ircot/${dataset}" \
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
  local code=$?

  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "reader_ircot_${dataset}" "DONE output=${output_json}"
    log_msg "DONE IRCoT GPT-4o-mini reader dataset=${dataset}"
  else
    write_status "reader_ircot_${dataset}" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED IRCoT GPT-4o-mini reader dataset=${dataset} code=${code}"
    return "${code}"
  fi
}

run_parallel_wave() {
  local phase="$1"
  shift
  local rc=0
  local pids=()
  while [[ "$#" -gt 0 ]]; do
    local dataset="$1"
    local lane="${2:-lane_a}"
    shift 2 || true
    if [[ "${phase}" == "retrieval" ]]; then
      run_retrieval_dataset "${dataset}" "${lane}" &
    else
      run_reader_dataset "${dataset}" &
    fi
    pids+=("$!")
  done

  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  return "${rc}"
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG} datasets=${DATASETS}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: IRCoT follow-up retrieval with ${GRAPH_LLM_NAME} /no_think, reader=${READER_LLM_NAME}, top=${QA_TOP_K}"
  preflight || {
    write_status "launcher" "FAILED preflight"
    return 1
  }

  local datasets=(${DATASETS})
  local rc=0

  local i=0
  while [[ "${i}" -lt "${#datasets[@]}" ]]; do
    local args=()
    args+=("${datasets[$i]}" lane_a)
    if [[ "$((i + 1))" -lt "${#datasets[@]}" ]]; then
      args+=("${datasets[$((i + 1))]}" lane_b)
    fi
    run_parallel_wave retrieval "${args[@]}" || rc=1
    i=$((i + 2))
  done
  if [[ "${rc}" -ne 0 ]]; then
    write_status "launcher" "FAILED retrieval"
    return "${rc}"
  fi

  local reader_args=()
  for dataset in "${datasets[@]}"; do
    reader_args+=("${dataset}" lane_a)
  done
  run_parallel_wave reader "${reader_args[@]}" || rc=1

  if [[ "${rc}" -eq 0 ]]; then
    write_status "launcher" "DONE run_tag=${RUN_TAG}"
    log_msg "DONE ${RUN_TAG}"
  else
    write_status "launcher" "FAILED reader"
    log_msg "FAILED ${RUN_TAG}"
  fi
  return "${rc}"
}

main "$@"
