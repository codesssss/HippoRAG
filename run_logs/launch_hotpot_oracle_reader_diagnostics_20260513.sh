#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"

RUN_DIR="${ROOT_DIR}/run_logs/hotpotqa_reader_diagnostics_20260513"
INPUT_DIR="${RUN_DIR}/inputs"
OUTPUT_DIR="${RUN_DIR}/outputs"
RUNTIME_DIR="${RUN_DIR}/reader_runtime"
LOG_DIR="${RUN_DIR}/logs"
STATUS_DIR="${RUN_DIR}/status"

MAX_QUERIES="${MAX_QUERIES:-1000}"
QA_TOP_K="${QA_TOP_K:-5}"
READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
KEY_SOURCE_SESSION="${KEY_SOURCE_SESSION:-baseline_qwen32b_top200_20260512}"

mkdir -p "${OUTPUT_DIR}" "${RUNTIME_DIR}" "${LOG_DIR}" "${STATUS_DIR}"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOG_DIR}/launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${STATUS_DIR}/${name}.status"
}

load_api_key_from_tmux() {
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    return 0
  fi
  if ! command -v tmux >/dev/null 2>&1; then
    return 1
  fi
  local line
  line="$(tmux show-environment -t "${KEY_SOURCE_SESSION}" OPENAI_API_KEY 2>/dev/null || true)"
  if [[ "${line}" == OPENAI_API_KEY=* ]]; then
    export OPENAI_API_KEY="${line#OPENAI_API_KEY=}"
  fi
  [[ -n "${OPENAI_API_KEY:-}" ]]
}

run_policy() {
  local policy="$1"
  local input_json="${INPUT_DIR}/hotpotqa_etv4_oracle_${policy}_reader_diagnostic.json"
  local output_json="${OUTPUT_DIR}/hotpotqa_etv4_oracle_${policy}_gpt4omini_reader_qa.json"
  local output_md="${OUTPUT_DIR}/hotpotqa_etv4_oracle_${policy}_gpt4omini_reader_qa.md"
  local log_path="${LOG_DIR}/reader_${policy}.log"

  if [[ ! -s "${input_json}" ]]; then
    write_status "${policy}" "FAILED missing input=${input_json}"
    log_msg "FAILED ${policy}; missing input=${input_json}"
    return 1
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "${policy}" "SKIP existing output=${output_json}"
    log_msg "SKIP ${policy}; existing output=${output_json}"
    return 0
  fi

  write_status "${policy}" "START input=${input_json} output=${output_json}"
  log_msg "START HotpotQA oracle reader policy=${policy}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_reader_qa.py \
      --retrieval-reports "${input_json}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --save-dir "${RUNTIME_DIR}/${policy}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens none \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${output_json}" \
      --output-md "${output_md}"
  ) > "${log_path}" 2>&1
  local code=$?

  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "${policy}" "DONE output=${output_json}"
    log_msg "DONE HotpotQA oracle reader policy=${policy}"
  else
    write_status "${policy}" "FAILED code=${code} log=${log_path}"
    log_msg "FAILED HotpotQA oracle reader policy=${policy} code=${code} log=${log_path}"
    return "${code}"
  fi
}

main() {
  write_status "launcher" "START"
  if ! load_api_key_from_tmux; then
    write_status "launcher" "FAILED missing OPENAI_API_KEY"
    log_msg "FAILED missing OPENAI_API_KEY"
    return 1
  fi

  run_policy "gold_first" &
  local pid_gold_first=$!
  run_policy "gold_only" &
  local pid_gold_only=$!

  local rc=0
  wait "${pid_gold_first}" || rc=1
  wait "${pid_gold_only}" || rc=1

  if [[ "${rc}" -eq 0 ]]; then
    write_status "launcher" "DONE"
    log_msg "DONE HotpotQA oracle reader diagnostics"
  else
    write_status "launcher" "FAILED"
    log_msg "FAILED HotpotQA oracle reader diagnostics"
  fi
  return "${rc}"
}

main "$@"
