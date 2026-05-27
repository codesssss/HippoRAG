#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
RUN_TAG="neocorrag_aligned_k3_all32_full1000_20260514"
OUT_DIR="${ROOT_DIR}/run_logs/${RUN_TAG}"
NEOCORRAG_ROOT="${NEOCORRAG_ROOT:-/mnt/nvme/code/NeocorRAG}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
LLM_NAME="${LLM_NAME:-qwen3-32b-judge}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8045/v1}"
RERETRIEVAL_LLM_NAME="${RERETRIEVAL_LLM_NAME:-/mnt/nvme/Qwen3-32B}"
CUDA_DEVICE="${NEOCORRAG_CUDA_VISIBLE_DEVICES:-4}"
DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"

mkdir -p "${OUT_DIR}"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_DIR}/launcher.log"
}

write_status() {
  printf '[%s] %s\n' "$(date -Is)" "$*" > "${OUT_DIR}/launcher.status"
}

check_endpoint() {
  local name="$1"
  local url="$2"
  if curl -fsS "${url}" >/dev/null 2>&1; then
    log_msg "OK endpoint ${name}: ${url}"
    return 0
  fi
  log_msg "FAILED endpoint ${name}: ${url}"
  return 1
}

is_done_json() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  "${PYTHON_BIN}" - "${path}" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
ok = (
    int(data.get("limit", 0) or 0) >= 1000
    and bool(data.get("overall_retrieval_result"))
    and bool(data.get("overall_qa_results"))
)
sys.exit(0 if ok else 1)
PY
}

run_one() {
  local dataset="$1"
  local output_json="${OUT_DIR}/${dataset}_neocorrag_k3_all32.json"
  local log_path="${OUT_DIR}/${dataset}_neocorrag_k3_all32.log"
  local status_path="${OUT_DIR}/${dataset}_neocorrag_k3_all32.status"

  if is_done_json "${output_json}"; then
    printf '[%s] SKIP existing output=%s\n' "$(date -Is)" "${output_json}" | tee "${status_path}"
    return 0
  fi

  printf '[%s] START dataset=%s variant=neocorrag_k3_all32_full1000 cuda=%s reretrieval=%s\n' \
    "$(date -Is)" "${dataset}" "${CUDA_DEVICE}" "${RERETRIEVAL_LLM_NAME}" | tee "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" scripts/run_neocorrag_aligned.py \
      --dataset "${dataset}" \
      --limit 1000 \
      --neocorrag_root "${NEOCORRAG_ROOT}" \
      --output_json "${output_json}" \
      --output_method_name "aligned_limit1000_${LLM_NAME}_beam_k3_ret100_reretrieval32b_nvembedv2_20260514" \
      --llm_name "${LLM_NAME}" \
      --llm_base_url "${LLM_BASE_URL}" \
      --graph_llm_name "${LLM_NAME}" \
      --graph_llm_base_url "${LLM_BASE_URL}" \
      --embedding_name nvidia/NV-Embed-v2 \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --reretrieval_llm_name "${RERETRIEVAL_LLM_NAME}" \
      --reretrieval_embedding_name nvidia/NV-Embed-v2 \
      --generation_mode beam \
      --k 3 \
      --qa_top_k 5 \
      --retrieval_top_k 100 \
      --embedding_batch_size 4 \
      --max_new_tokens 2048
  ) > "${log_path}" 2>&1
  local code=$?

  if [[ "${code}" -eq 0 ]] && is_done_json "${output_json}"; then
    printf '[%s] DONE dataset=%s output=%s\n' "$(date -Is)" "${dataset}" "${output_json}" | tee "${status_path}"
  else
    printf '[%s] FAILED dataset=%s code=%s log=%s\n' "$(date -Is)" "${dataset}" "${code}" "${log_path}" | tee "${status_path}"
    return "${code}"
  fi
}

main() {
  write_status "START run_tag=${RUN_TAG} datasets=${DATASETS}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: NeocorRAG k=3 full1000 all32; API LLM/graph LLM=${LLM_NAME} at ${LLM_BASE_URL}; local reretrieval=${RERETRIEVAL_LLM_NAME}"

  local preflight=0
  check_endpoint "qwen32b" "${LLM_BASE_URL}/models" || preflight=1
  check_endpoint "nv_embed" "http://localhost:8019/v1/models" || preflight=1
  if [[ "${preflight}" -ne 0 ]]; then
    write_status "FAILED preflight"
    return 1
  fi

  local rc=0
  local dataset
  for dataset in ${DATASETS}; do
    run_one "${dataset}" || {
      rc=1
      break
    }
  done

  if [[ "${rc}" -eq 0 ]]; then
    write_status "DONE run_tag=${RUN_TAG}"
    log_msg "DONE ${RUN_TAG}"
  else
    write_status "FAILED run_tag=${RUN_TAG} rc=${rc}"
    log_msg "FAILED ${RUN_TAG} rc=${rc}"
  fi
  return "${rc}"
}

main "$@"
