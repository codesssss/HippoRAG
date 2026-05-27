#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/agsto_qwen32b_nvembed_sync_limit100_20260505"
SAVE_DIR="outputs_step0_general_nvembed"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
EMBEDDING_API_MODEL="nvidia/NV-Embed-v2"
EVAL_EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"
LLM_MODEL="qwen3-32b-judge"
LLM_NAME="qwen3-32b"
LLM_PORT="${LLM_PORT:-8045}"
LIMIT="${LIMIT:-100}"
POOL_K="${POOL_K:-100}"
MATCH_MODE="${MATCH_MODE:-wiki_title}"
OPENIE_WORKERS="${OPENIE_WORKERS:-8}"
OPENIE_MAX_TOKENS="${OPENIE_MAX_TOKENS:-768}"

export HIPPORAG_RERANK_FORCE_NO_THINK=1
mkdir -p "${OUT_DIR}/openie" "${OUT_DIR}/pools" "${OUT_DIR}/evals" "${OUT_DIR}/logs" "${OUT_DIR}/status"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_DIR}/logs/launcher.log"
}

corpus_json() {
  echo "${ROOT_DIR}/reproduce/dataset/${1}_corpus.json"
}

expected_doc_count() {
  local dataset="$1"
  "${PYTHON_BIN}" - "${ROOT_DIR}/reproduce/dataset/${dataset}_corpus.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
rows = payload.get("docs", payload) if isinstance(payload, dict) else payload
print(len(rows))
PY
}

openie_ready() {
  local openie_json="$1"
  local expected_docs="$2"
  [[ -s "${openie_json}" ]] || return 1
  "${PYTHON_BIN}" - "${openie_json}" "${expected_docs}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
payload = json.loads(path.read_text())
metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
docs = payload.get("docs", []) if isinstance(payload, dict) else []
ok = (
    isinstance(docs, list)
    and len(docs) == expected
    and int(metadata.get("doc_count", -1)) == expected
    and not bool(metadata.get("partial", False))
)
raise SystemExit(0 if ok else 1)
PY
}

build_openie() {
  local dataset="$1"
  local openie_json="${OUT_DIR}/openie/${dataset}_agsto_openie_qwen3_32b_no_think.json"
  local status_path="${OUT_DIR}/status/${dataset}_openie.status"
  local log_path="${OUT_DIR}/logs/${dataset}_openie.log"
  local expected
  expected="$(expected_doc_count "${dataset}")"

  if openie_ready "${openie_json}" "${expected}"; then
    log_msg "SKIP openie dataset=${dataset} existing=${openie_json}"
    echo "[SKIP] dataset=${dataset} stage=openie existing=${openie_json} docs=${expected} time=$(date -Is)" > "${status_path}"
    return 0
  fi

  log_msg "START openie dataset=${dataset} docs=${expected} model=${LLM_MODEL} port=${LLM_PORT} workers=${OPENIE_WORKERS}"
  echo "[START] dataset=${dataset} stage=openie docs=${expected} model=${LLM_MODEL} port=${LLM_PORT} workers=${OPENIE_WORKERS} time=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" src/agsto_v12/build_openie.py \
      --corpus-json "$(corpus_json "${dataset}")" \
      --output-openie-json "${openie_json}" \
      --mode llm \
      --workers "${OPENIE_WORKERS}" \
      --checkpoint-every 100 \
      --resume \
      --continue-on-error \
      --llm-base-url "http://localhost:${LLM_PORT}/v1" \
      --llm-model "${LLM_MODEL}" \
      --allow-empty-api-key \
      --qwen-disable-thinking \
      --timeout 180 \
      --max-tokens "${OPENIE_MAX_TOKENS}" \
      --retries 3
  ) > "${log_path}" 2>&1

  if openie_ready "${openie_json}" "${expected}"; then
    echo "[DONE] dataset=${dataset} stage=openie docs=${expected} time=$(date -Is)" > "${status_path}"
    log_msg "DONE openie dataset=${dataset} output=${openie_json}"
  else
    echo "[FAILED] dataset=${dataset} stage=openie output_incomplete=${openie_json} time=$(date -Is)" > "${status_path}"
    log_msg "FAILED openie dataset=${dataset} output_incomplete=${openie_json}"
    return 1
  fi
}

export_pool() {
  local dataset="$1"
  local openie_json="${OUT_DIR}/openie/${dataset}_agsto_openie_qwen3_32b_no_think.json"
  local pool_json="${OUT_DIR}/pools/${dataset}_qwen32b_nvembed_sync_pool${POOL_K}_limit${LIMIT}.json"
  local status_path="${OUT_DIR}/status/${dataset}_export.status"
  local log_path="${OUT_DIR}/logs/${dataset}_export.log"

  if [[ -s "${pool_json}" ]]; then
    log_msg "SKIP export dataset=${dataset} existing=${pool_json}"
    echo "[SKIP] dataset=${dataset} stage=export existing=${pool_json} time=$(date -Is)" > "${status_path}"
    return 0
  fi

  log_msg "START export dataset=${dataset} limit=${LIMIT} openie=${openie_json}"
  echo "[START] dataset=${dataset} stage=export limit=${LIMIT} openie=${openie_json} time=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/export_agsto_pool.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --pool_k "${POOL_K}" \
      --openie_results "${openie_json}" \
      --retrieval_top_k 20 \
      --candidate_limit 120 \
      --proposal_candidate_depth 10 \
      --support_proposal_depth 6 \
      --beam_size 12 \
      --native_dense_anchor true \
      --native_dense_anchor_top_k 20 \
      --chunk_embedding_template "outputs_step0_general_nvembed_{dataset}/qwen3-8b_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --embedding_model "${EMBEDDING_API_MODEL}" \
      --embedding_batch_size 8 \
      --dense_query_instruction_mode raw \
      --semantic_residual_weight 12.0 \
      --output_json "${pool_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]] && [[ -s "${pool_json}" ]]; then
    echo "[DONE] dataset=${dataset} stage=export time=$(date -Is)" > "${status_path}"
    log_msg "DONE export dataset=${dataset} output=${pool_json}"
  else
    echo "[FAILED] dataset=${dataset} stage=export code=${code} time=$(date -Is)" > "${status_path}"
    log_msg "FAILED export dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

eval_daec() {
  local dataset="$1"
  local pool_json="${OUT_DIR}/pools/${dataset}_qwen32b_nvembed_sync_pool${POOL_K}_limit${LIMIT}.json"
  local output_json="${OUT_DIR}/evals/${dataset}_qwen32b_nvembed_sync_daec_llm_limit${LIMIT}.json"
  local cache_path="${OUT_DIR}/evals/${dataset}_qwen32b_nvembed_sync_daec_llm_limit${LIMIT}.binding_cache.json"
  local status_path="${OUT_DIR}/status/${dataset}_daec_eval.status"
  local log_path="${OUT_DIR}/logs/${dataset}_daec_eval.log"

  if [[ -s "${output_json}" ]]; then
    log_msg "SKIP daec eval dataset=${dataset} existing=${output_json}"
    echo "[SKIP] dataset=${dataset} stage=daec_eval existing=${output_json} time=$(date -Is)" > "${status_path}"
    return 0
  fi

  if [[ ! -s "${pool_json}" ]]; then
    log_msg "FAILED daec eval dataset=${dataset} missing_pool=${pool_json}"
    echo "[FAILED] dataset=${dataset} stage=daec_eval missing_pool=${pool_json} time=$(date -Is)" > "${status_path}"
    return 2
  fi

  log_msg "START daec eval dataset=${dataset} limit=${LIMIT} model=${LLM_MODEL} port=${LLM_PORT}"
  echo "[START] dataset=${dataset} stage=daec_eval limit=${LIMIT} model=${LLM_MODEL} port=${LLM_PORT} time=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --save_dir "${SAVE_DIR}" \
      --llm_name "${LLM_NAME}" \
      --llm_request_name "${LLM_MODEL}" \
      --max_retry_attempts 20 \
      --qwen_disable_thinking \
      --llm_base_url "http://localhost:${LLM_PORT}/v1" \
      --embedding_name "${EVAL_EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --external_pool_json "${pool_json}" \
      --external_pool_source_name "agsto_qwen32b_nvembed_sync_pool${POOL_K}" \
      --external_pool_strict_questions true \
      --setwise_selector daec_noisyor_llm \
      --setwise_pool_k "${POOL_K}" \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --dtc_decomposition_mode llm \
      --dtc_binding_max_candidates 5 \
      --llm_binding_url "http://localhost:${LLM_PORT}/v1" \
      --llm_binding_model "${LLM_MODEL}" \
      --llm_binding_cache_path "${cache_path}" \
      --llm_binding_title_match_mode "${MATCH_MODE}" \
      --causal_enabled false \
      --causal_engine_version v2 \
      --causal_v2_base_retrieval_mode dense \
      --structure_rerank_enabled false \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]] && [[ -s "${output_json}" ]]; then
    echo "[DONE] dataset=${dataset} stage=daec_eval time=$(date -Is)" > "${status_path}"
    log_msg "DONE daec eval dataset=${dataset} output=${output_json}"
  else
    echo "[FAILED] dataset=${dataset} stage=daec_eval code=${code} time=$(date -Is)" > "${status_path}"
    log_msg "FAILED daec eval dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_dataset() {
  local dataset="$1"
  build_openie "${dataset}" || return 1
  export_pool "${dataset}" || return 1
  eval_daec "${dataset}" || return 1
}

main() {
  echo "[START] agsto_qwen32b_nvembed_sync_limit${LIMIT} time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
  log_msg "AG-STO Qwen3-32B + NV-Embed-v2 sync limit${LIMIT} begin"
  log_msg "32B endpoint: port=${LLM_PORT} model=${LLM_MODEL}; qwen_disable_thinking=true"
  log_msg "datasets run sequentially through one 32B endpoint: 2wikimultihopqa -> hotpotqa -> musique"

  local status=0
  for dataset in 2wikimultihopqa hotpotqa musique; do
    if ! run_dataset "${dataset}"; then
      log_msg "FAILED dataset=${dataset}"
      status=1
      break
    fi
  done

  if [[ "${status}" -eq 0 ]]; then
    echo "[DONE] agsto_qwen32b_nvembed_sync_limit${LIMIT} time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
    log_msg "AG-STO Qwen3-32B + NV-Embed-v2 sync limit${LIMIT} complete"
  else
    echo "[FAILED] agsto_qwen32b_nvembed_sync_limit${LIMIT} status=${status} time=$(date -Is)" > "${OUT_DIR}/status/launcher.status"
    log_msg "AG-STO Qwen3-32B + NV-Embed-v2 sync limit${LIMIT} finished with failures"
  fi
  return "${status}"
}

main "$@"
