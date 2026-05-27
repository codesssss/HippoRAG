#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
OUT_DIR="${ROOT_DIR}/run_logs/agsto_owned_openie_full_20260505"
SAVE_DIR="outputs_step0_general_nvembed"
EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"
LLM_MODEL="qwen3-8b-train"
LIMIT="${LIMIT:-0}"
POOL_K="${POOL_K:-100}"
OPENIE_WORKERS="${OPENIE_WORKERS:-16}"
OPENIE_MAX_TOKENS="${OPENIE_MAX_TOKENS:-768}"
MATCH_MODE="${MATCH_MODE:-wiki_title}"

mkdir -p "${OUT_DIR}/openie" "${OUT_DIR}/pools" "${OUT_DIR}/evals" "${OUT_DIR}/logs" "${OUT_DIR}/status"

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_DIR}/logs/launcher.log"
}

corpus_json() {
  echo "${ROOT_DIR}/reproduce/dataset/${1}_corpus.json"
}

dataset_port() {
  case "$1" in
    musique) echo "8041" ;;
    hotpotqa) echo "8042" ;;
    2wikimultihopqa) echo "8043" ;;
    *) return 2 ;;
  esac
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

build_owned_openie() {
  local dataset="$1"
  local port="$2"
  local openie_json="${OUT_DIR}/openie/${dataset}_agsto_owned_openie_qwen3_8b_no_think.json"
  local status_path="${OUT_DIR}/status/${dataset}_openie.status"
  local log_path="${OUT_DIR}/logs/${dataset}_openie.log"
  local expected
  expected="$(expected_doc_count "${dataset}")"

  if openie_ready "${openie_json}" "${expected}"; then
    log_msg "SKIP openie dataset=${dataset} existing=${openie_json}"
    echo "[SKIP] dataset=${dataset} stage=openie existing=${openie_json} docs=${expected} time=$(date -Is)" > "${status_path}"
    return 0
  fi

  log_msg "START openie dataset=${dataset} docs=${expected} workers=${OPENIE_WORKERS} port=${port}"
  echo "[START] dataset=${dataset} stage=openie docs=${expected} workers=${OPENIE_WORKERS} port=${port} time=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" src/agsto_v12/build_openie.py \
      --corpus-json "$(corpus_json "${dataset}")" \
      --output-openie-json "${openie_json}" \
      --mode llm \
      --workers "${OPENIE_WORKERS}" \
      --checkpoint-every 100 \
      --resume \
      --continue-on-error \
      --llm-base-url "http://localhost:${port}/v1" \
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
    return 1
  fi
}

export_pool() {
  local dataset="$1"
  local openie_json="${OUT_DIR}/openie/${dataset}_agsto_owned_openie_qwen3_8b_no_think.json"
  local pool_json="${OUT_DIR}/pools/${dataset}_agsto_owned_openie_pool${POOL_K}_full.json"
  local status_path="${OUT_DIR}/status/${dataset}_export.status"
  local log_path="${OUT_DIR}/logs/${dataset}_export.log"

  if [[ -s "${pool_json}" ]]; then
    log_msg "SKIP export dataset=${dataset} existing=${pool_json}"
    echo "[SKIP] dataset=${dataset} stage=export existing=${pool_json} time=$(date -Is)" > "${status_path}"
    return 0
  fi

  log_msg "START export dataset=${dataset} openie=${openie_json}"
  echo "[START] dataset=${dataset} stage=export openie=${openie_json} time=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" scripts/export_agsto_pool.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --pool_k "${POOL_K}" \
      --openie_results "${openie_json}" \
      --candidate_limit 180 \
      --proposal_candidate_depth 24 \
      --support_proposal_depth 12 \
      --beam_size 12 \
      --output_json "${pool_json}"
  ) > "${log_path}" 2>&1
  echo "[DONE] dataset=${dataset} stage=export time=$(date -Is)" > "${status_path}"
  log_msg "DONE export dataset=${dataset} output=${pool_json}"
}

eval_agsto_base() {
  local dataset="$1"
  local port="$2"
  local pool_json="${OUT_DIR}/pools/${dataset}_agsto_owned_openie_pool${POOL_K}_full.json"
  local output_json="${OUT_DIR}/evals/${dataset}_agsto_owned_openie_pool${POOL_K}_base_full.json"
  local status_path="${OUT_DIR}/status/${dataset}_base_eval.status"
  local log_path="${OUT_DIR}/logs/${dataset}_base_eval.log"

  if [[ -s "${output_json}" ]]; then
    log_msg "SKIP base eval dataset=${dataset} existing=${output_json}"
    echo "[SKIP] dataset=${dataset} stage=base_eval existing=${output_json} time=$(date -Is)" > "${status_path}"
    return 0
  fi

  log_msg "START base eval dataset=${dataset} port=${port}"
  echo "[START] dataset=${dataset} stage=base_eval port=${port} time=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name "${LLM_MODEL}" \
      --max_retry_attempts 20 \
      --qwen_disable_thinking \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name "${EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --external_pool_json "${pool_json}" \
      --external_pool_source_name "agsto_owned_openie_pool${POOL_K}" \
      --external_pool_strict_questions true \
      --setwise_selector none \
      --setwise_pool_k "${POOL_K}" \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --causal_enabled false \
      --causal_engine_version v2 \
      --causal_v2_base_retrieval_mode dense \
      --structure_rerank_enabled false \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1
  echo "[DONE] dataset=${dataset} stage=base_eval time=$(date -Is)" > "${status_path}"
  log_msg "DONE base eval dataset=${dataset} output=${output_json}"
}

eval_agsto_daec_llm() {
  local dataset="$1"
  local port="$2"
  local pool_json="${OUT_DIR}/pools/${dataset}_agsto_owned_openie_pool${POOL_K}_full.json"
  local output_json="${OUT_DIR}/evals/${dataset}_agsto_owned_openie_pool${POOL_K}_daec_llm_full.json"
  local cache_path="${OUT_DIR}/evals/${dataset}_agsto_owned_openie_pool${POOL_K}_daec_llm_full.binding_cache.json"
  local status_path="${OUT_DIR}/status/${dataset}_daec_eval.status"
  local log_path="${OUT_DIR}/logs/${dataset}_daec_eval.log"

  if [[ -s "${output_json}" ]]; then
    log_msg "SKIP daec eval dataset=${dataset} existing=${output_json}"
    echo "[SKIP] dataset=${dataset} stage=daec_eval existing=${output_json} time=$(date -Is)" > "${status_path}"
    return 0
  fi

  log_msg "START daec eval dataset=${dataset} port=${port}"
  echo "[START] dataset=${dataset} stage=daec_eval port=${port} time=$(date -Is)" > "${status_path}"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit "${LIMIT}" \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name "${LLM_MODEL}" \
      --max_retry_attempts 20 \
      --qwen_disable_thinking \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name "${EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --external_pool_json "${pool_json}" \
      --external_pool_source_name "agsto_owned_openie_pool${POOL_K}" \
      --external_pool_strict_questions true \
      --setwise_selector daec_noisyor_llm \
      --setwise_pool_k "${POOL_K}" \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --dtc_decomposition_mode llm \
      --dtc_binding_max_candidates 5 \
      --llm_binding_url "http://localhost:${port}/v1" \
      --llm_binding_model "${LLM_MODEL}" \
      --llm_binding_cache_path "${cache_path}" \
      --llm_binding_title_match_mode "${MATCH_MODE}" \
      --causal_enabled false \
      --causal_engine_version v2 \
      --causal_v2_base_retrieval_mode dense \
      --structure_rerank_enabled false \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1
  echo "[DONE] dataset=${dataset} stage=daec_eval time=$(date -Is)" > "${status_path}"
  log_msg "DONE daec eval dataset=${dataset} output=${output_json}"
}

run_dataset() {
  local dataset="$1"
  local port
  port="$(dataset_port "${dataset}")"
  build_owned_openie "${dataset}" "${port}"
  export_pool "${dataset}"
  eval_agsto_base "${dataset}" "${port}"
  eval_agsto_daec_llm "${dataset}" "${port}"
}

log_msg "AG-STO owned-OpenIE full comparison begin limit=${LIMIT} pool_k=${POOL_K} openie_workers=${OPENIE_WORKERS}"
log_msg "lanes: musique->8041 hotpotqa->8042 2wikimultihopqa->8043"

(run_dataset musique) &
pid_musique=$!
(run_dataset hotpotqa) &
pid_hotpotqa=$!
(run_dataset 2wikimultihopqa) &
pid_2wiki=$!

status=0
if ! wait "${pid_musique}"; then
  log_msg "FAILED dataset=musique"
  status=1
fi
if ! wait "${pid_hotpotqa}"; then
  log_msg "FAILED dataset=hotpotqa"
  status=1
fi
if ! wait "${pid_2wiki}"; then
  log_msg "FAILED dataset=2wikimultihopqa"
  status=1
fi

if [[ "${status}" -eq 0 ]]; then
  log_msg "AG-STO owned-OpenIE full comparison complete"
else
  log_msg "AG-STO owned-OpenIE full comparison finished with failures"
fi
exit "${status}"
