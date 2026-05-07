#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
RUN_DIR="${ROOT_DIR}/run_logs/counterfactual_fill_mechanism_20260507"
REPORT_DIR="${ROOT_DIR}/reports/counterfactual_fill_mechanism_20260507"
SAVE_DIR="outputs_step0_general_nvembed"
LLM_MODEL="qwen3-8b-train"
EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"
EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"

mkdir -p "${RUN_DIR}" "${REPORT_DIR}"

run_cmd() {
  local label="$1"
  shift
  local log_path="${RUN_DIR}/${label}.log"
  local status_path="${RUN_DIR}/${label}.status"
  if [[ -f "${status_path}" ]] && grep -q "^DONE " "${status_path}"; then
    echo "[SKIP] ${label}"
    return 0
  fi
  echo "START ${label} $(date -Is)" > "${status_path}"
  echo "[START] ${label}"
  (cd "${ROOT_DIR}" && "$@") > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    echo "DONE ${label} $(date -Is)" > "${status_path}"
    echo "[DONE] ${label}"
  else
    echo "FAILED ${label} code=${code} $(date -Is)" > "${status_path}"
    echo "[FAILED] ${label} code=${code}"
    return "${code}"
  fi
}

run_eval() {
  local dataset="$1"
  local port="$2"
  local variant="$3"
  local pool_json="$4"
  local output_json="${REPORT_DIR}/${dataset}_${variant}.eval.json"
  run_cmd "eval_${dataset}_${variant}" \
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit 0 \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name "${LLM_MODEL}" \
      --max_retry_attempts 20 \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name "${EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --external_pool_json "${pool_json}" \
      --external_pool_source_name "counterfactual_${variant}" \
      --external_pool_strict_questions true \
      --setwise_selector none \
      --output_json "${output_json}"
}

main() {
  echo "[START] counterfactual_fill_mechanism $(date -Is)" > "${RUN_DIR}/launcher.status"
  local rc=0
  run_eval "2wikimultihopqa_dbec_cf_underselect_20260507" "8041" "rank_fill5" "run_logs/counterfactual_fill_mechanism_20260507/2wikimultihopqa_dbec_cf_underselect_20260507_rank_fill5.pool.json" || rc=1
  run_eval "2wikimultihopqa_dbec_cf_underselect_20260507" "8041" "oracle_fill5" "run_logs/counterfactual_fill_mechanism_20260507/2wikimultihopqa_dbec_cf_underselect_20260507_oracle_fill5.pool.json" || rc=1
  run_eval "2wikimultihopqa_dbec_cf_underselect_20260507" "8041" "dbec_repair_fill5" "run_logs/counterfactual_fill_mechanism_20260507/2wikimultihopqa_dbec_cf_underselect_20260507_dbec_repair_fill5.pool.json" || rc=1
  run_eval "musique_dbec_cf_underselect_20260507" "8043" "rank_fill5" "run_logs/counterfactual_fill_mechanism_20260507/musique_dbec_cf_underselect_20260507_rank_fill5.pool.json" || rc=1
  run_eval "musique_dbec_cf_underselect_20260507" "8043" "oracle_fill5" "run_logs/counterfactual_fill_mechanism_20260507/musique_dbec_cf_underselect_20260507_oracle_fill5.pool.json" || rc=1
  run_eval "musique_dbec_cf_underselect_20260507" "8043" "dbec_repair_fill5" "run_logs/counterfactual_fill_mechanism_20260507/musique_dbec_cf_underselect_20260507_dbec_repair_fill5.pool.json" || rc=1
  if [[ "${rc}" -eq 0 ]]; then
    echo "[DONE] counterfactual_fill_mechanism $(date -Is)" > "${RUN_DIR}/launcher.status"
  else
    echo "[FAILED] counterfactual_fill_mechanism rc=${rc} $(date -Is)" > "${RUN_DIR}/launcher.status"
  fi
  return "${rc}"
}

main "$@"
