#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
RUN_DIR="${ROOT_DIR}/run_logs/repair_gated_arbitration_20260507"
REPORT_DIR="${ROOT_DIR}/reports/repair_gated_arbitration_reader_20260507"
SAVE_DIR="outputs_step0_general_nvembed"
LLM_MODEL="${REPAIR_GATED_LLM_MODEL:-qwen3-8b-train}"
EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"
EMBEDDING_BASE_URL="${REPAIR_GATED_EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
PORTS_CSV="${REPAIR_GATED_QWEN_PORTS:-8041,8042,8043}"
MAX_PARALLEL="${REPAIR_GATED_MAX_PARALLEL:-3}"
IFS="," read -r -a PORTS <<< "${PORTS_CSV}"

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
  local variant="$2"
  local pool_json="$3"
  local port="$4"
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
      --external_pool_source_name "repair_gated_${variant}" \
      --external_pool_strict_questions true \
      --setwise_selector none \
      --output_json "${output_json}"
}

throttle() {
  while [[ "$(jobs -pr | wc -l)" -ge "${MAX_PARALLEL}" ]]; do
    wait -n || RC=1
  done
}

main() {
  echo "[START] repair_gated_reader_eval $(date -Is)" > "${RUN_DIR}/launcher.status"
  RC=0
  local job_index=0
  throttle
  local port="${PORTS[$((job_index % ${#PORTS[@]}))]}"
  (
    group_rc=0
    run_eval "2wikimultihopqa_repair_gated_underselect_20260507" "rq1_mr1_g010_branchstrict" "run_logs/repair_gated_arbitration_20260507/2wikimultihopqa_repair_gated_underselect_20260507_rq1_mr1_g010_branchstrict.pool.json" "${port}" || group_rc=1
    run_eval "2wikimultihopqa_repair_gated_underselect_20260507" "rq1_mr2_g010_branchstrict" "run_logs/repair_gated_arbitration_20260507/2wikimultihopqa_repair_gated_underselect_20260507_rq1_mr2_g010_branchstrict.pool.json" "${port}" || group_rc=1
    run_eval "2wikimultihopqa_repair_gated_underselect_20260507" "rq1_mr3_g010_branchstrict" "run_logs/repair_gated_arbitration_20260507/2wikimultihopqa_repair_gated_underselect_20260507_rq1_mr3_g010_branchstrict.pool.json" "${port}" || group_rc=1
    exit "${group_rc}"
  ) &
  job_index=$((job_index + 1))
  throttle
  local port="${PORTS[$((job_index % ${#PORTS[@]}))]}"
  (
    group_rc=0
    run_eval "hotpotqa_repair_gated_underselect_20260507" "rq1_mr1_g010_branchstrict" "run_logs/repair_gated_arbitration_20260507/hotpotqa_repair_gated_underselect_20260507_rq1_mr1_g010_branchstrict.pool.json" "${port}" || group_rc=1
    run_eval "hotpotqa_repair_gated_underselect_20260507" "rq1_mr2_g010_branchstrict" "run_logs/repair_gated_arbitration_20260507/hotpotqa_repair_gated_underselect_20260507_rq1_mr2_g010_branchstrict.pool.json" "${port}" || group_rc=1
    run_eval "hotpotqa_repair_gated_underselect_20260507" "rq1_mr3_g010_branchstrict" "run_logs/repair_gated_arbitration_20260507/hotpotqa_repair_gated_underselect_20260507_rq1_mr3_g010_branchstrict.pool.json" "${port}" || group_rc=1
    exit "${group_rc}"
  ) &
  job_index=$((job_index + 1))
  throttle
  local port="${PORTS[$((job_index % ${#PORTS[@]}))]}"
  (
    group_rc=0
    run_eval "musique_repair_gated_underselect_20260507" "rq1_mr1_g010_branchstrict" "run_logs/repair_gated_arbitration_20260507/musique_repair_gated_underselect_20260507_rq1_mr1_g010_branchstrict.pool.json" "${port}" || group_rc=1
    run_eval "musique_repair_gated_underselect_20260507" "rq1_mr2_g010_branchstrict" "run_logs/repair_gated_arbitration_20260507/musique_repair_gated_underselect_20260507_rq1_mr2_g010_branchstrict.pool.json" "${port}" || group_rc=1
    run_eval "musique_repair_gated_underselect_20260507" "rq1_mr3_g010_branchstrict" "run_logs/repair_gated_arbitration_20260507/musique_repair_gated_underselect_20260507_rq1_mr3_g010_branchstrict.pool.json" "${port}" || group_rc=1
    exit "${group_rc}"
  ) &
  job_index=$((job_index + 1))
  while [[ "$(jobs -pr | wc -l)" -gt 0 ]]; do
    wait -n || RC=1
  done
  if [[ "${RC}" -eq 0 ]]; then
    echo "[DONE] repair_gated_reader_eval $(date -Is)" > "${RUN_DIR}/launcher.status"
  else
    echo "[FAILED] repair_gated_reader_eval rc=${RC} $(date -Is)" > "${RUN_DIR}/launcher.status"
  fi
  return "${RC}"
}

main "$@"
