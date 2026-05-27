#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/nvme/code/HippoRAG}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/run_logs/launch_nq_popqa_32b_no_think_gpt4omini_full1000_20260516.sh}"

RUN_TAG="${RUN_TAG:-evidenceflow_nq_popqa_limit100_32b_no_think_gpt4omini_20260518}"
DATASETS="${DATASETS:-nq popqa}"
MAX_QUERIES="${MAX_QUERIES:-100}"
RUN_NEOCOR=0
PARALLEL_DATASETS="${PARALLEL_DATASETS:-0}"

# Reuse the already audited NQ/PopQA EvidenceFlow functions without running the
# full launcher, which also exports baselines and NeocorRAG.
# shellcheck disable=SC1090
source <(sed '$d' "${BASE_LAUNCHER}")

evidence_reader_report_path() {
  local display_dataset="$1"
  printf '%s/reader_qa/evidenceflow/%s_evidenceflow_qwen32b_no_think_gpt4omini_reader_top%s_limit%s.json' \
    "${OUT_ROOT}" "${display_dataset}" "${QA_TOP_K}" "${MAX_QUERIES}"
}

evidence_reader_report_md_path() {
  local display_dataset="$1"
  printf '%s/reader_qa/evidenceflow/%s_evidenceflow_qwen32b_no_think_gpt4omini_reader_top%s_limit%s.md' \
    "${OUT_ROOT}" "${display_dataset}" "${QA_TOP_K}" "${MAX_QUERIES}"
}

run_evidenceflow_lane() {
  local requested_dataset="$1"
  local llm_base_url evidence_dataset display_dataset
  llm_base_url="$(qwen_url_for_dataset "${requested_dataset}")"
  evidence_dataset="$(evidence_dataset_name "${requested_dataset}")"
  display_dataset="$(display_dataset_name "${evidence_dataset}")"

  write_status "lane_${display_dataset}" "START endpoint=${llm_base_url}"
  log_msg "START EvidenceFlow lane requested=${requested_dataset} evidence_dataset=${evidence_dataset} endpoint=${llm_base_url}"

  run_etv4_fresh_dataset "${evidence_dataset}" "${llm_base_url}" || return 1
  export_evidence_pool_dataset "${evidence_dataset}" || return 1
  run_dbec_dataset "${evidence_dataset}" "${llm_base_url}" || return 1
  run_pcec_dataset "${evidence_dataset}" "${llm_base_url}" || return 1
  run_evidence_reader_dataset "${evidence_dataset}" || return 1

  write_status "lane_${display_dataset}" "DONE endpoint=${llm_base_url}"
  log_msg "DONE EvidenceFlow lane requested=${requested_dataset} evidence_dataset=${evidence_dataset}"
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: EvidenceFlow only; limit=${MAX_QUERIES}; Qwen3-32B no_think non-reader; GPT-4o-mini reader; datasets=${DATASETS}; parallel=${PARALLEL_DATASETS}"

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

  local dataset pid rc=0
  if [[ "${PARALLEL_DATASETS}" == "1" ]]; then
    local pids=()
    for dataset in ${DATASETS}; do
      run_evidenceflow_lane "${dataset}" > "${OUT_ROOT}/logs/lane_${dataset}.log" 2>&1 &
      pids+=("$!")
      log_msg "LAUNCHED EvidenceFlow lane dataset=${dataset} pid=$!"
    done
    for pid in "${pids[@]}"; do
      wait "${pid}" || rc=1
    done
  else
    for dataset in ${DATASETS}; do
      run_evidenceflow_lane "${dataset}" > "${OUT_ROOT}/logs/lane_${dataset}.log" 2>&1 || rc=1
      if [[ "${rc}" -ne 0 ]]; then
        break
      fi
    done
  fi

  if [[ "${rc}" -ne 0 ]]; then
    write_status "launcher" "FAILED evidenceflow rc=${rc}"
    log_msg "FAILED EvidenceFlow lanes rc=${rc}"
    return 1
  fi

  write_summary
  write_status "launcher" "DONE run_tag=${RUN_TAG}"
  log_msg "DONE ${RUN_TAG}"
}

main "$@"
