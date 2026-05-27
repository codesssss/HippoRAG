#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"

RUN_TAG="${RUN_TAG:-evidencelink_doc_transition_controls_retrieval_full1000_20260520}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/run_logs/${RUN_TAG}}"
REFERENCE_POOL_ROOT="${REFERENCE_POOL_ROOT:-${ROOT_DIR}/run_logs/etv4_pcec_qwen32b_gpt4omini_none_full1000_20260514/pools}"

DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique}"
POLICIES="${POLICIES:-dense_doc_knn degree_matched_shuffle}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
POOL_K="${POOL_K:-100}"
SETWISE_POOL_K="${SETWISE_POOL_K:-100}"
READER_BUDGET_K="${READER_BUDGET_K:-5}"
PREFIX_BUDGET_M="${PREFIX_BUDGET_M:-4}"
QA_TOP_K="${QA_TOP_K:-5}"

DENSE_SEED_K="${DENSE_SEED_K:-20}"
PREFIX_K="${PREFIX_K:-5}"
NEIGHBOR_K="${NEIGHBOR_K:-20}"
CLOSURE_HOPS="${CLOSURE_HOPS:-2}"
SHUFFLE_SEED="${SHUFFLE_SEED:-20260520}"
KNN_BLOCK_SIZE="${KNN_BLOCK_SIZE:-256}"

GRAPH_LLM_NAME="${GRAPH_LLM_NAME:-qwen3-32b-judge}"
QWEN_BASE_URLS="${QWEN_BASE_URLS:-http://localhost:8046/v1 http://localhost:8045/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
EXPORT_EMBEDDING_BATCH_SIZE="${EXPORT_EMBEDDING_BATCH_SIZE:-32}"

mkdir -p \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/status" \
  "${OUT_ROOT}/pools" \
  "${OUT_ROOT}/dbec_assets" \
  "${OUT_ROOT}/pcec"

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
  if curl --max-time 15 -fsS "${url}" >/dev/null 2>&1; then
    log_msg "OK endpoint ${name}: ${url}"
    return 0
  fi
  log_msg "FAILED endpoint ${name}: ${url}"
  return 1
}

pool_json_path() {
  local policy="$1"
  local dataset="$2"
  printf '%s/pools/%s/%s_%s_pool%s_limit%s.json' \
    "${OUT_ROOT}" "${policy}" "${dataset}" "${policy}" "${POOL_K}" "${MAX_QUERIES}"
}

dbec_report_path() {
  local policy="$1"
  local dataset="$2"
  printf '%s/dbec_assets/%s/evals/%s_%s_pool%s_dbec_qwen32b_stable_limit%s.json' \
    "${OUT_ROOT}" "${policy}" "${dataset}" "${policy}" "${POOL_K}" "${MAX_QUERIES}"
}

dbec_binding_cache_path() {
  local policy="$1"
  local dataset="$2"
  printf '%s/dbec_assets/%s/evals/%s_%s_pool%s_dbec_qwen32b.binding_cache.json' \
    "${OUT_ROOT}" "${policy}" "${dataset}" "${policy}" "${POOL_K}"
}

pcec_report_path() {
  local policy="$1"
  local dataset="$2"
  printf '%s/pcec/%s/evals/%s_%s_pcec_native_pool_prefix%s_residual%s_pool%s_limit%s.json' \
    "${OUT_ROOT}" "${policy}" "${dataset}" "${policy}" "${PREFIX_BUDGET_M}" \
    "$((READER_BUDGET_K - PREFIX_BUDGET_M))" "${POOL_K}" "${MAX_QUERIES}"
}

run_export_pool_dataset() {
  local policy="$1"
  local dataset="$2"
  local output_json log_path
  output_json="$(pool_json_path "${policy}" "${dataset}")"
  log_path="${OUT_ROOT}/logs/01_export_${policy}_${dataset}.log"
  mkdir -p "$(dirname "${output_json}")"

  if [[ -s "${output_json}" ]]; then
    write_status "01_export_${policy}_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP export policy=${policy} dataset=${dataset}"
    return 0
  fi

  write_status "01_export_${policy}_${dataset}" "START output=${output_json}"
  log_msg "START export policy=${policy} dataset=${dataset}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    local reference_args=()
    if [[ "${policy}" == "edge_count_matched_dense_doc_knn" || "${policy}" == "same_seed_edge_count_matched_dense_doc_knn" ]]; then
      reference_args=(
        --reference-pool-json
        "${REFERENCE_POOL_ROOT}/${dataset}_etv4_pool100_limit1000.json"
      )
    fi
    "${PYTHON_BIN}" scripts/export_doc_transition_control_pool.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --pool-k "${POOL_K}" \
      --dense-seed-k "${DENSE_SEED_K}" \
      --prefix-k "${PREFIX_K}" \
      --neighbor-k "${NEIGHBOR_K}" \
      --closure-hops "${CLOSURE_HOPS}" \
      --policy "${policy}" \
      --shuffle-seed "${SHUFFLE_SEED}" \
      --knn-block-size "${KNN_BLOCK_SIZE}" \
      --save-dir outputs_step0_general_nvembed \
      --llm-name "${GRAPH_LLM_NAME}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EXPORT_EMBEDDING_BATCH_SIZE}" \
      "${reference_args[@]}" \
      --output-json "${output_json}"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "01_export_${policy}_${dataset}" "DONE output=${output_json}"
    log_msg "DONE export policy=${policy} dataset=${dataset}"
  else
    write_status "01_export_${policy}_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED export policy=${policy} dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_dbec_asset_dataset() {
  local policy="$1"
  local dataset="$2"
  local qwen_url="$3"
  local pool_json output_json binding_json log_path
  pool_json="$(pool_json_path "${policy}" "${dataset}")"
  output_json="$(dbec_report_path "${policy}" "${dataset}")"
  binding_json="$(dbec_binding_cache_path "${policy}" "${dataset}")"
  log_path="${OUT_ROOT}/logs/02_dbec32_${policy}_${dataset}.log"
  mkdir -p "$(dirname "${output_json}")"

  if [[ ! -s "${pool_json}" ]]; then
    write_status "02_dbec32_${policy}_${dataset}" "FAILED missing_pool=${pool_json}"
    log_msg "FAILED DBEC32 policy=${policy} dataset=${dataset} missing pool=${pool_json}"
    return 2
  fi
  if [[ -s "${output_json}" && -s "${binding_json}" ]]; then
    write_status "02_dbec32_${policy}_${dataset}" "SKIP existing report=${output_json} binding=${binding_json}"
    log_msg "SKIP DBEC32 policy=${policy} dataset=${dataset}"
    return 0
  fi

  write_status "02_dbec32_${policy}_${dataset}" "START pool=${pool_json} qwen_url=${qwen_url}"
  log_msg "START DBEC32 policy=${policy} dataset=${dataset} qwen_url=${qwen_url}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    export HIPPORAG_RERANK_FORCE_NO_THINK=1
    "${PYTHON_BIN}" evidence_transition_graphragv3_dbec_latest/run_eval.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --pool-k "${POOL_K}" \
      --setwise-pool-k "${SETWISE_POOL_K}" \
      --qa-top-k "${QA_TOP_K}" \
      --pool-json "${pool_json}" \
      --output-root "${OUT_ROOT}/dbec_assets/${policy}" \
      --output-json "${output_json}" \
      --binding-cache-path "${binding_json}" \
      --save-dir outputs_step0_general_nvembed \
      --llm-name "${GRAPH_LLM_NAME}" \
      --llm-request-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${qwen_url}" \
      --llm-binding-url "${qwen_url}" \
      --llm-binding-model "${GRAPH_LLM_NAME}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --qwen-disable-thinking
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" && -s "${binding_json}" ]]; then
    write_status "02_dbec32_${policy}_${dataset}" "DONE report=${output_json} binding=${binding_json}"
    log_msg "DONE DBEC32 policy=${policy} dataset=${dataset}"
  else
    write_status "02_dbec32_${policy}_${dataset}" "FAILED code=${code} report=${output_json} binding=${binding_json} log=${log_path}"
    log_msg "FAILED DBEC32 policy=${policy} dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_pcec_dataset() {
  local policy="$1"
  local dataset="$2"
  local qwen_url="$3"
  local pool_json requirement_json binding_json output_json log_path
  pool_json="$(pool_json_path "${policy}" "${dataset}")"
  requirement_json="$(dbec_report_path "${policy}" "${dataset}")"
  binding_json="$(dbec_binding_cache_path "${policy}" "${dataset}")"
  output_json="$(pcec_report_path "${policy}" "${dataset}")"
  log_path="${OUT_ROOT}/logs/03_pcec32_${policy}_${dataset}.log"
  mkdir -p "$(dirname "${output_json}")" "${OUT_ROOT}/pcec/${policy}/runtime/${dataset}"

  if [[ ! -s "${pool_json}" || ! -s "${requirement_json}" || ! -s "${binding_json}" ]]; then
    write_status "03_pcec32_${policy}_${dataset}" "FAILED missing_input pool=${pool_json} report=${requirement_json} binding=${binding_json}"
    log_msg "FAILED PCEC policy=${policy} dataset=${dataset} missing input"
    return 2
  fi
  if [[ -s "${output_json}" ]]; then
    write_status "03_pcec32_${policy}_${dataset}" "SKIP existing output=${output_json}"
    log_msg "SKIP PCEC policy=${policy} dataset=${dataset}"
    return 0
  fi

  write_status "03_pcec32_${policy}_${dataset}" "START pool=${pool_json} qwen_url=${qwen_url}"
  log_msg "START PCEC policy=${policy} dataset=${dataset} qwen_url=${qwen_url}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    export HIPPORAG_RERANK_FORCE_NO_THINK=1
    "${PYTHON_BIN}" evidenceflow/run_native_pool.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --max-queries "${MAX_QUERIES}" \
      --pool-k "${POOL_K}" \
      --reader-budget-k "${READER_BUDGET_K}" \
      --prefix-budget-m "${PREFIX_BUDGET_M}" \
      --pool-json "${pool_json}" \
      --requirement-report "${requirement_json}" \
      --binding-cache-path "${binding_json}" \
      --output-json "${output_json}" \
      --output-root "${OUT_ROOT}/pcec/${policy}" \
      --data-root reproduce/dataset \
      --save-dir outputs_step0_general_nvembed \
      --llm-name "${GRAPH_LLM_NAME}" \
      --llm-request-name "${GRAPH_LLM_NAME}" \
      --llm-base-url "${qwen_url}" \
      --llm-binding-model "${GRAPH_LLM_NAME}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --qwen-disable-thinking
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${output_json}" ]]; then
    write_status "03_pcec32_${policy}_${dataset}" "DONE output=${output_json}"
    log_msg "DONE PCEC policy=${policy} dataset=${dataset}"
  else
    write_status "03_pcec32_${policy}_${dataset}" "FAILED code=${code} output=${output_json} log=${log_path}"
    log_msg "FAILED PCEC policy=${policy} dataset=${dataset} code=${code}"
  fi
  return "${code}"
}

run_job() {
  local policy="$1"
  local dataset="$2"
  local qwen_url="$3"
  run_export_pool_dataset "${policy}" "${dataset}" || return 1
  run_dbec_asset_dataset "${policy}" "${dataset}" "${qwen_url}" || return 1
  run_pcec_dataset "${policy}" "${dataset}" "${qwen_url}" || return 1
}

worker_loop() {
  local slot_idx="$1"
  local slot_count="$2"
  local qwen_url="$3"
  local job_id=0
  local policy dataset

  write_status "worker_${slot_idx}" "START qwen_url=${qwen_url}"
  log_msg "START worker slot=${slot_idx}/${slot_count} qwen_url=${qwen_url}"
  for policy in ${POLICIES}; do
    for dataset in ${DATASETS}; do
      if [[ $((job_id % slot_count)) -eq "${slot_idx}" ]]; then
        write_status "worker_${slot_idx}_${policy}_${dataset}" "START job_id=${job_id} qwen_url=${qwen_url}"
        run_job "${policy}" "${dataset}" "${qwen_url}" || {
          write_status "worker_${slot_idx}_${policy}_${dataset}" "FAILED job_id=${job_id}"
          write_status "worker_${slot_idx}" "FAILED at policy=${policy} dataset=${dataset}"
          return 1
        }
        write_status "worker_${slot_idx}_${policy}_${dataset}" "DONE job_id=${job_id}"
      fi
      job_id=$((job_id + 1))
    done
  done
  write_status "worker_${slot_idx}" "DONE qwen_url=${qwen_url}"
  log_msg "DONE worker slot=${slot_idx}/${slot_count} qwen_url=${qwen_url}"
  return 0
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG} datasets=${DATASETS} policies=${POLICIES}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: doc-transition controls, full1000 main3, Qwen32B no-think DBEC/PCEC, retrieval-only, no GPT-4o-mini reader"
  log_msg "Control policies: ${POLICIES}"
  log_msg "Graph utility model: ${GRAPH_LLM_NAME}; Qwen URLs: ${QWEN_BASE_URLS}"
  log_msg "Pool/readout: pool_k=${POOL_K} prefix_budget_m=${PREFIX_BUDGET_M} reader_budget_k=${READER_BUDGET_K}; dense_seed_k=${DENSE_SEED_K} neighbor_k=${NEIGHBOR_K} closure_hops=${CLOSURE_HOPS}"

  local preflight=0
  local urls=()
  local url
  for url in ${QWEN_BASE_URLS}; do
    urls+=("${url}")
    check_endpoint "qwen32b:${url}" "${url}/models" || preflight=1
  done
  check_endpoint "nv_embed" "http://localhost:8019/v1/models" || preflight=1
  if [[ "${#urls[@]}" -eq 0 ]]; then
    log_msg "FAILED no Qwen URLs configured"
    preflight=1
  fi
  if [[ "${preflight}" -ne 0 ]]; then
    write_status "launcher" "FAILED preflight"
    return 1
  fi

  local slot_count="${#urls[@]}"
  local status=0
  local pids=()
  local slot_idx pid
  for slot_idx in "${!urls[@]}"; do
    worker_loop "${slot_idx}" "${slot_count}" "${urls[slot_idx]}" &
    pid="$!"
    pids+=("${pid}")
    log_msg "LAUNCHED worker slot=${slot_idx} pid=${pid} qwen_url=${urls[slot_idx]}"
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done

  if [[ "${status}" -eq 0 ]]; then
    write_status "launcher" "DONE run_tag=${RUN_TAG}"
    log_msg "DONE ${RUN_TAG}"
  else
    write_status "launcher" "FAILED run_tag=${RUN_TAG}"
    log_msg "FAILED ${RUN_TAG}"
  fi
  return "${status}"
}

main "$@"
