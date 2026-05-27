#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
BASE_INDEX="${ROOT_DIR}/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/index"
RUN_ROOT="${ROOT_DIR}/run_logs/etv4_pool300_musique_limit100_20260514"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status"

run_variant() {
  local tag="$1"
  local candidate_mode="$2"
  local preserved_dense_count="$3"
  local out="${RUN_ROOT}/${tag}"
  local ds_dir="${out}/musique"
  local log="${RUN_ROOT}/logs/${tag}.log"
  local status="${RUN_ROOT}/status/${tag}.status"

  mkdir -p "${ds_dir}"
  ln -sfn "${BASE_INDEX}" "${ds_dir}/index"
  printf 'running\t%s\n' "$(date -Is)" > "${status}"
  printf '[%s] START %s mode=%s preserved=%s\n' "$(date -Is)" "${tag}" "${candidate_mode}" "${preserved_dense_count}" | tee -a "${log}"

  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONUNBUFFERED=1 PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_fresh_e2e.py \
      --datasets musique \
      --output-root "${out}" \
      --reuse-current-fresh-index \
      --max-queries 100 \
      --llm-name qwen3-32b-judge \
      --llm-base-url http://localhost:8045/v1 \
      --max-new-tokens 2048 \
      --max-retry-attempts 20 \
      --qwen-disable-thinking \
      --embedding-name VLLM/nvidia/NV-Embed-v2 \
      --embedding-base-url http://localhost:8019/v1/embeddings \
      --embedding-batch-size 32 \
      --candidate-generator-mode "${candidate_mode}" \
      --preserved-dense-count "${preserved_dense_count}" \
      --dense-root-count 20 \
      --agsto-textual-seed-top-k 20 \
      --agsto-closure-hops 2 \
      --agsto-max-endpoint-degree 30 \
      --candidate-pool-k 300 \
      --top-k 5 \
      --enable-variable-flow-traversal \
      --profile-retrieval
  ) >> "${log}" 2>&1
  local rc=$?

  if [[ "${rc}" -eq 0 ]]; then
    printf 'done\t%s\n' "$(date -Is)" > "${status}"
  else
    printf 'failed\trc=%s\t%s\n' "${rc}" "$(date -Is)" > "${status}"
  fi
  printf '[%s] END %s rc=%s\n' "$(date -Is)" "${tag}" "${rc}" | tee -a "${log}"
  return "${rc}"
}

rc=0
run_variant clean_pool300 dense_seeded_sto 300 || rc=1
run_variant dense_preserve_pool300 dense_preserving_sto 300 || rc=1
exit "${rc}"
