#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
BASE_INDEX="${ROOT_DIR}/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/index"
RUN_ROOT="${ROOT_DIR}/run_logs/etv4_readout_grid_musique_limit100_20260514"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status"

run_variant() {
  local policy="$1"
  local out="${RUN_ROOT}/${policy}"
  local ds_dir="${out}/musique"
  local log="${RUN_ROOT}/logs/${policy}.log"
  local status="${RUN_ROOT}/status/${policy}.status"

  mkdir -p "${ds_dir}"
  ln -sfn "${BASE_INDEX}" "${ds_dir}/index"
  printf 'running\t%s\n' "$(date -Is)" > "${status}"
  printf '[%s] START %s\n' "$(date -Is)" "${policy}" | tee -a "${log}"

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
      --candidate-generator-mode dense_seeded_sto \
      --dense-root-count 20 \
      --agsto-textual-seed-top-k 20 \
      --agsto-closure-hops 2 \
      --agsto-max-endpoint-degree 30 \
      --candidate-pool-k 200 \
      --top-k 5 \
      --enable-variable-flow-traversal \
      --readout-policy "${policy}"
  ) >> "${log}" 2>&1
  local rc=$?

  if [[ "${rc}" -eq 0 ]]; then
    printf 'done\t%s\n' "$(date -Is)" > "${status}"
  else
    printf 'failed\trc=%s\t%s\n' "${rc}" "$(date -Is)" > "${status}"
  fi
  printf '[%s] END %s rc=%s\n' "$(date -Is)" "${policy}" "${rc}" | tee -a "${log}"
  return "${rc}"
}

run_wave() {
  local rc=0
  local pids=()
  for policy in "$@"; do
    run_variant "${policy}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  return "${rc}"
}

rc=0
run_wave fact_witnessed_path_cover transition_valid_closure source_aligned || rc=1
run_wave layered_transition root_balanced_transition branch_balanced || rc=1
run_wave no_multi_anchor_precision raw_sto_adjacency || rc=1
exit "${rc}"
