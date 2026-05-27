#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
BASE_INDEX="${ROOT_DIR}/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/index"
RUN_ROOT="${ROOT_DIR}/run_logs/etv4_seed_grid_musique_limit100_20260513"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status"

run_variant() {
  local roots="$1"
  local tag="dense_seeded_roots${roots}"
  local out="${RUN_ROOT}/${tag}"
  local ds_dir="${out}/musique"
  local log="${RUN_ROOT}/logs/${tag}.log"
  local status="${RUN_ROOT}/status/${tag}.status"
  mkdir -p "${ds_dir}"
  ln -sfn "${BASE_INDEX}" "${ds_dir}/index"
  printf 'running\t%s\n' "$(date -Is)" > "${status}"
  (
    cd "${ROOT_DIR}" || exit 1
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_fresh_e2e.py \
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
      --dense-root-count "${roots}" \
      --agsto-textual-seed-top-k "${roots}" \
      --candidate-pool-k 200 \
      --top-k 5 \
      --enable-variable-flow-traversal
  ) > "${log}" 2>&1
  local rc=$?
  if [[ "${rc}" -eq 0 ]]; then
    printf 'done\t%s\n' "$(date -Is)" > "${status}"
  else
    printf 'failed\trc=%s\t%s\n' "${rc}" "$(date -Is)" > "${status}"
  fi
  return "${rc}"
}

run_variant 50 &
pid50=$!
run_variant 100 &
pid100=$!
run_variant 200 &
pid200=$!

rc=0
wait "${pid50}" || rc=1
wait "${pid100}" || rc=1
wait "${pid200}" || rc=1
exit "${rc}"
