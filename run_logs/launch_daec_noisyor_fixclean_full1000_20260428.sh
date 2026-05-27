#!/usr/bin/env bash
set -u

cd /mnt/nvme/code/HippoRAG || exit 1

RUN_TAG="fixclean_full1000_20260428"
PYTHON_BIN="${PYTHON_BIN:-python}"
PORTS=(8041 8042 8043)

port_busy() {
  local port="$1"
  "${PYTHON_BIN}" - "$port" <<'PY'
import os
import subprocess
import sys

port = sys.argv[1]
needle = f"localhost:{port}/v1"
try:
    output = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
except Exception:
    sys.exit(1)

self_pid = os.getpid()
for line in output.splitlines():
    stripped = line.strip()
    if not stripped:
        continue
    parts = stripped.split(None, 1)
    pid = int(parts[0]) if parts and parts[0].isdigit() else -1
    cmd = parts[1] if len(parts) > 1 else ""
    if pid == self_pid:
        continue
    if needle in cmd and "vllm.entrypoints.openai.api_server" not in cmd:
        sys.exit(0)
sys.exit(1)
PY
}

first_free_port() {
  local port
  for port in "${PORTS[@]}"; do
    if ! port_busy "$port"; then
      echo "$port"
      return 0
    fi
  done
  return 1
}

launch_eval() {
  local name="$1"
  local port="$2"
  local dataset="$3"
  local pool_json="$4"
  local pool_source="$5"
  local output_json="$6"
  local log_path="run_logs/${name}_${RUN_TAG}.log"
  local pid_path="run_logs/${name}_${RUN_TAG}.pid"

  echo "[$(date -Is)] launching ${name} on ${port}" | tee -a "run_logs/daec_noisyor_${RUN_TAG}_launcher.log"
  nohup "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
    --dataset "${dataset}" \
    --limit 1000 \
    --save_dir outputs_step0_general_nvembed \
    --llm_name qwen3-8b \
    --llm_request_name qwen3-8b-train \
    --llm_base_url "http://localhost:${port}/v1" \
    --embedding_name VLLM/nvidia/NV-Embed-v2 \
    --embedding_base_url http://localhost:8019/v1/embeddings \
    --external_pool_json "${pool_json}" \
    --external_pool_source_name "${pool_source}" \
    --external_pool_strict_questions true \
    --setwise_selector daec_noisyor \
    --setwise_pool_k 100 \
    --qa_top_k 5 \
    --dtc_decomposition_mode llm \
    --dtc_binding_max_candidates 5 \
    --output_json "${output_json}" \
    >"${log_path}" 2>&1 &
  echo "$!" > "${pid_path}"
  echo "[$(date -Is)] ${name} pid=$(cat "${pid_path}") log=${log_path} json=${output_json}" | tee -a "run_logs/daec_noisyor_${RUN_TAG}_launcher.log"
}

wait_and_launch() {
  local name="$1"
  local dataset="$2"
  local pool_json="$3"
  local pool_source="$4"
  local output_json="$5"
  local port=""

  while true; do
    port="$(first_free_port || true)"
    if [[ -n "${port}" ]]; then
      launch_eval "${name}" "${port}" "${dataset}" "${pool_json}" "${pool_source}" "${output_json}"
      sleep 10
      return 0
    fi
    echo "[$(date -Is)] waiting for a free Qwen3-8B endpoint for ${name}" | tee -a "run_logs/daec_noisyor_${RUN_TAG}_launcher.log"
    sleep 60
  done
}

echo "[$(date -Is)] DAEC noisy-OR full1000 launcher started" | tee -a "run_logs/daec_noisyor_${RUN_TAG}_launcher.log"

wait_and_launch \
  daec_noisyor_dense_pool100_2wiki \
  2wikimultihopqa \
  run_logs/dense_pool_exports_full1000_20260424/2wikimultihopqa_dense_pool100.json \
  dense_pool100 \
  run_logs/daec_noisyor_dense_pool100_2wiki_fixclean_full1000_20260428.json

wait_and_launch \
  daec_noisyor_dense_pool100_hotpotqa \
  hotpotqa \
  run_logs/dense_pool_exports_full1000_20260424/hotpotqa_dense_pool100.json \
  dense_pool100 \
  run_logs/daec_noisyor_dense_pool100_hotpotqa_fixclean_full1000_20260428.json

wait_and_launch \
  daec_noisyor_dense_pool100_musique \
  musique \
  run_logs/dense_pool_exports_full1000_20260424/musique_dense_pool100.json \
  dense_pool100 \
  run_logs/daec_noisyor_dense_pool100_musique_fixclean_full1000_20260428.json

wait_and_launch \
  daec_noisyor_proprag_pool100_2wiki \
  2wikimultihopqa \
  run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json \
  proprag_pool100 \
  run_logs/daec_noisyor_proprag_pool100_2wiki_fixclean_full1000_20260428.json

echo "[$(date -Is)] DAEC noisy-OR full1000 launcher finished dispatching all jobs" | tee -a "run_logs/daec_noisyor_${RUN_TAG}_launcher.log"
