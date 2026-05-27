#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
OUT_DIR="${ROOT_DIR}/run_logs/prop_neocorr_daec_limit100_20260429"
NEOCORRAG_ROOT="${NEOCORRAG_ROOT:-/mnt/nvme/code/NeocorRAG}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
CUDA_DEVICE="${NEOCORRAG_CUDA_VISIBLE_DEVICES:-4}"

mkdir -p "${OUT_DIR}"

is_done_json() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  "${PYTHON_BIN}" - "${path}" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
ok = bool(data.get("overall_retrieval_result")) and bool(data.get("overall_qa_results"))
sys.exit(0 if ok else 1)
PY
}

run_one() {
  local dataset="$1"
  local port="$2"
  local output_json="${OUT_DIR}/${dataset}_neocorrag.json"
  local log_path="${OUT_DIR}/${dataset}_neocorrag.log"
  local status_path="${OUT_DIR}/${dataset}_neocorrag.status"

  if is_done_json "${output_json}"; then
    echo "[SKIP] dataset=${dataset} neocorrag complete output=${output_json}" | tee "${status_path}"
    return 0
  fi

  echo "[START] dataset=${dataset} variant=neocorrag port=${port} cuda=${CUDA_DEVICE} start=$(date -Is)" | tee "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" scripts/run_neocorrag_aligned.py \
      --dataset "${dataset}" \
      --limit 100 \
      --neocorrag_root "${NEOCORRAG_ROOT}" \
      --output_json "${output_json}" \
      --llm_name qwen3-8b-train \
      --llm_base_url "http://localhost:${port}/v1" \
      --graph_llm_name qwen3-8b-train \
      --graph_llm_base_url "http://localhost:${port}/v1" \
      --embedding_name nvidia/NV-Embed-v2 \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --reretrieval_llm_name /mnt/nvme/Qwen3-8B \
      --reretrieval_embedding_name nvidia/NV-Embed-v2 \
      --generation_mode greedy \
      --k 1 \
      --qa_top_k 5 \
      --retrieval_top_k 100 \
      --embedding_batch_size 4 \
      --max_new_tokens 2048
  ) > "${log_path}" 2>&1
  local code=$?

  if [[ "${code}" -eq 0 ]] && is_done_json "${output_json}"; then
    echo "[DONE] dataset=${dataset} variant=neocorrag end=$(date -Is) output=${output_json}" | tee "${status_path}"
  else
    echo "[FAILED] dataset=${dataset} variant=neocorrag code=${code} end=$(date -Is) log=${log_path}" | tee "${status_path}"
    return "${code}"
  fi
}

echo "[START] neocorrag limit100 chain $(date -Is)" | tee "${OUT_DIR}/neocorrag_launcher.status"
rc=0
run_one 2wikimultihopqa 8041 || rc=1
if [[ "${rc}" -eq 0 ]]; then
  run_one hotpotqa 8042 || rc=1
fi
if [[ "${rc}" -eq 0 ]]; then
  run_one musique 8043 || rc=1
fi

if [[ "${rc}" -eq 0 ]]; then
  echo "[DONE] neocorrag limit100 chain $(date -Is)" | tee "${OUT_DIR}/neocorrag_launcher.status"
else
  echo "[FAILED] neocorrag limit100 chain rc=${rc} $(date -Is)" | tee "${OUT_DIR}/neocorrag_launcher.status"
fi
exit "${rc}"
