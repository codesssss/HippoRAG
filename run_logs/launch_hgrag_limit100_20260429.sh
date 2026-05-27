#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
RUN_TAG="hgrag_aligned_limit100_20260429"
OUT_DIR="${ROOT_DIR}/run_logs/${RUN_TAG}"
HGRAG_ROOT="${HGRAG_ROOT:-/mnt/nvme/code/HGRAG}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"

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
ok = (
    data.get("method") == "hgrag"
    and data.get("num_queries") == 100
    and bool(data.get("overall_retrieval_result"))
    and bool(data.get("overall_qa_results"))
)
sys.exit(0 if ok else 1)
PY
}

dataset_port() {
  case "$1" in
    2wikimultihopqa) echo 8041 ;;
    hotpotqa) echo 8042 ;;
    musique) echo 8043 ;;
    *) return 2 ;;
  esac
}

run_one() {
  local dataset="$1"
  local port
  port="$(dataset_port "${dataset}")" || return 2

  local output_json="${OUT_DIR}/${dataset}_hgrag.json"
  local log_path="${OUT_DIR}/${dataset}_hgrag.log"
  local status_path="${OUT_DIR}/${dataset}_hgrag.status"

  if is_done_json "${output_json}"; then
    echo "[SKIP] dataset=${dataset} hgrag complete output=${output_json}" | tee "${status_path}"
    return 0
  fi

  echo "[START] dataset=${dataset} variant=hgrag port=${port} start=$(date -Is)" | tee "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/run_hgrag_aligned.py \
      --dataset "${dataset}" \
      --limit 100 \
      --hgrag_root "${HGRAG_ROOT}" \
      --output_json "${output_json}" \
      --llm_name qwen3-8b-train \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name nvidia/NV-Embed-v2 \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --qa_top_k 5 \
      --recall_top_k 20 \
      --e2e_top_k 20 \
      --ent_topk 1 \
      --beta 0.5 \
      --step 2 \
      --embedding_batch_size 4 \
      --max_workers 16 \
      --hgraph_device cpu \
      --ner_max_new_tokens 512 \
      --qa_max_new_tokens 512
  ) > "${log_path}" 2>&1
  local code=$?

  if [[ "${code}" -eq 0 ]] && is_done_json "${output_json}"; then
    echo "[DONE] dataset=${dataset} variant=hgrag end=$(date -Is) output=${output_json}" | tee "${status_path}"
  else
    echo "[FAILED] dataset=${dataset} variant=hgrag code=${code} end=$(date -Is) log=${log_path}" | tee "${status_path}"
    return "${code}"
  fi
}

echo "[START] ${RUN_TAG} $(date -Is)" | tee "${OUT_DIR}/launcher.status"
rc=0
run_one 2wikimultihopqa || rc=1
if [[ "${rc}" -eq 0 ]]; then
  run_one hotpotqa || rc=1
fi
if [[ "${rc}" -eq 0 ]]; then
  run_one musique || rc=1
fi

if [[ "${rc}" -eq 0 ]]; then
  echo "[DONE] ${RUN_TAG} $(date -Is)" | tee "${OUT_DIR}/launcher.status"
else
  echo "[FAILED] ${RUN_TAG} rc=${rc} $(date -Is)" | tee "${OUT_DIR}/launcher.status"
fi
exit "${rc}"
