#!/usr/bin/env bash
set -u

ROOT="/mnt/nvme/code/HippoRAG"
PY="${ROOT}/.venv-hipporag/bin/python"
OUT="${ROOT}/run_logs/neocorrag_aligned_full1000_20260429"
EMB="http://localhost:8019/v1/embeddings"
NEO_ROOT="/mnt/nvme/code/NeocorRAG"
CUDA_DEV="${NEOCORRAG_CUDA_VISIBLE_DEVICES:-4}"

mkdir -p "${OUT}"

dataset_port() {
  case "$1" in
    2wikimultihopqa) echo 8041 ;;
    hotpotqa) echo 8042 ;;
    musique) echo 8043 ;;
    *) return 2 ;;
  esac
}

is_done_json() {
  local dataset="$1"
  local path="$2"
  [[ -s "${path}" ]] || return 1
  "${PY}" - "${dataset}" "${path}" <<'PY'
import json
import sys

dataset, path = sys.argv[1:3]
try:
    data = json.load(open(path))
except Exception:
    sys.exit(1)

config = data.get("config") or {}
ok = (
    data.get("dataset") == dataset
    and data.get("method") == "neocorrag"
    and int(data.get("limit") or 0) == 1000
    and int(config.get("retrieval_top_k") or 0) == 100
    and config.get("reretrieval_llm_name") == "/mnt/nvme/Qwen3-8B"
    and config.get("reretrieval_embedding_name") == "nvidia/NV-Embed-v2"
    and config.get("no_think") is True
    and bool(data.get("overall_retrieval_result"))
    and bool(data.get("overall_qa_results"))
)
sys.exit(0 if ok else 1)
PY
}

run_one() {
  local dataset="$1"
  local port
  port="$(dataset_port "${dataset}")" || return 2

  local output_json="${OUT}/${dataset}_neocorrag.json"
  local log_path="${OUT}/${dataset}_neocorrag.log"
  local status_path="${OUT}/${dataset}_neocorrag.status"

  if is_done_json "${dataset}" "${output_json}"; then
    echo "[SKIP] dataset=${dataset} variant=neocorrag existing=${output_json}" > "${status_path}"
    return 0
  fi

  rm -f "${output_json}.lock"
  echo "[START] dataset=${dataset} variant=neocorrag transformers=4.57.6 parallel=true port=${port} cuda=${CUDA_DEV} start=$(date -Is)" > "${status_path}"

  (
    cd "${ROOT}" || exit 1
    CUDA_VISIBLE_DEVICES="${CUDA_DEV}" "${PY}" scripts/run_neocorrag_aligned.py \
      --dataset "${dataset}" \
      --limit 1000 \
      --neocorrag_root "${NEO_ROOT}" \
      --output_json "${output_json}" \
      --llm_name qwen3-8b-train \
      --llm_base_url "http://localhost:${port}/v1" \
      --graph_llm_name qwen3-8b-train \
      --graph_llm_base_url "http://localhost:${port}/v1" \
      --embedding_name nvidia/NV-Embed-v2 \
      --embedding_base_url "${EMB}" \
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

  if [[ "${code}" -eq 0 ]] && is_done_json "${dataset}" "${output_json}"; then
    echo "[DONE] dataset=${dataset} variant=neocorrag transformers=4.57.6 parallel=true end=$(date -Is) output=${output_json}" > "${status_path}"
    return 0
  fi

  echo "[FAILED] dataset=${dataset} variant=neocorrag transformers=4.57.6 parallel=true code=${code} end=$(date -Is) log=${log_path}" > "${status_path}"
  return "${code}"
}

main() {
  local launcher_status="${OUT}/relaunch_transformers457.status"
  echo "[START] relaunch_neocorrag_transformers457 start=$(date -Is)" > "${launcher_status}"

  local pids=()
  local dataset
  for dataset in 2wikimultihopqa hotpotqa musique; do
    run_one "${dataset}" &
    pids+=("$!")
  done

  local rc=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done

  if [[ "${rc}" -eq 0 ]]; then
    echo "[DONE] relaunch_neocorrag_transformers457 end=$(date -Is)" > "${launcher_status}"
  else
    echo "[FAILED] relaunch_neocorrag_transformers457 rc=${rc} end=$(date -Is)" > "${launcher_status}"
  fi
  return "${rc}"
}

main "$@"
