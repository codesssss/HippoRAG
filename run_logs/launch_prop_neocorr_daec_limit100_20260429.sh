#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
RUN_TAG="prop_neocorr_daec_limit100_20260429"
OUT_DIR="${ROOT_DIR}/run_logs/${RUN_TAG}"
POOL_DIR="${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424"
SAVE_DIR="outputs_step0_general_nvembed"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
NEOCORRAG_ROOT="${NEOCORRAG_ROOT:-/mnt/nvme/code/NeocorRAG}"
NEOCORRAG_CUDA_VISIBLE_DEVICES="${NEOCORRAG_CUDA_VISIBLE_DEVICES:-4}"

mkdir -p "${OUT_DIR}"

is_done_json() {
  local path="$1"
  local variant="$2"
  [[ -s "${path}" ]] || return 1
  "${PYTHON_BIN}" - "${path}" "${variant}" <<'PY'
import json
import sys

path, variant = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(path))
except Exception:
    sys.exit(1)

if data.get("limit") != 100 and data.get("num_queries") != 100:
    # NeocorRAG combined outputs use num_queries; HippoRAG reports use limit.
    sys.exit(1)

if variant == "proprag":
    overall = data.get("overall_recomputed") or {}
    ok = all(k in overall for k in ["Recall@5", "Recall@20", "ExactMatch", "F1"])
elif variant in {"daec", "daec_l1"}:
    selector = data.get("setwise_selector_qa") or {}
    retrieval = selector.get("selector_retrieval_metrics") or {}
    ok = all(k in retrieval for k in ["Recall@5", "Recall@20"]) and all(
        k in selector for k in ["selector_EM", "selector_F1"]
    )
elif variant == "neocorrag":
    ok = bool(data.get("overall_retrieval_result")) and bool(data.get("overall_qa_results"))
else:
    ok = False
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

run_hipporag_eval() {
  local dataset="$1"
  local variant="$2"
  local port="$3"
  local pool_json="${POOL_DIR}/${dataset}_pool100.json"
  local output_json="${OUT_DIR}/${dataset}_${variant}.json"
  local log_path="${OUT_DIR}/${dataset}_${variant}.log"
  local status_path="${OUT_DIR}/${dataset}_${variant}.status"

  if [[ ! -s "${pool_json}" ]]; then
    echo "[ERROR] missing PropRAG pool: ${pool_json}" | tee "${status_path}"
    return 2
  fi

  if is_done_json "${output_json}" "${variant}"; then
    echo "[SKIP] ${dataset}/${variant} already complete: ${output_json}" | tee "${status_path}"
    return 0
  fi

  local selector_args=()
  case "${variant}" in
    proprag)
      selector_args=(--setwise_selector none)
      ;;
    daec)
      selector_args=(
        --setwise_selector dtc_embed
        --setwise_pool_k 100
        --setwise_reserve_top_m 0
        --dtc_rank_weight 0.2
        --dtc_include_satisfiable_by true
        --dtc_repairable_filter_enabled true
        --dtc_satisfiable_by_policy binding_override
        --dtc_enable_dependency_binding true
        --dtc_binding_max_candidates 4
        --dtc_binding_entity_hit_required true
      )
      ;;
    daec_l1)
      selector_args=(
        --setwise_selector daec_noisyor
        --setwise_pool_k 100
        --dtc_decomposition_mode llm
        --dtc_binding_max_candidates 5
      )
      ;;
    *)
      echo "[ERROR] unknown variant=${variant}" | tee "${status_path}"
      return 2
      ;;
  esac

  echo "[START] dataset=${dataset} variant=${variant} port=${port} start=$(date -Is)" | tee "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit 100 \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name qwen3-8b-train \
      --max_retry_attempts 20 \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name VLLM/nvidia/NV-Embed-v2 \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --external_pool_json "${pool_json}" \
      --external_pool_source_name proprag_clean_nothink_top100 \
      --external_pool_strict_questions true \
      "${selector_args[@]}" \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1

  local code=$?
  if [[ "${code}" -eq 0 ]] && is_done_json "${output_json}" "${variant}"; then
    echo "[DONE] dataset=${dataset} variant=${variant} end=$(date -Is) output=${output_json}" | tee "${status_path}"
  else
    echo "[FAILED] dataset=${dataset} variant=${variant} code=${code} end=$(date -Is) log=${log_path}" | tee "${status_path}"
    return "${code}"
  fi
}

run_dataset_chain() {
  local dataset="$1"
  local port
  port="$(dataset_port "${dataset}")" || return 2

  run_hipporag_eval "${dataset}" proprag "${port}" || return 1
  run_hipporag_eval "${dataset}" daec "${port}" || return 1
  run_hipporag_eval "${dataset}" daec_l1 "${port}" || return 1
}

run_neocorrag_dataset() {
  local dataset="$1"
  local port="$2"
  local output_json="${OUT_DIR}/${dataset}_neocorrag.json"
  local log_path="${OUT_DIR}/${dataset}_neocorrag.log"
  local status_path="${OUT_DIR}/${dataset}_neocorrag.status"

  if is_done_json "${output_json}" neocorrag; then
    echo "[SKIP] ${dataset}/neocorrag already complete: ${output_json}" | tee "${status_path}"
    return 0
  fi

  echo "[START] dataset=${dataset} variant=neocorrag port=${port} cuda=${NEOCORRAG_CUDA_VISIBLE_DEVICES} start=$(date -Is)" | tee "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    CUDA_VISIBLE_DEVICES="${NEOCORRAG_CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" scripts/run_neocorrag_aligned.py \
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
  if [[ "${code}" -eq 0 ]] && is_done_json "${output_json}" neocorrag; then
    echo "[DONE] dataset=${dataset} variant=neocorrag end=$(date -Is) output=${output_json}" | tee "${status_path}"
  else
    echo "[FAILED] dataset=${dataset} variant=neocorrag code=${code} end=$(date -Is) log=${log_path}" | tee "${status_path}"
    return "${code}"
  fi
}

run_neocorrag_chain() {
  # Run sequentially because NeocorRAG loads a local constrained-decoding model.
  run_neocorrag_dataset 2wikimultihopqa 8041 || return 1
  run_neocorrag_dataset hotpotqa 8042 || return 1
  run_neocorrag_dataset musique 8043 || return 1
}

echo "[START] ${RUN_TAG} $(date -Is)" | tee "${OUT_DIR}/launcher.status"

rc=0
pids=()
for dataset in 2wikimultihopqa hotpotqa musique; do
  run_dataset_chain "${dataset}" &
  pid=$!
  pids+=("${pid}")
  echo "${pid}" > "${OUT_DIR}/${dataset}_hipporag_chain.pid"
done

run_neocorrag_chain &
neocorr_pid=$!
echo "${neocorr_pid}" > "${OUT_DIR}/neocorrag_chain.pid"
pids+=("${neocorr_pid}")

echo "[RUNNING] ${RUN_TAG} pids=${pids[*]} $(date -Is)" | tee "${OUT_DIR}/launcher.status"

for pid in "${pids[@]}"; do
  wait "${pid}" || rc=1
done

"${PYTHON_BIN}" scripts/summarize_prop_neocorr_daec_limit100.py \
  --results_dir "${OUT_DIR}" \
  --output_md "${ROOT_DIR}/reports/dpathrag/${RUN_TAG}.md" \
  --output_json "${ROOT_DIR}/reports/dpathrag/${RUN_TAG}.json" \
  > "${OUT_DIR}/summary.log" 2>&1 || rc=1

if [[ "${rc}" -eq 0 ]]; then
  echo "[DONE] ${RUN_TAG} $(date -Is)" | tee "${OUT_DIR}/launcher.status"
else
  echo "[FAILED] ${RUN_TAG} rc=${rc} $(date -Is)" | tee "${OUT_DIR}/launcher.status"
fi
exit "${rc}"
