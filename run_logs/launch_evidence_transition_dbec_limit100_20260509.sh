#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
RUN_TAG="evidence_transition_dbec_limit100_20260509"
OUT_DIR="${ROOT_DIR}/run_logs/${RUN_TAG}"
POOL_JSON="${OUT_DIR}/pools/musique_evidence_transition_pool100.json"
SAVE_DIR="outputs_step0_general_nvembed"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"

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
selector = data.get("setwise_selector_qa") or {}
retrieval = selector.get("selector_retrieval_metrics") or {}
ok = (
    data.get("limit") == 100
    and selector.get("selector") in {"dtc_embed", "daec_noisyor"}
    and "Recall@5" in retrieval
    and "selector_EM" in selector
    and "selector_F1" in selector
)
if variant == "daec":
    ok = ok and selector.get("selector") == "dtc_embed"
elif variant == "daec_l1":
    ok = ok and selector.get("selector") == "daec_noisyor"
else:
    ok = False
sys.exit(0 if ok else 1)
PY
}

run_variant() {
  local variant="$1"
  local output_json="${OUT_DIR}/musique_${variant}.json"
  local log_path="${OUT_DIR}/musique_${variant}.log"
  local status_path="${OUT_DIR}/musique_${variant}.status"

  if [[ ! -s "${POOL_JSON}" ]]; then
    echo "[ERROR] missing pool json: ${POOL_JSON}" | tee "${status_path}"
    return 2
  fi
  if is_done_json "${output_json}" "${variant}"; then
    echo "[SKIP] musique/${variant} already complete: ${output_json}" | tee "${status_path}"
    return 0
  fi

  local selector_args=()
  case "${variant}" in
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

  echo "[START] dataset=musique variant=${variant} start=$(date -Is)" | tee "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset musique \
      --limit 100 \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name qwen3-8b-train \
      --max_retry_attempts 20 \
      --qwen_disable_thinking \
      --llm_base_url http://localhost:8043/v1 \
      --embedding_name VLLM/nvidia/NV-Embed-v2 \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --qa_top_k 5 \
      --qa_doc_max_chars 2048 \
      --external_pool_json "${POOL_JSON}" \
      --external_pool_source_name evidence_transition_pool100 \
      --external_pool_strict_questions true \
      "${selector_args[@]}" \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1

  local code=$?
  if [[ "${code}" -eq 0 ]] && is_done_json "${output_json}" "${variant}"; then
    echo "[DONE] dataset=musique variant=${variant} end=$(date -Is) output=${output_json}" | tee "${status_path}"
  else
    echo "[FAILED] dataset=musique variant=${variant} code=${code} end=$(date -Is) log=${log_path}" | tee "${status_path}"
    return "${code}"
  fi
}

echo "[START] ${RUN_TAG} $(date -Is)" | tee "${OUT_DIR}/launcher.status"
rc=0
run_variant daec || rc=1
run_variant daec_l1 || rc=1
if [[ "${rc}" -eq 0 ]]; then
  echo "[DONE] ${RUN_TAG} $(date -Is)" | tee "${OUT_DIR}/launcher.status"
else
  echo "[FAILED] ${RUN_TAG} rc=${rc} $(date -Is)" | tee "${OUT_DIR}/launcher.status"
fi
exit "${rc}"
