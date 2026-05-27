#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
SAVE_DIR="${SAVE_DIR:-outputs_step0_general_nvembed}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
NEOCORRAG_ROOT="${NEOCORRAG_ROOT:-/mnt/nvme/code/NeocorRAG}"
HGRAG_ROOT="${HGRAG_ROOT:-/mnt/nvme/code/HGRAG}"
NEOCORRAG_CUDA_VISIBLE_DEVICES="${NEOCORRAG_CUDA_VISIBLE_DEVICES:-4}"

RUN_TAG="aligned_full1000_20260429"
OUT_DIR="${ROOT_DIR}/run_logs/${RUN_TAG}"
DAEC_L1_DIR="${ROOT_DIR}/run_logs/daec_noisyor_proprag_pool100_full1000_20260429"
HGRAG_OUT_DIR="${ROOT_DIR}/run_logs/hgrag_aligned_full1000_20260429"
NEOCORRAG_OUT_DIR="${ROOT_DIR}/run_logs/neocorrag_aligned_full1000_20260429"

mkdir -p "${OUT_DIR}" "${DAEC_L1_DIR}" "${HGRAG_OUT_DIR}" "${NEOCORRAG_OUT_DIR}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${OUT_DIR}/launcher.log"
}

dataset_port() {
  case "$1" in
    2wikimultihopqa) echo 8041 ;;
    hotpotqa) echo 8042 ;;
    musique) echo 8043 ;;
    *) return 2 ;;
  esac
}

dataset_short() {
  case "$1" in
    2wikimultihopqa) echo 2wiki ;;
    hotpotqa) echo hotpotqa ;;
    musique) echo musique ;;
    *) return 2 ;;
  esac
}

pool_json() {
  case "$1" in
    2wikimultihopqa) echo "${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json" ;;
    hotpotqa) echo "${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424/hotpotqa_pool100.json" ;;
    musique) echo "${ROOT_DIR}/run_logs/proprag_pool_exports_full1000_20260424/musique_pool100.json" ;;
    *) return 2 ;;
  esac
}

prop_daec_json() {
  case "$1" in
    2wikimultihopqa) echo "${ROOT_DIR}/run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json" ;;
    hotpotqa) echo "${ROOT_DIR}/run_logs/layer1_proprag_pool_eval_fixed_20260424/hotpotqa_proprag_pool_daec_oracle.json" ;;
    musique) echo "${ROOT_DIR}/run_logs/layer1_proprag_pool_eval_fixed_20260424/musique_proprag_pool_daec_oracle.json" ;;
    *) return 2 ;;
  esac
}

daec_l1_json() {
  case "$1" in
    2wikimultihopqa) echo "${ROOT_DIR}/run_logs/daec_noisyor_proprag_pool100_2wiki_fixclean_full1000_20260428.json" ;;
    hotpotqa) echo "${DAEC_L1_DIR}/hotpotqa_daec_l1.json" ;;
    musique) echo "${DAEC_L1_DIR}/musique_daec_l1.json" ;;
    *) return 2 ;;
  esac
}

is_done_json() {
  local kind="$1"
  local dataset="$2"
  local path="$3"
  [[ -s "${path}" ]] || return 1
  "${PYTHON_BIN}" - "${kind}" "${dataset}" "${path}" <<'PY'
import json
import sys

kind, dataset, path = sys.argv[1:4]
try:
    data = json.load(open(path))
except Exception:
    sys.exit(1)

ok = data.get("dataset") == dataset
if kind in {"prop_daec", "daec_l1"}:
    selector = data.get("setwise_selector_qa") or {}
    retrieval = selector.get("selector_retrieval_metrics") or {}
    ok = (
        ok
        and int(data.get("limit") or 0) == 1000
        and all(k in retrieval for k in ["Recall@5", "Recall@20"])
        and all(k in selector for k in ["selector_EM", "selector_F1"])
    )
elif kind == "hgrag":
    ok = (
        ok
        and data.get("method") == "hgrag"
        and int(data.get("num_queries") or 0) == 1000
        and bool(data.get("overall_retrieval_result"))
        and bool(data.get("overall_qa_results"))
        and (data.get("config") or {}).get("no_think") is True
    )
elif kind == "neocorrag":
    config = data.get("config") or {}
    ok = (
        ok
        and data.get("method") == "neocorrag"
        and int(data.get("limit") or 0) == 1000
        and int(config.get("retrieval_top_k") or 0) == 100
        and config.get("reretrieval_embedding_name") == "nvidia/NV-Embed-v2"
        and config.get("no_think") is True
        and bool(data.get("overall_retrieval_result"))
        and bool(data.get("overall_qa_results"))
    )
else:
    ok = False

sys.exit(0 if ok else 1)
PY
}

record_existing_prop_daec() {
  local dataset="$1"
  local path
  path="$(prop_daec_json "${dataset}")" || return 2
  local status_path="${OUT_DIR}/${dataset}_prop_daec.status"
  if is_done_json prop_daec "${dataset}" "${path}"; then
    echo "[SKIP] dataset=${dataset} variant=prop_daec existing=${path}" | tee "${status_path}"
    return 0
  fi
  echo "[MISSING] dataset=${dataset} variant=prop_daec expected=${path}" | tee "${status_path}"
  return 1
}

run_daec_l1_dataset() {
  local dataset="$1"
  local port
  local pool
  local output_json
  local log_path
  local status_path
  port="$(dataset_port "${dataset}")" || return 2
  pool="$(pool_json "${dataset}")" || return 2
  output_json="$(daec_l1_json "${dataset}")" || return 2
  log_path="${DAEC_L1_DIR}/${dataset}_daec_l1.log"
  status_path="${DAEC_L1_DIR}/${dataset}_daec_l1.status"

  if is_done_json daec_l1 "${dataset}" "${output_json}"; then
    echo "[SKIP] dataset=${dataset} variant=daec_l1 existing=${output_json}" | tee "${status_path}"
    return 0
  fi

  echo "[START] dataset=${dataset} variant=daec_l1 port=${port} start=$(date -Is)" | tee "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \
      --dataset "${dataset}" \
      --limit 1000 \
      --save_dir "${SAVE_DIR}" \
      --llm_name qwen3-8b \
      --llm_request_name qwen3-8b-train \
      --max_retry_attempts 20 \
      --llm_base_url "http://localhost:${port}/v1" \
      --embedding_name VLLM/nvidia/NV-Embed-v2 \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --external_pool_json "${pool}" \
      --external_pool_source_name proprag_pool100 \
      --external_pool_strict_questions true \
      --setwise_selector daec_noisyor \
      --setwise_pool_k 100 \
      --qa_top_k 5 \
      --dtc_decomposition_mode llm \
      --dtc_binding_max_candidates 5 \
      --output_json "${output_json}"
  ) > "${log_path}" 2>&1
  local code=$?

  if [[ "${code}" -eq 0 ]] && is_done_json daec_l1 "${dataset}" "${output_json}"; then
    echo "[DONE] dataset=${dataset} variant=daec_l1 end=$(date -Is) output=${output_json}" | tee "${status_path}"
    return 0
  fi
  echo "[FAILED] dataset=${dataset} variant=daec_l1 code=${code} end=$(date -Is) log=${log_path}" | tee "${status_path}"
  return "${code}"
}

seed_hgrag_full1000_cache() {
  local dataset="$1"
  "${PYTHON_BIN}" - "${dataset}" "${HGRAG_ROOT}" "${ROOT_DIR}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

dataset, hgrag_root, root = sys.argv[1:4]
root = Path(root)
hgrag_root = Path(hgrag_root)
old_json = root / "run_logs" / "hgrag_aligned_limit100_20260429" / f"{dataset}_hgrag.json"
new_work = hgrag_root / "output" / "aligned_limit1000_qwen3-8b-train_nothink" / dataset
if not old_json.exists():
    print(f"[HGRAG CACHE] no limit100 json for {dataset}: {old_json}")
    sys.exit(0)
try:
    old = json.load(open(old_json))
except Exception as exc:
    print(f"[HGRAG CACHE] cannot read {old_json}: {exc}")
    sys.exit(0)
old_work = Path(old.get("work_dir") or "")
if not old_work.exists():
    print(f"[HGRAG CACHE] no old work_dir for {dataset}: {old_work}")
    sys.exit(0)

pairs = [
    ("ner/c_ner_resp.jsonl", "ner/c_ner_resp.jsonl"),
    ("vecs/c_ent_vecs.pkl", "vecs/c_ent_vecs.pkl"),
    ("vecs/doc_vecs.pkl", "vecs/doc_vecs.pkl"),
]
copied = []
for src_rel, dst_rel in pairs:
    src = old_work / src_rel
    dst = new_work / dst_rel
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst_rel)
print(f"[HGRAG CACHE] dataset={dataset} copied={copied}")
PY
}

run_hgrag_dataset() {
  local dataset="$1"
  local port
  local output_json="${HGRAG_OUT_DIR}/${dataset}_hgrag.json"
  local log_path="${HGRAG_OUT_DIR}/${dataset}_hgrag.log"
  local status_path="${HGRAG_OUT_DIR}/${dataset}_hgrag.status"
  port="$(dataset_port "${dataset}")" || return 2

  if is_done_json hgrag "${dataset}" "${output_json}"; then
    echo "[SKIP] dataset=${dataset} variant=hgrag existing=${output_json}" | tee "${status_path}"
    return 0
  fi

  seed_hgrag_full1000_cache "${dataset}" | tee -a "${status_path}"
  echo "[START] dataset=${dataset} variant=hgrag port=${port} start=$(date -Is)" | tee "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    "${PYTHON_BIN}" scripts/run_hgrag_aligned.py \
      --dataset "${dataset}" \
      --limit 1000 \
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

  if [[ "${code}" -eq 0 ]] && is_done_json hgrag "${dataset}" "${output_json}"; then
    echo "[DONE] dataset=${dataset} variant=hgrag end=$(date -Is) output=${output_json}" | tee "${status_path}"
    return 0
  fi
  echo "[FAILED] dataset=${dataset} variant=hgrag code=${code} end=$(date -Is) log=${log_path}" | tee "${status_path}"
  return "${code}"
}

run_neocorrag_dataset() {
  local dataset="$1"
  local port
  local output_json="${NEOCORRAG_OUT_DIR}/${dataset}_neocorrag.json"
  local log_path="${NEOCORRAG_OUT_DIR}/${dataset}_neocorrag.log"
  local status_path="${NEOCORRAG_OUT_DIR}/${dataset}_neocorrag.status"
  port="$(dataset_port "${dataset}")" || return 2

  if is_done_json neocorrag "${dataset}" "${output_json}"; then
    echo "[SKIP] dataset=${dataset} variant=neocorrag existing=${output_json}" | tee "${status_path}"
    return 0
  fi

  echo "[START] dataset=${dataset} variant=neocorrag port=${port} cuda=${NEOCORRAG_CUDA_VISIBLE_DEVICES} start=$(date -Is)" | tee "${status_path}"
  (
    cd "${ROOT_DIR}" || exit 1
    CUDA_VISIBLE_DEVICES="${NEOCORRAG_CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" scripts/run_neocorrag_aligned.py \
      --dataset "${dataset}" \
      --limit 1000 \
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

  if [[ "${code}" -eq 0 ]] && is_done_json neocorrag "${dataset}" "${output_json}"; then
    echo "[DONE] dataset=${dataset} variant=neocorrag end=$(date -Is) output=${output_json}" | tee "${status_path}"
    return 0
  fi
  echo "[FAILED] dataset=${dataset} variant=neocorrag code=${code} end=$(date -Is) log=${log_path}" | tee "${status_path}"
  return "${code}"
}

run_daec_l1_missing_parallel() {
  local pids=()
  for dataset in 2wikimultihopqa hotpotqa musique; do
    local output_json
    output_json="$(daec_l1_json "${dataset}")" || return 2
    if is_done_json daec_l1 "${dataset}" "${output_json}"; then
      echo "[SKIP] dataset=${dataset} variant=daec_l1 existing=${output_json}" | tee "${DAEC_L1_DIR}/${dataset}_daec_l1.status"
      continue
    fi
    run_daec_l1_dataset "${dataset}" &
    pids+=("$!")
  done
  local rc=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  return "${rc}"
}

run_hgrag_chain() {
  local dataset
  for dataset in 2wikimultihopqa hotpotqa musique; do
    run_hgrag_dataset "${dataset}" || return 1
  done
}

run_hgrag_remaining_parallel() {
  local pids=()
  local dataset
  for dataset in 2wikimultihopqa hotpotqa musique; do
    local output_json="${HGRAG_OUT_DIR}/${dataset}_hgrag.json"
    if is_done_json hgrag "${dataset}" "${output_json}"; then
      echo "[SKIP] dataset=${dataset} variant=hgrag existing=${output_json}" | tee "${HGRAG_OUT_DIR}/${dataset}_hgrag.status"
      continue
    fi
    run_hgrag_dataset "${dataset}" &
    pids+=("$!")
  done
  local rc=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  return "${rc}"
}

run_neocorrag_chain() {
  local dataset
  for dataset in 2wikimultihopqa hotpotqa musique; do
    run_neocorrag_dataset "${dataset}" || return 1
  done
}

main() {
  echo "[START] ${RUN_TAG} $(date -Is)" | tee "${OUT_DIR}/launcher.status"
  log "Preflight: Prop+DAEC existing full1000 rows"
  local rc=0
  local dataset
  for dataset in 2wikimultihopqa hotpotqa musique; do
    record_existing_prop_daec "${dataset}" || rc=1
  done
  if [[ "${rc}" -ne 0 ]]; then
    log "One or more Prop+DAEC full1000 artifacts are missing; not fabricating replacements in this launcher."
  fi

  log "Stage 1: Prop+DAEC-L1 missing rows plus HGRAG 2Wiki in parallel"
  stage1_pids=()
  run_daec_l1_missing_parallel &
  stage1_pids+=("$!")
  run_hgrag_dataset 2wikimultihopqa &
  stage1_pids+=("$!")
  for pid in "${stage1_pids[@]}"; do
    wait "${pid}" || rc=1
  done

  log "Stage 2: remaining HGRAG native-shaped full1000 rows"
  if [[ "${rc}" -eq 0 ]]; then
    run_hgrag_remaining_parallel || rc=1
  else
    log "Skipping HGRAG because previous stage failed"
  fi

  log "Stage 3: NEOCORRAG native full1000 rows"
  if [[ "${rc}" -eq 0 ]]; then
    run_neocorrag_chain || rc=1
  else
    log "Skipping NEOCORRAG because previous stage failed"
  fi

  if [[ "${rc}" -eq 0 ]]; then
    echo "[DONE] ${RUN_TAG} $(date -Is)" | tee "${OUT_DIR}/launcher.status"
  else
    echo "[FAILED] ${RUN_TAG} rc=${rc} $(date -Is)" | tee "${OUT_DIR}/launcher.status"
  fi
  return "${rc}"
}

main "$@"
