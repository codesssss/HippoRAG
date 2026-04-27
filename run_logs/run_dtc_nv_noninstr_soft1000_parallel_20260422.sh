#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/nvme/code/HippoRAG
cd "${ROOT}"

PYTHON=.venv-hipporag/bin/python
EMBED_URL=http://localhost:8019/v1/embeddings

if ! grep -q '"input": input_text' src/hipporag/embedding_model/VLLM.py; then
  echo "[$(date '+%F %T')] REFUSE: VLLM embedding wrapper does not look like raw non-instruction input." >&2
  exit 2
fi

launch_one() {
  local dataset="$1"
  local outdir="$2"
  local port="$3"
  local tag="$4"
  local output_json="${outdir}/eval_reports/dtc_embed_nvembed_noninstr_soft_limit1000_anchor2_${tag}.json"
  local log="run_logs/dtc_nv_noninstr_soft1000_${dataset}_${tag}_20260422.log"
  local pidfile="run_logs/dtc_nv_noninstr_soft1000_${dataset}_${tag}_20260422.pid"

  mkdir -p "${outdir}/eval_reports"
  if [[ -s "${output_json}" ]]; then
    echo "[$(date '+%F %T')] SKIP ${dataset}: existing ${output_json}"
    return 0
  fi

  echo "[$(date '+%F %T')] START ${dataset}/noninstr_soft1000 on ${port} -> ${output_json}"
  setsid env PYTHONUNBUFFERED=1 "${PYTHON}" -u scripts/eval_causal_qwen3.py \
    --dataset "${dataset}" \
    --limit 1000 \
    --save_dir outputs_step0_general_nvembed \
    --llm_name qwen3-8b \
    --llm_request_name qwen3-8b-train \
    --llm_base_url "http://localhost:${port}/v1" \
    --embedding_name VLLM/nvidia/NV-Embed-v2 \
    --embedding_base_url "${EMBED_URL}" \
    --qa_top_k 5 \
    --setwise_selector dtc_embed \
    --setwise_pool_k 100 \
    --setwise_anchor_count 2 \
    --setwise_reserve_top_m 0 \
    --setwise_non_anchor_title_dedup true \
    --dtc_decomposition_mode llm \
    --dtc_enforce_dependencies true \
    --dtc_require_new_crossing false \
    --dtc_enable_dependency_binding false \
    --dtc_max_steps 4 \
    --dtc_match_threshold 0.35 \
    --dtc_redundancy_weight 0.10 \
    --dtc_base_weight 0.05 \
    --dtc_anchor_bonus_weight 0.10 \
    --dtc_dependency_bonus_weight 0.10 \
    --dtc_max_completion_tokens 512 \
    --output_json "${output_json}" \
    > "${log}" 2>&1 &
  echo "$!" > "${pidfile}"
  echo "[$(date '+%F %T')] LAUNCHED ${dataset}: pid=$(cat "${pidfile}") log=${log}"
}

launch_one 2wikimultihopqa outputs_step0_general_nvembed_2wikimultihopqa 8043 8043
launch_one hotpotqa outputs_step0_general_nvembed_hotpotqa 8042 8042
launch_one musique outputs_step0_general_nvembed_musique 8041 8041

echo "[$(date '+%F %T')] ALL_LAUNCHED noninstr soft1000"
