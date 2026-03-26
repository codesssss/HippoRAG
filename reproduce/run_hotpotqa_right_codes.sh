#!/usr/bin/env bash

set -euo pipefail

cd /mnt/nvme/code/HippoRAG

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HUGGINGFACE_HUB_CACHE=/mnt/nvme/hf
export TRANSFORMERS_CACHE=/mnt/nvme/hf
export HF_HOME=/mnt/nvme/hf
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# This key is only used for this one experiment run.
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-a5e91a4aa6ae4e46bafd5c348a6cb51c}"

# External OpenAI-compatible endpoint.
LLM_BASE_URL="https://www.right.codes/codex/v1"
LLM_NAME="gpt-5.4"

# Local qwen3-8b fallback. Uncomment these three lines if you want to switch
# back to the current local setup.
# export OPENAI_API_KEY=EMPTY
# LLM_BASE_URL="http://localhost:8039/v1"
# LLM_NAME="qwen3-8b"

LOG_FILE="/mnt/nvme/code/hotpotqa_gpt54_nvembed.log"
MAX_RETRY_ATTEMPTS=12
RUN_MAX_RESTARTS=5
RESTART_SLEEP_SECONDS=30

nohup /bin/bash -lc "
set -euo pipefail
cd /mnt/nvme/code/HippoRAG
for attempt in \$(seq 1 ${RUN_MAX_RESTARTS}); do
  echo \"[\$(date '+%F %T')] run attempt \$attempt/${RUN_MAX_RESTARTS}\" >> \"${LOG_FILE}\"
  if /mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python main.py \
    --dataset hotpotqa \
    --llm_base_url \"${LLM_BASE_URL}\" \
    --llm_name \"${LLM_NAME}\" \
    --embedding_name nvidia/NV-Embed-v2 \
    --max_retry_attempts ${MAX_RETRY_ATTEMPTS} \
    >> \"${LOG_FILE}\" 2>&1; then
    echo \"[\$(date '+%F %T')] run finished successfully\" >> \"${LOG_FILE}\"
    exit 0
  fi
  if [[ \$attempt -lt ${RUN_MAX_RESTARTS} ]]; then
    echo \"[\$(date '+%F %T')] run failed, sleeping ${RESTART_SLEEP_SECONDS}s before retry\" >> \"${LOG_FILE}\"
    sleep ${RESTART_SLEEP_SECONDS}
  fi
done
echo \"[\$(date '+%F %T')] run failed after ${RUN_MAX_RESTARTS} attempts\" >> \"${LOG_FILE}\"
exit 1
" >/dev/null 2>&1 &

echo "Started HotpotQA run."
echo "Log: ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
