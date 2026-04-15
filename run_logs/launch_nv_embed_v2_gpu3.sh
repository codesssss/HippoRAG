#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

PY=".venv-hipporag/bin/python"
SCRIPT="scripts/serve_nv_embed_v2.py"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8019}"
CUDA_DEVICE="${CUDA_DEVICE:-3}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/mnt/nvme/hf/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/nv_embed_v2_runtime}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia/NV-Embed-v2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
STARTUP_WARMUP_BATCH_SIZE="${STARTUP_WARMUP_BATCH_SIZE:-4}"
CUDA_RESERVE_MIB="${CUDA_RESERVE_MIB:-8192}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF_VALUE:-expandable_segments:True}"

LOG="run_logs/launch_nv_embed_v2_gpu3.log"
STATUS="run_logs/launch_nv_embed_v2_gpu3.status"
PID_FILE="run_logs/launch_nv_embed_v2_gpu3.pid"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"
}

if ss -tnlp | grep -q ":${PORT} "; then
  echo "PORT_BUSY" > "$STATUS"
  log "Port ${PORT} is already in use. Refusing to start a second NV-Embed-v2 service."
  exit 1
fi

echo "STARTING" > "$STATUS"
: > "$LOG"

log "Starting NV-Embed-v2 service on CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} port=${PORT}"
LAUNCH_CMD="cd '$ROOT' && exec env \
CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}' \
HF_HOME=/mnt/nvme/hf \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF_VALUE}' \
'$PY' '$SCRIPT' \
--host '$HOST' \
--port '$PORT' \
--served-model-name '$SERVED_MODEL_NAME' \
--model-snapshot '$MODEL_SNAPSHOT' \
--runtime-root '$RUNTIME_ROOT' \
--max-length '$MAX_LENGTH' \
--max-batch-size '$MAX_BATCH_SIZE' \
--torch-dtype '$TORCH_DTYPE' \
--startup-warmup-batch-size '$STARTUP_WARMUP_BATCH_SIZE' \
--cuda-reserve-mib '$CUDA_RESERVE_MIB' \
>> '$LOG' 2>&1"

setsid /bin/bash -lc "$LAUNCH_CMD" >/dev/null 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
log "Spawned PID=${PID}"

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    echo "READY" > "$STATUS"
    log "Service is READY at http://127.0.0.1:${PORT}/v1"
    exit 0
  fi
  sleep 2
done

echo "FAILED" > "$STATUS"
log "Service failed to become ready within timeout. Check ${LOG}"
exit 1
