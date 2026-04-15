#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/nvme/code/HippoRAG"
cd "$ROOT"

DATE_TAG="${DATE_TAG:-20260410nqpopqa100dual}"
STATUS="run_logs/launch_width_matched_control_nq_popqa_100_${DATE_TAG}.status"
LOG="run_logs/launch_width_matched_control_nq_popqa_100_${DATE_TAG}.log"
PID_FILE="run_logs/launch_width_matched_control_nq_popqa_100_${DATE_TAG}.pid"

NQ_SCRIPT="run_logs/width_matched_control_nq_100_20260410.sh"
POPQA_SCRIPT="run_logs/width_matched_control_popqa_100_20260410.sh"
SUMMARY_PY="run_logs/build_width_matched_control_nq_popqa_100_summary.py"

PY=".venv-hipporag/bin/python"

NQ_LLM_BASE_URL="${NQ_LLM_BASE_URL:-http://localhost:8043/v1}"
NQ_LLM_REQUEST_NAME="${NQ_LLM_REQUEST_NAME:-qwen3-8b-train}"
POPQA_LLM_BASE_URL="${POPQA_LLM_BASE_URL:-http://36.133.236.142:8002/v1}"
POPQA_LLM_REQUEST_NAME="${POPQA_LLM_REQUEST_NAME:-qwen3-8b}"
STANDBY_LLM_BASE_URL="${STANDBY_LLM_BASE_URL:-http://36.133.236.142:8003/v1}"
STANDBY_LLM_REQUEST_NAME="${STANDBY_LLM_REQUEST_NAME:-qwen3-8b}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://localhost:8018/v1/embeddings}"
SAVE_DIR="${SAVE_DIR:-outputs_step0_general}"
LIMIT="${LIMIT:-100}"
OPENAI_API_KEY_VALUE="${OPENAI_API_KEY_VALUE:-sk-local-placeholder}"
NQ_CUDA_VISIBLE_DEVICES="${NQ_CUDA_VISIBLE_DEVICES:-6}"
POPQA_CUDA_VISIBLE_DEVICES="${POPQA_CUDA_VISIBLE_DEVICES:-5}"
NQ_CE_DEVICE="${NQ_CE_DEVICE:-cuda:0}"
POPQA_CE_DEVICE="${POPQA_CE_DEVICE:-cuda:0}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"
}

healthcheck_models() {
  local name="$1"
  local url="$2"
  local body
  body="$(curl -sS "${url}/models")"
  [[ "$body" == *"\"data\""* ]] || {
    log "HEALTHCHECK FAILED name=${name} url=${url}/models"
    return 1
  }
  log "HEALTHCHECK OK name=${name} url=${url}/models"
}

healthcheck_chat() {
  local name="$1"
  local url="$2"
  local model_name="$3"
  local body
  body="$(curl -sS "${url}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${model_name}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK only.\"}],\"temperature\":0,\"max_tokens\":8}")"
  [[ "$body" == *"\"choices\""* ]] || {
    log "CHATCHECK FAILED name=${name} url=${url}/chat/completions model=${model_name}"
    return 1
  }
  log "CHATCHECK OK name=${name} url=${url}/chat/completions model=${model_name}"
}

check_local_embedding() {
  local body
  body="$(curl -sS "${EMBED_BASE_URL%/embeddings}/models")"
  [[ "$body" == *"\"data\""* ]] || {
    log "EMBEDCHECK FAILED url=${EMBED_BASE_URL%/embeddings}/models"
    return 1
  }
  log "EMBEDCHECK OK url=${EMBED_BASE_URL%/embeddings}/models"
}

check_index_ready() {
  local dataset="$1"
  local parquet_path="outputs_step0_general_${dataset}/qwen3-8b_VLLM__mnt_nvme_Qwen3-Embedding-8B/chunk_embeddings/vdb_chunk.parquet"
  [[ -f "$parquet_path" ]] || {
    log "INDEX MISSING dataset=${dataset} path=${parquet_path}"
    return 1
  }
  log "INDEX READY dataset=${dataset} path=${parquet_path}"
}

launch_lane() {
  local lane_name="$1"
  local script_path="$2"
  local lane_cuda_visible_devices="$3"
  shift 3
  nohup env DATE_TAG="$DATE_TAG" LIMIT="$LIMIT" SAVE_DIR="$SAVE_DIR" EMBED_BASE_URL="$EMBED_BASE_URL" OPENAI_API_KEY="$OPENAI_API_KEY_VALUE" CUDA_VISIBLE_DEVICES="$lane_cuda_visible_devices" "$@" bash "$script_path" >/dev/null 2>&1 &
  log "LANE LAUNCHED name=${lane_name} pid=$! cuda_visible_devices=${lane_cuda_visible_devices} script=${script_path}"
}

build_summary() {
  "$PY" "$SUMMARY_PY" --date-tag "$DATE_TAG" --limit "$LIMIT"
}

: > "$LOG"
printf '%s\n' "$$" > "$PID_FILE"
echo "RUNNING boot" > "$STATUS"
log "Launch started date_tag=${DATE_TAG} limit=${LIMIT}"

healthcheck_models "local8043" "$NQ_LLM_BASE_URL"
healthcheck_chat "local8043" "$NQ_LLM_BASE_URL" "$NQ_LLM_REQUEST_NAME"
healthcheck_models "remote8002" "$POPQA_LLM_BASE_URL"
healthcheck_chat "remote8002" "$POPQA_LLM_BASE_URL" "$POPQA_LLM_REQUEST_NAME"
healthcheck_models "standby8003" "$STANDBY_LLM_BASE_URL"
healthcheck_chat "standby8003" "$STANDBY_LLM_BASE_URL" "$STANDBY_LLM_REQUEST_NAME"
check_local_embedding
check_index_ready "nq"
check_index_ready "popqa"
build_summary

echo "RUNNING launch_nq" > "$STATUS"
launch_lane "nq" "$NQ_SCRIPT" "$NQ_CUDA_VISIBLE_DEVICES" LLM_BASE_URL="$NQ_LLM_BASE_URL" LLM_REQUEST_NAME="$NQ_LLM_REQUEST_NAME" CE_DEVICE="$NQ_CE_DEVICE"
echo "RUNNING launch_popqa" > "$STATUS"
launch_lane "popqa" "$POPQA_SCRIPT" "$POPQA_CUDA_VISIBLE_DEVICES" LLM_BASE_URL="$POPQA_LLM_BASE_URL" LLM_REQUEST_NAME="$POPQA_LLM_REQUEST_NAME" CE_DEVICE="$POPQA_CE_DEVICE"

echo "DONE" > "$STATUS"
build_summary
log "Launch finished; monitor lane status files under run_logs/"
