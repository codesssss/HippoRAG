#!/usr/bin/env bash
set -u

ROOT_DIR="/mnt/nvme/code/HippoRAG"
LIMIT100_STATUS="${ROOT_DIR}/run_logs/neocorrag_aligned_k3_limit100_20260430/launcher.status"
FULL_DIR="${ROOT_DIR}/run_logs/neocorrag_aligned_k3_full1000_20260430"
FULL_LAUNCHER="${ROOT_DIR}/run_logs/launch_neocorrag_aligned_k3_full1000_20260430.sh"
WATCH_LOG="${FULL_DIR}/watch_after_limit100.log"

mkdir -p "${FULL_DIR}"

echo "[WATCH_START] waiting for NEO K3 limit100 completion $(date -Is)" | tee -a "${WATCH_LOG}"

while true; do
  if [[ -s "${LIMIT100_STATUS}" ]] && grep -q '^\[DONE\]' "${LIMIT100_STATUS}"; then
    echo "[WATCH_DONE] limit100 complete; launching full1000 $(date -Is)" | tee -a "${WATCH_LOG}"
    break
  fi
  if [[ -s "${LIMIT100_STATUS}" ]] && grep -q '^\[FAILED\]' "${LIMIT100_STATUS}"; then
    echo "[WATCH_ABORT] limit100 failed; not launching full1000 $(date -Is)" | tee -a "${WATCH_LOG}"
    exit 1
  fi
  echo "[WATCH_WAIT] $(date -Is) status=$(cat "${LIMIT100_STATUS}" 2>/dev/null || echo missing)" >> "${WATCH_LOG}"
  sleep 120
done

if pgrep -af 'launch_neocorrag_aligned_k3_full1000_20260430.sh' >/dev/null; then
  echo "[WATCH_SKIP] full1000 launcher already running $(date -Is)" | tee -a "${WATCH_LOG}"
  exit 0
fi

setsid bash "${FULL_LAUNCHER}" > "${FULL_DIR}/driver.log" 2>&1 < /dev/null &
pid="$!"
echo "[WATCH_LAUNCHED] full1000_pid=${pid} $(date -Is)" | tee -a "${WATCH_LOG}"
