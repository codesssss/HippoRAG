#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"
METHOD="evidence_transition_graphragv4_fact_witnessed_sto"
INDEX_ROOT="${ROOT_DIR}/run_logs/evidence_transition_graphragv4_clean_mainline_multi_anchor_strict_musique100_20260511"
RUN_TAG="etv4_clean_vs_no_multi_anchor_full1000_20260511"
GATE_ROOT="${ROOT_DIR}/run_logs/${RUN_TAG}"
CLEAN_ROOT="${ROOT_DIR}/run_logs/etv4_clean_mainline_full1000_reuse_gpt4omini_20260511"
ABL_ROOT="${ROOT_DIR}/run_logs/etv4_no_multi_anchor_precision_full1000_reuse_gpt4omini_20260511"

mkdir -p "${GATE_ROOT}/logs" "${GATE_ROOT}/status" "${GATE_ROOT}/jobs" "${GATE_ROOT}/reports"

setup_reuse_index() {
  local out_root="$1"
  local dataset="$2"
  local src="${INDEX_ROOT}/${dataset}/index"
  local dst_parent="${out_root}/${dataset}"
  local dst="${dst_parent}/index"
  if [[ ! -d "${src}" ]]; then
    echo "[ERROR] missing reusable index: ${src}" >&2
    exit 1
  fi
  mkdir -p "${dst_parent}"
  if [[ -e "${dst}" && ! -L "${dst}" ]]; then
    echo "[ERROR] refusing to replace non-symlink index path: ${dst}" >&2
    exit 1
  fi
  ln -sfn "${src}" "${dst}"
}

report_path() {
  local out_root="$1"
  local dataset="$2"
  printf '%s/%s/reports/%s_%s_retrieval.json' "${out_root}" "${dataset}" "${dataset}" "${METHOD}"
}

short_policy() {
  case "$1" in
    clean_mainline) printf 'clean' ;;
    no_multi_anchor_precision) printf 'nomulti' ;;
    *) printf '%s' "$1" | tr -cd '[:alnum:]_' ;;
  esac
}

launch_job() {
  local policy="$1"
  local dataset="$2"
  local out_root="$3"
  local short
  short="$(short_policy "${policy}")"
  local session="etv4_f1000_${short}_${dataset}_20260511"
  local log_path="${GATE_ROOT}/logs/${short}_${dataset}.log"
  local status_path="${GATE_ROOT}/status/${short}_${dataset}.status"
  local job_script="${GATE_ROOT}/jobs/${short}_${dataset}.sh"
  local report
  report="$(report_path "${out_root}" "${dataset}")"

  setup_reuse_index "${out_root}" "${dataset}"

  if [[ -f "${report}" ]]; then
    echo "[SKIP] ${session} report already exists: ${report}"
    printf 'done_existing\t%s\t%s\n' "$(date -Is)" "${report}" > "${status_path}"
    return 0
  fi

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[SKIP] ${session} already exists"
    return 0
  fi

  cat > "${job_script}" <<JOB
#!/usr/bin/env bash
set -uo pipefail
{
  echo "[START] session=${session} policy=${policy} dataset=${dataset} start=\$(date -Is)"
  echo "[OUTPUT] ${out_root}"
  echo "[REPORT] ${report}"
  cd "${ROOT_DIR}"
  PYTHONPATH=. "${PYTHON_BIN}" \\
    evidence_transition_graphragv4_fact_witnessed_sto/run_fresh_e2e.py \\
    --datasets "${dataset}" \\
    --output-root "${out_root}" \\
    --reuse-current-fresh-index \\
    --max-queries 1000 \\
    --llm-name gpt-4o-mini \\
    --llm-base-url https://yunwu.ai/v1 \\
    --embedding-name VLLM/nvidia/NV-Embed-v2 \\
    --embedding-base-url http://localhost:8019/v1/embeddings \\
    --readout-policy "${policy}" \\
    --candidate-pool-k 200 \\
    --top-k 5
  cmd_rc=\$?
  echo "[COMMAND_EXIT] session=${session} rc=\${cmd_rc} end=\$(date -Is)"
  exit "\${cmd_rc}"
} 2>&1 | tee "${log_path}"
rc=\${PIPESTATUS[0]}
if [[ "\${rc}" -eq 0 && -f "${report}" ]]; then
  printf 'done\t%s\t%s\n' "\$(date -Is)" "${report}" > "${status_path}"
else
  printf 'failed\t%s\trc=%s\treport_exists=%s\n' "\$(date -Is)" "\${rc}" "\$(test -f "${report}" && echo yes || echo no)" > "${status_path}"
fi
exit "\${rc}"
JOB
  chmod +x "${job_script}"

  printf 'launched\t%s\t%s\n' "$(date -Is)" "${session}" > "${status_path}"
  tmux new-session -d -s "${session}" "${job_script}"
  tmux set-option -t "${session}" remain-on-exit on >/dev/null
  echo "[LAUNCHED] ${session} log=${log_path}"
}

launch_job "clean_mainline" "musique" "${CLEAN_ROOT}"
launch_job "clean_mainline" "hotpotqa" "${CLEAN_ROOT}"
launch_job "no_multi_anchor_precision" "2wikimultihopqa" "${ABL_ROOT}"
launch_job "no_multi_anchor_precision" "musique" "${ABL_ROOT}"
launch_job "no_multi_anchor_precision" "hotpotqa" "${ABL_ROOT}"
