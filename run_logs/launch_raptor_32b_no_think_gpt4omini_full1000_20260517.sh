#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/mnt/nvme/code/HippoRAG}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-hipporag/bin/python}"
RAPTOR_ROOT="${RAPTOR_ROOT:-/mnt/nvme/code/RAPTOR}"
RUN_TAG="${RUN_TAG:-raptor_32b_no_think_gpt4omini_full1000_20260517}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/run_logs/${RUN_TAG}}"
SECRET_ENV="${SECRET_ENV:-${ROOT_DIR}/.secrets/gpt4omini_yunwu.env}"

DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa musique nq_rear popqa}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
CORPUS_LIMIT="${CORPUS_LIMIT:-0}"
POOL_K="${POOL_K:-200}"
QA_TOP_K="${QA_TOP_K:-5}"

GRAPH_LLM_NAME="${GRAPH_LLM_NAME:-qwen3-32b-judge}"
QWEN32_URL_A="${QWEN32_URL_A:-http://localhost:8045/v1}"
QWEN32_URL_B="${QWEN32_URL_B:-http://localhost:8046/v1}"
INDEX_LLM_NAME="${INDEX_LLM_NAME:-qwen3-32b-judge}"
EMBEDDING_NAME="${EMBEDDING_NAME:-VLLM/nvidia/NV-Embed-v2}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

RAPTOR_NUM_LAYERS="${RAPTOR_NUM_LAYERS:-5}"
RAPTOR_REDUCTION_DIMENSION="${RAPTOR_REDUCTION_DIMENSION:-10}"
RAPTOR_CLUSTER_THRESHOLD="${RAPTOR_CLUSTER_THRESHOLD:-0.1}"
RAPTOR_MAX_LENGTH_IN_CLUSTER="${RAPTOR_MAX_LENGTH_IN_CLUSTER:-3500}"
RAPTOR_SUMMARIZATION_LENGTH="${RAPTOR_SUMMARIZATION_LENGTH:-120}"
RAPTOR_SUMMARY_WORKERS="${RAPTOR_SUMMARY_WORKERS:-2}"
RAPTOR_TOP_NODES="${RAPTOR_TOP_NODES:-80}"
RAPTOR_LEAVES_PER_SUMMARY="${RAPTOR_LEAVES_PER_SUMMARY:-10}"
RAPTOR_TREE_CACHE_ROOT="${RAPTOR_TREE_CACHE_ROOT:-${ROOT_DIR}/run_logs/raptor_tree_cache_32b_no_think_nvembed}"

READER_LLM_NAME="${READER_LLM_NAME:-gpt-4o-mini}"
READER_LLM_BASE_URL="${READER_LLM_BASE_URL:-https://yunwu.ai/v1}"
READER_MAX_NEW_TOKENS="${READER_MAX_NEW_TOKENS:-none}"

mkdir -p \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/status" \
  "${OUT_ROOT}/pools/raptor" \
  "${OUT_ROOT}/reader_inputs/raptor" \
  "${OUT_ROOT}/reader_qa/raptor" \
  "${OUT_ROOT}/reader_runtime/raptor"

if [[ -f "${SECRET_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRET_ENV}"
  set +a
fi

log_msg() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/logs/launcher.log"
}

write_status() {
  local name="$1"
  shift
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee "${OUT_ROOT}/status/${name}.status"
}

check_endpoint() {
  local name="$1"
  local url="$2"
  if curl -fsS --max-time 10 "${url}" >/dev/null 2>&1; then
    log_msg "OK endpoint ${name}: ${url}"
    return 0
  fi
  log_msg "FAILED endpoint ${name}: ${url}"
  return 1
}

display_dataset_name() {
  case "$1" in
    nq|nq_rear|natural_questions) printf 'nq\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

evidence_dataset_name() {
  case "$1" in
    nq|natural_questions) printf 'nq_rear\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

qwen_url_for_lane() {
  case "$1" in
    lane_b) printf '%s\n' "${QWEN32_URL_B}" ;;
    *) printf '%s\n' "${QWEN32_URL_A}" ;;
  esac
}

pool_path() {
  local dataset="$1"
  local display
  display="$(display_dataset_name "${dataset}")"
  printf '%s/pools/raptor/%s_raptor_pool%s_limit%s.json\n' "${OUT_ROOT}" "${display}" "${POOL_K}" "${MAX_QUERIES}"
}

reader_input_path() {
  local dataset="$1"
  local display
  display="$(display_dataset_name "${dataset}")"
  printf '%s/reader_inputs/raptor/%s_raptor_reader_input.json\n' "${OUT_ROOT}" "${display}"
}

reader_report_path() {
  local dataset="$1"
  local display
  display="$(display_dataset_name "${dataset}")"
  printf '%s/reader_qa/raptor/%s_raptor_qwen32b_no_think_gpt4omini_reader.json\n' "${OUT_ROOT}" "${display}"
}

reader_report_md_path() {
  local dataset="$1"
  local display
  display="$(display_dataset_name "${dataset}")"
  printf '%s/reader_qa/raptor/%s_raptor_qwen32b_no_think_gpt4omini_reader.md\n' "${OUT_ROOT}" "${display}"
}

run_logged() {
  local desc="$1"
  local status_name="$2"
  local log_path="$3"
  shift 3
  write_status "${status_name}" "START ${desc} log=${log_path}"
  log_msg "START ${desc}"
  (
    cd "${ROOT_DIR}" || exit 1
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=false
    "$@"
  ) > "${log_path}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    write_status "${status_name}" "DONE ${desc}"
    log_msg "DONE ${desc}"
  else
    write_status "${status_name}" "FAILED ${desc} code=${code} log=${log_path}"
    log_msg "FAILED ${desc} code=${code} log=${log_path}"
  fi
  return "${code}"
}

export_raptor_dataset() {
  local dataset="$1"
  local lane="$2"
  local display output_json log_path qwen_url
  dataset="$(evidence_dataset_name "${dataset}")"
  display="$(display_dataset_name "${dataset}")"
  output_json="$(pool_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/export_raptor_${display}.log"
  qwen_url="$(qwen_url_for_lane "${lane}")"

  if [[ -s "${output_json}" ]]; then
    write_status "export_raptor_${display}" "SKIP existing output=${output_json}"
    return 0
  fi

  run_logged "raptor export dataset=${display} lane=${lane} qwen=${qwen_url}" \
    "export_raptor_${display}" "${log_path}" \
    env OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" "${PYTHON_BIN}" scripts/export_raptor_pool.py \
      --dataset "${dataset}" \
      --limit "${MAX_QUERIES}" \
      --corpus-limit "${CORPUS_LIMIT}" \
      --pool_k "${POOL_K}" \
      --raptor_root "${RAPTOR_ROOT}" \
      --tree_cache_root "${RAPTOR_TREE_CACHE_ROOT}" \
      --llm_name "${GRAPH_LLM_NAME}" \
      --llm_base_url "${qwen_url}" \
      --index_llm_name "${INDEX_LLM_NAME}" \
      --embedding_name "${EMBEDDING_NAME}" \
      --embedding_base_url "${EMBEDDING_BASE_URL}" \
      --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
      --num_layers "${RAPTOR_NUM_LAYERS}" \
      --reduction_dimension "${RAPTOR_REDUCTION_DIMENSION}" \
      --cluster_threshold "${RAPTOR_CLUSTER_THRESHOLD}" \
      --max_length_in_cluster "${RAPTOR_MAX_LENGTH_IN_CLUSTER}" \
      --summarization_length "${RAPTOR_SUMMARIZATION_LENGTH}" \
      --summary_workers "${RAPTOR_SUMMARY_WORKERS}" \
      --top_nodes "${RAPTOR_TOP_NODES}" \
      --leaves_per_summary "${RAPTOR_LEAVES_PER_SUMMARY}" \
      --output_json "${output_json}"
}

wrap_raptor_dataset() {
  local dataset="$1"
  local display source_json input_json log_path
  dataset="$(evidence_dataset_name "${dataset}")"
  display="$(display_dataset_name "${dataset}")"
  source_json="$(pool_path "${dataset}")"
  input_json="$(reader_input_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/wrap_raptor_${display}.log"

  if [[ -s "${input_json}" ]]; then
    write_status "wrap_raptor_${display}" "SKIP existing input=${input_json}"
    return 0
  fi

  run_logged "wrap raptor dataset=${display}" "wrap_raptor_${display}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" scripts/make_external_pool_reader_inputs.py \
      --pool-json "${source_json}" \
      --method-name "raptor_qwen32b_no_think" \
      --output-json "${input_json}"
}

run_raptor_reader_dataset() {
  local dataset="$1"
  local display input_json output_json output_md log_path
  dataset="$(evidence_dataset_name "${dataset}")"
  display="$(display_dataset_name "${dataset}")"
  input_json="$(reader_input_path "${dataset}")"
  output_json="$(reader_report_path "${dataset}")"
  output_md="$(reader_report_md_path "${dataset}")"
  log_path="${OUT_ROOT}/logs/reader_raptor_${display}.log"

  if [[ -s "${output_json}" ]]; then
    write_status "reader_raptor_${display}" "SKIP existing output=${output_json}"
    return 0
  fi

  run_logged "reader raptor dataset=${display}" "reader_raptor_${display}" "${log_path}" \
    env PYTHONPATH=. "${PYTHON_BIN}" evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py \
      --input-files "${input_json}" \
      --method-name "raptor_qwen32b_no_think_top${QA_TOP_K}" \
      --max-queries "${MAX_QUERIES}" \
      --qa-top-k "${QA_TOP_K}" \
      --doc-source external_pool_topk_docs \
      --save-dir "${OUT_ROOT}/reader_runtime/raptor/${display}" \
      --llm-name "${READER_LLM_NAME}" \
      --llm-base-url "${READER_LLM_BASE_URL}" \
      --max-new-tokens "${READER_MAX_NEW_TOKENS}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --openie-mode online \
      --output-json "${output_json}" \
      --output-md "${output_md}"
}

run_dataset() {
  local dataset="$1"
  local lane="$2"
  local display
  dataset="$(evidence_dataset_name "${dataset}")"
  display="$(display_dataset_name "${dataset}")"
  write_status "lane_${lane}_${display}" "START dataset=${display}"
  export_raptor_dataset "${dataset}" "${lane}" || return 1
  wrap_raptor_dataset "${dataset}" || return 1
  run_raptor_reader_dataset "${dataset}" || return 1
  write_status "lane_${lane}_${display}" "DONE dataset=${display}"
}

run_sequence() {
  local lane="$1"
  shift
  local dataset
  for dataset in "$@"; do
    run_dataset "${dataset}" "${lane}" || return 1
  done
}

write_summary() {
  env OUT_ROOT="${OUT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
import os

out = Path(os.environ["OUT_ROOT"])
rows = []
for path in sorted((out / "reader_qa" / "raptor").glob("*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for dataset in payload.get("datasets", []) or []:
        label = dataset.get("dataset")
        for method_name, method_payload in (dataset.get("methods") or {}).items():
            metrics = method_payload.get("metrics") or {}
            rows.append({
                "dataset": label,
                "method": method_name,
                "count": metrics.get("count"),
                "Recall@5": metrics.get("Recall@5") or metrics.get("r5"),
                "Recall@20": metrics.get("Recall@20"),
                "ExactMatch": metrics.get("ExactMatch"),
                "F1": metrics.get("F1"),
                "mean_reader_docs": metrics.get("mean_reader_docs"),
                "source_json": str(path),
            })
summary = {"rows": rows}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lines = ["| Dataset | Method | Count | R@5 | R@20 | EM | F1 | Reader docs |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
for row in rows:
    def fmt(value):
        return "" if value is None else f"{float(value):.4f}"
    lines.append(
        f"| {row['dataset']} | {row['method']} | {int(row.get('count') or 0)} | "
        f"{fmt(row.get('Recall@5'))} | {fmt(row.get('Recall@20'))} | "
        f"{fmt(row.get('ExactMatch'))} | {fmt(row.get('F1'))} | {fmt(row.get('mean_reader_docs'))} |"
    )
(out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

main() {
  write_status "launcher" "START run_tag=${RUN_TAG} datasets=${DATASETS}"
  log_msg "BEGIN ${RUN_TAG}"
  log_msg "Protocol: RAPTOR summaries=${GRAPH_LLM_NAME} no_think; embeddings=${EMBEDDING_NAME}; reader=${READER_LLM_NAME}; datasets=${DATASETS}"

  local preflight=0
  [[ -d "${RAPTOR_ROOT}" ]] || { log_msg "FAILED missing RAPTOR_ROOT=${RAPTOR_ROOT}"; preflight=1; }
  check_endpoint "qwen32_a" "${QWEN32_URL_A}/models" || preflight=1
  check_endpoint "qwen32_b" "${QWEN32_URL_B}/models" || preflight=1
  check_endpoint "embedding" "http://localhost:8019/v1/models" || preflight=1
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    log_msg "FAILED missing OPENAI_API_KEY after loading ${SECRET_ENV}"
    preflight=1
  fi
  if [[ "${preflight}" -ne 0 ]]; then
    write_status "launcher" "FAILED preflight"
    return 1
  fi

  local lane_a=()
  local lane_b=()
  local idx=0
  local dataset
  for dataset in ${DATASETS}; do
    if (( idx % 2 == 0 )); then
      lane_a+=("${dataset}")
    else
      lane_b+=("${dataset}")
    fi
    idx=$((idx + 1))
  done

  local pid_a=0
  local pid_b=0
  if [[ "${#lane_a[@]}" -gt 0 ]]; then
    run_sequence lane_a "${lane_a[@]}" &
    pid_a="$!"
    log_msg "LAUNCHED RAPTOR lane_a pid=${pid_a} datasets=${lane_a[*]}"
  fi
  if [[ "${#lane_b[@]}" -gt 0 ]]; then
    run_sequence lane_b "${lane_b[@]}" &
    pid_b="$!"
    log_msg "LAUNCHED RAPTOR lane_b pid=${pid_b} datasets=${lane_b[*]}"
  fi

  local rc=0
  if [[ "${pid_a}" != "0" ]]; then
    wait "${pid_a}" || rc=1
  fi
  if [[ "${pid_b}" != "0" ]]; then
    wait "${pid_b}" || rc=1
  fi

  if [[ "${rc}" -eq 0 ]]; then
    write_summary | tee -a "${OUT_ROOT}/logs/launcher.log"
    write_status "launcher" "DONE run_tag=${RUN_TAG}"
    log_msg "DONE ${RUN_TAG}"
  else
    write_status "launcher" "FAILED run_tag=${RUN_TAG} rc=${rc}"
    log_msg "FAILED ${RUN_TAG} rc=${rc}"
  fi
  return "${rc}"
}

main "$@"
