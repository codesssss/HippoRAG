#!/bin/bash
set -euo pipefail

cd /mnt/nvme/code/HippoRAG
export OPENAI_API_KEY=EMPTY
export PYTHONPATH=.

ROOT="/mnt/nvme/code/HippoRAG/run_logs"
QUEUE_LOG="$ROOT/setwise_exactproj_queue_20260401.log"
STATUS="$ROOT/setwise_exactproj_queue_20260401.status"
SUMMARY_JSON="$ROOT/setwise_exactproj_queue_20260401_summary.json"
mkdir -p "$ROOT"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"
}

run_eval() {
  local dataset="$1"
  local limit="$2"
  local factor="$3"
  local output_json="$4"
  local step_name="${dataset} limit=${limit} factor=${factor}"
  echo "RUNNING: ${step_name}" > "$STATUS"
  log "START ${step_name}"
  .venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
    --dataset "${dataset}" \
    --limit "${limit}" \
    --save_dir outputs_step0_general \
    --max_retry_attempts 12 \
    --setwise_selector bridge_beam \
    --setwise_score_mode set_closure \
    --setwise_pool_k 100 \
    --setwise_anchor_count 2 \
    --setwise_reserve_top_m 3 \
    --setwise_non_anchor_title_dedup true \
    --setwise_beam_width 4 \
    --setwise_beam_expand_per_state 4 \
    --setwise_beam_projected_shortlist_factor "${factor}" \
    --output_json "${output_json}" >> "$QUEUE_LOG" 2>&1
  log "DONE ${step_name}"
}

choose_best_factor() {
  local dataset="$1"
  local factor1_json="$2"
  local factor3_json="$3"
  .venv-hipporag/bin/python - <<PY
import json
from pathlib import Path

factor1_path = Path(${factor1_json@Q})
factor3_path = Path(${factor3_json@Q})

def score(path: Path):
    data = json.loads(path.read_text())
    qa = data["setwise_selector_qa"]
    return float(qa["selector_EM"]), float(qa["selector_F1"])

score1 = score(factor1_path)
score3 = score(factor3_path)
best = 3 if score3 > score1 else 1
print(best)
PY
}

append_summary() {
  local dataset="$1"
  local phase="$2"
  local output_json="$3"
  local factor="$4"
  .venv-hipporag/bin/python - <<PY
import json
from pathlib import Path

summary_path = Path(${SUMMARY_JSON@Q})
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
else:
    summary = {"runs": []}

data = json.loads(Path(${output_json@Q}).read_text())
qa = data["setwise_selector_qa"]
summary["runs"].append({
    "dataset": ${dataset@Q},
    "phase": ${phase@Q},
    "factor": int(${factor}),
    "output_json": ${output_json@Q},
    "selector_EM": float(qa["selector_EM"]),
    "selector_F1": float(qa["selector_F1"]),
    "baseline_EM": float(qa["baseline_EM"]),
    "baseline_F1": float(qa["baseline_F1"]),
    "EM_delta": float(qa["EM_delta"]),
    "F1_delta": float(qa["F1_delta"]),
})
summary_path.write_text(json.dumps(summary, indent=2))
PY
}

trap 'code=$?; log "FAILED exit=${code} while running: $(cat "$STATUS" 2>/dev/null || echo unknown)"; exit $code' ERR

echo "RUNNING: boot" > "$STATUS"
log "Queue started"

datasets=("2wikimultihopqa" "musique" "hotpotqa")
declare -A best_factor_by_dataset

for dataset in "${datasets[@]}"; do
  smoke_factor1_json="outputs_step0_general_${dataset}/eval_reports/setwise_bridge_beam_set_closure_exactonly_smoke40_legacy_reserve3_dedup.json"
  smoke_factor3_json="outputs_step0_general_${dataset}/eval_reports/setwise_bridge_beam_set_closure_exactproj3_smoke40_legacy_reserve3_dedup.json"

  run_eval "${dataset}" 40 1 "${smoke_factor1_json}"
  append_summary "${dataset}" "smoke40" "${smoke_factor1_json}" 1

  run_eval "${dataset}" 40 3 "${smoke_factor3_json}"
  append_summary "${dataset}" "smoke40" "${smoke_factor3_json}" 3

  best_factor="$(choose_best_factor "${dataset}" "${smoke_factor1_json}" "${smoke_factor3_json}")"
  best_factor_by_dataset["${dataset}"]="${best_factor}"
  log "SELECT ${dataset} best_factor=${best_factor}"
done

for dataset in "${datasets[@]}"; do
  factor="${best_factor_by_dataset[$dataset]}"
  tag="exactonly"
  if [[ "${factor}" == "3" ]]; then
    tag="exactproj3"
  fi
  full_json="outputs_step0_general_${dataset}/eval_reports/setwise_bridge_beam_set_closure_${tag}_100_legacy_reserve3_dedup.json"
  run_eval "${dataset}" 100 "${factor}" "${full_json}"
  append_summary "${dataset}" "full100" "${full_json}" "${factor}"
done

echo "DONE" > "$STATUS"
log "Queue finished"
