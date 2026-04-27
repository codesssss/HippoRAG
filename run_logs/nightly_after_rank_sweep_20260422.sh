#!/usr/bin/env bash
set -euo pipefail

cd /mnt/nvme/code/HippoRAG

WATCH_PID="${1:-3724219}"
LOG_PREFIX="[nightly-rank]"

echo "${LOG_PREFIX} $(date '+%F %T') waiting for rank sweep pid=${WATCH_PID}"
while kill -0 "${WATCH_PID}" 2>/dev/null; do
  sleep 60
done
echo "${LOG_PREFIX} $(date '+%F %T') rank sweep finished; selecting global best rank_weight"

BEST_WEIGHT="$(
  .venv-hipporag/bin/python - <<'PY'
import glob
import json
import os
import re
import sys
from collections import defaultdict

datasets = ["2wikimultihopqa", "hotpotqa", "musique"]
rows = []
for dataset in datasets:
    files = glob.glob(
        f"outputs_step0_general_nvembed_{dataset}/eval_reports/"
        "dtc_embed_nvembed_rankw*_pilot100_anchor2_8043.json"
    )
    if len(files) < 5:
        raise SystemExit(f"missing pilot rank files for {dataset}: found {len(files)}")
    for path in files:
        match = re.search(r"rankw([^_]+)_", os.path.basename(path))
        if not match:
            continue
        weight = float(match.group(1).replace("p", "."))
        data = json.load(open(path))
        metrics = data["setwise_selector_qa"]
        rows.append((dataset, weight, float(metrics["F1_delta"]), float(metrics["EM_delta"])))

by_weight = defaultdict(list)
for dataset, weight, f1_delta, em_delta in rows:
    by_weight[weight].append((dataset, f1_delta, em_delta))

valid = {w: vals for w, vals in by_weight.items() if len({v[0] for v in vals}) == len(datasets)}
if not valid:
    raise SystemExit("no rank_weight has all datasets")

def score(item):
    weight, vals = item
    avg_f1 = sum(v[1] for v in vals) / len(vals)
    avg_em = sum(v[2] for v in vals) / len(vals)
    min_f1 = min(v[1] for v in vals)
    return (avg_f1, min_f1, avg_em, -weight)

best_weight, best_vals = max(valid.items(), key=score)

summary_path = "run_logs/dtc_rank_sweep_best_20260422.tsv"
with open(summary_path, "w") as out:
    out.write("rank_weight\tdataset\tF1_delta\tEM_delta\n")
    for weight in sorted(valid):
        for dataset, f1_delta, em_delta in sorted(valid[weight]):
            out.write(f"{weight}\t{dataset}\t{f1_delta:.6f}\t{em_delta:.6f}\n")
    out.write(f"\nBEST\t{best_weight}\n")

print(best_weight)
PY
)"

echo "${LOG_PREFIX} $(date '+%F %T') selected rank_weight=${BEST_WEIGHT}"
TAG="${BEST_WEIGHT//./p}"

run_dataset() {
  local dataset="$1"
  local outdir="outputs_step0_general_nvembed_${dataset}"
  local output_json="${outdir}/eval_reports/dtc_embed_nvembed_rankw${TAG}_limit1000_anchor2_8043.json"
  mkdir -p "${outdir}/eval_reports"
  if [[ -s "${output_json}" ]]; then
    echo "${LOG_PREFIX} $(date '+%F %T') SKIP ${dataset} existing ${output_json}"
    return 0
  fi
  echo "${LOG_PREFIX} $(date '+%F %T') START ${dataset} limit1000 rank_weight=${BEST_WEIGHT} -> ${output_json}"
  PYTHONUNBUFFERED=1 .venv-hipporag/bin/python -u scripts/eval_causal_qwen3.py \
    --dataset "${dataset}" \
    --limit 1000 \
    --save_dir outputs_step0_general_nvembed \
    --llm_name qwen3-8b \
    --llm_request_name qwen3-8b-train \
    --llm_base_url http://localhost:8043/v1 \
    --embedding_name VLLM/nvidia/NV-Embed-v2 \
    --embedding_base_url http://localhost:8019/v1/embeddings \
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
    --dtc_rank_weight "${BEST_WEIGHT}" \
    --dtc_anchor_bonus_weight 0.10 \
    --dtc_dependency_bonus_weight 0.10 \
    --dtc_max_completion_tokens 512 \
    --output_json "${output_json}"
  echo "${LOG_PREFIX} $(date '+%F %T') DONE ${dataset} limit1000 rank_weight=${BEST_WEIGHT}"
}

run_dataset 2wikimultihopqa
run_dataset hotpotqa
run_dataset musique

.venv-hipporag/bin/python - <<'PY'
import glob
import json
import os
import re

rows = []
for path in sorted(glob.glob("outputs_step0_general_nvembed_*/eval_reports/dtc_embed_nvembed_rankw*_limit1000_anchor2_8043.json")):
    data = json.load(open(path))
    metrics = data["setwise_selector_qa"]
    rows.append((
        data["dataset"],
        metrics["baseline_EM"],
        metrics["selector_EM"],
        metrics["EM_delta"],
        metrics["baseline_F1"],
        metrics["selector_F1"],
        metrics["F1_delta"],
        os.path.basename(path),
    ))
with open("run_logs/dtc_rank_best1000_summary_20260422.tsv", "w") as out:
    out.write("dataset\tbaseEM\tselEM\tdEM\tbaseF1\tselF1\tdF1\tfile\n")
    for row in rows:
        out.write("%s\t%.6f\t%.6f\t%+.6f\t%.6f\t%.6f\t%+.6f\t%s\n" % row)
PY

echo "${LOG_PREFIX} $(date '+%F %T') ALL DONE"
