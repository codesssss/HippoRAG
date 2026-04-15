#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="$ROOT_DIR/.venv-hipporag/bin/python"

cd "$ROOT_DIR"

"$PYTHON_BIN" scripts/run_swap_value_smoke.py \
  --dataset musique \
  --baseline_report outputs_step0_general_musique/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260409smoke.json \
  --candidate_report outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260409smoke.json \
  --output_json run_logs/musique_swap_value_smoke_20260411.json \
  --output_md run_logs/musique_swap_value_smoke_20260411.md

"$PYTHON_BIN" scripts/run_swap_value_smoke.py \
  --dataset 2wikimultihopqa \
  --baseline_report outputs_step0_general_2wikimultihopqa/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260407.json \
  --candidate_report outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260407.json \
  --output_json run_logs/2wiki_swap_value_smoke_20260411.json \
  --output_md run_logs/2wiki_swap_value_smoke_20260411.md
