#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/code/HippoRAG"
PYTHON_BIN="$ROOT_DIR/.venv-hipporag/bin/python"
BASE_URL="${SWAP_UTILITY_JUDGE_BASE_URL:-https://api.shenfengwl.fun}"
MODEL_NAME="${SWAP_UTILITY_JUDGE_MODEL:-gpt-5.4}"
REASONING_EFFORT="${SWAP_UTILITY_JUDGE_REASONING_EFFORT:-medium}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required" >&2
  exit 1
fi

cd "$ROOT_DIR"

"$PYTHON_BIN" scripts/run_swap_utility_judge.py \
  --dataset musique \
  --baseline_report outputs_step0_general_musique/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260409smoke.json \
  --candidate_report outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260409smoke.json \
  --oracle_swap_json run_logs/musique_swap_value_smoke_20260411.json \
  --output_json run_logs/musique_swap_utility_judge_20260413.json \
  --output_md run_logs/musique_swap_utility_judge_20260413.md \
  --replace_bottom_n 2 \
  --max_queries 100 \
  --setwise_late_rerank_judge_backend responses \
  --setwise_late_rerank_judge_model "$MODEL_NAME" \
  --setwise_late_rerank_judge_base_url "$BASE_URL" \
  --setwise_late_rerank_judge_reasoning_effort "$REASONING_EFFORT"

"$PYTHON_BIN" scripts/run_swap_utility_judge.py \
  --dataset 2wikimultihopqa \
  --baseline_report outputs_step0_general_2wikimultihopqa/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260407.json \
  --candidate_report outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260407.json \
  --oracle_swap_json run_logs/2wiki_swap_value_smoke_20260411.json \
  --output_json run_logs/2wiki_swap_utility_judge_20260413.json \
  --output_md run_logs/2wiki_swap_utility_judge_20260413.md \
  --replace_bottom_n 2 \
  --max_queries 100 \
  --setwise_late_rerank_judge_backend responses \
  --setwise_late_rerank_judge_model "$MODEL_NAME" \
  --setwise_late_rerank_judge_base_url "$BASE_URL" \
  --setwise_late_rerank_judge_reasoning_effort "$REASONING_EFFORT"

"$PYTHON_BIN" scripts/run_swap_utility_judge.py \
  --dataset hotpotqa \
  --baseline_report outputs_step0_general_hotpotqa/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260407.json \
  --candidate_report outputs_step0_general_hotpotqa/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260407.json \
  --output_json run_logs/hotpotqa_swap_utility_judge_30_20260413.json \
  --output_md run_logs/hotpotqa_swap_utility_judge_30_20260413.md \
  --replace_bottom_n 2 \
  --max_queries 30 \
  --setwise_late_rerank_judge_backend responses \
  --setwise_late_rerank_judge_model "$MODEL_NAME" \
  --setwise_late_rerank_judge_base_url "$BASE_URL" \
  --setwise_late_rerank_judge_reasoning_effort "$REASONING_EFFORT"
