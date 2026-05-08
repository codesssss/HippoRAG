# RankGPT Fixed-Pool Baseline

This report evaluates RankGPT-style no-thinking listwise selection over the fixed PropRAG pool100. It does not retrieve new documents and does not run the reader.

## Setup

- Datasets: `['2wikimultihopqa', 'hotpotqa', 'musique']`
- Limit: `1000`
- Variants: `['rank5_no_think', 'select5_no_think']`
- Passage id convention: `1-based`
- Run LLM: `True`
- LLM base URLs: `['http://localhost:8041/v1', 'http://localhost:8042/v1', 'http://localhost:8043/v1']`
- Workers: `9`
- All prompts use `/no_think`: `True`

## Selector Support Metrics

| Dataset | Method | Q | Support R@5 | Support Complete@5 | Parse ok | selected count |
|---|---|---:|---:|---:|---:|---:|
| 2Wiki | dbec_selective | 1000 | 94.1% | 86.3% |  |  |
| 2Wiki | rankgpt_rank5_no_think | 1000 | 89.6% | 78.7% | 100.0% | 4.97 |
| 2Wiki | rankgpt_select5_no_think | 1000 | 88.5% | 75.2% | 100.0% | 5.00 |
| 2Wiki | setr_faithful | 1000 | 94.2% | 85.5% |  |  |
| 2Wiki | source_order | 1000 | 90.3% | 77.2% |  |  |
| HotpotQA | dbec_selective | 1000 | 96.2% | 93.0% |  |  |
| HotpotQA | rankgpt_rank5_no_think | 1000 | 86.6% | 76.2% | 100.0% | 4.99 |
| HotpotQA | rankgpt_select5_no_think | 1000 | 85.0% | 73.3% | 100.0% | 5.00 |
| HotpotQA | setr_faithful | 1000 | 97.4% | 94.9% |  |  |
| HotpotQA | source_order | 1000 | 95.2% | 90.7% |  |  |
| MuSiQue | dbec_selective | 1000 | 77.4% | 52.7% |  |  |
| MuSiQue | rankgpt_rank5_no_think | 1000 | 65.8% | 37.4% | 100.0% | 4.99 |
| MuSiQue | rankgpt_select5_no_think | 1000 | 65.2% | 36.6% | 100.0% | 5.00 |
| MuSiQue | setr_faithful | 1000 | 78.4% | 54.2% |  |  |
| MuSiQue | source_order | 1000 | 73.8% | 48.3% |  |  |

## MuSiQue Source-Visible Missing-Gold Slice

| Method | Missing titles | Q | Hit@5 | New@5 |
|---|---:|---:|---:|---:|
| dbec_selective | 96 | 72 | 0.0% | 0.0% |
| rankgpt_rank5_no_think | 96 | 72 | 20.8% | 20.8% |
| rankgpt_select5_no_think | 96 | 72 | 25.0% | 25.0% |
| setr_faithful | 96 | 72 | 0.0% | 0.0% |
| source_order | 96 | 72 | 1.0% | 0.0% |

## Files

- Per-query rows: `reports/rankgpt_fixed_pool_baseline_full1000_1based_20260508/selector_rows.csv`
- Method summary: `reports/rankgpt_fixed_pool_baseline_full1000_1based_20260508/method_summary.csv`
- Prompt outputs: `reports/rankgpt_fixed_pool_baseline_full1000_1based_20260508/prompt_outputs.jsonl`
- MuSiQue missing rows: `reports/rankgpt_fixed_pool_baseline_full1000_1based_20260508/musique_missing_rows.csv`
- Full summary: `reports/rankgpt_fixed_pool_baseline_full1000_1based_20260508/summary.json`
