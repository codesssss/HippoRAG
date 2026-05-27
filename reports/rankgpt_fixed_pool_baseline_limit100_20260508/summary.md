# RankGPT Fixed-Pool Baseline

This report evaluates RankGPT-style no-thinking listwise selection over the fixed PropRAG pool100. It does not retrieve new documents and does not run the reader.

## Setup

- Datasets: `['2wikimultihopqa', 'hotpotqa', 'musique']`
- Limit: `100`
- Variants: `['rank5_no_think', 'select5_no_think']`
- Run LLM: `True`
- LLM base URLs: `['http://localhost:8041/v1', 'http://localhost:8042/v1', 'http://localhost:8043/v1']`
- Workers: `9`
- All prompts use `/no_think`: `True`

## Selector Support Metrics

| Dataset | Method | Q | Support R@5 | Support Complete@5 | Parse ok | selected count |
|---|---|---:|---:|---:|---:|---:|
| 2Wiki | dbec_selective | 100 | 94.0% | 89.0% |  |  |
| 2Wiki | rankgpt_rank5_no_think | 100 | 81.8% | 60.0% | 100.0% | 4.96 |
| 2Wiki | rankgpt_select5_no_think | 100 | 85.8% | 71.0% | 100.0% | 5.00 |
| 2Wiki | setr_faithful | 100 | 96.0% | 90.0% |  |  |
| 2Wiki | source_order | 100 | 93.5% | 85.0% |  |  |
| HotpotQA | dbec_selective | 100 | 95.0% | 90.0% |  |  |
| HotpotQA | rankgpt_rank5_no_think | 100 | 80.0% | 63.0% | 100.0% | 4.96 |
| HotpotQA | rankgpt_select5_no_think | 100 | 77.5% | 61.0% | 100.0% | 5.00 |
| HotpotQA | setr_faithful | 100 | 97.0% | 94.0% |  |  |
| HotpotQA | source_order | 100 | 94.5% | 89.0% |  |  |
| MuSiQue | dbec_selective | 100 | 75.1% | 47.0% |  |  |
| MuSiQue | rankgpt_rank5_no_think | 100 | 57.2% | 29.0% | 100.0% | 4.97 |
| MuSiQue | rankgpt_select5_no_think | 100 | 59.8% | 35.0% | 100.0% | 5.00 |
| MuSiQue | setr_faithful | 100 | 76.3% | 47.0% |  |  |
| MuSiQue | source_order | 100 | 69.5% | 40.0% |  |  |

## MuSiQue Source-Visible Missing-Gold Slice

| Method | Missing titles | Q | Hit@5 | New@5 |
|---|---:|---:|---:|---:|
| dbec_selective | 16 | 11 | 0.0% | 0.0% |
| rankgpt_rank5_no_think | 16 | 11 | 18.8% | 18.8% |
| rankgpt_select5_no_think | 16 | 11 | 18.8% | 18.8% |
| setr_faithful | 16 | 11 | 0.0% | 0.0% |
| source_order | 16 | 11 | 0.0% | 0.0% |

## Files

- Per-query rows: `reports/rankgpt_fixed_pool_baseline_limit100_20260508/selector_rows.csv`
- Method summary: `reports/rankgpt_fixed_pool_baseline_limit100_20260508/method_summary.csv`
- Prompt outputs: `reports/rankgpt_fixed_pool_baseline_limit100_20260508/prompt_outputs.jsonl`
- MuSiQue missing rows: `reports/rankgpt_fixed_pool_baseline_limit100_20260508/musique_missing_rows.csv`
- Full summary: `reports/rankgpt_fixed_pool_baseline_limit100_20260508/summary.json`

