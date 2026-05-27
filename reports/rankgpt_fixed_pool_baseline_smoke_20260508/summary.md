# RankGPT Fixed-Pool Baseline

This report evaluates RankGPT-style no-thinking listwise selection over the fixed PropRAG pool100. It does not retrieve new documents and does not run the reader.

## Setup

- Datasets: `['2wikimultihopqa', 'hotpotqa', 'musique']`
- Limit: `5`
- Variants: `['rank5_no_think', 'select5_no_think']`
- Run LLM: `True`
- LLM base URLs: `['http://localhost:8041/v1', 'http://localhost:8042/v1', 'http://localhost:8043/v1']`
- Workers: `6`
- All prompts use `/no_think`: `True`

## Selector Support Metrics

| Dataset | Method | Q | Support R@5 | Support Complete@5 | Parse ok | selected count |
|---|---|---:|---:|---:|---:|---:|
| 2Wiki | dbec_selective | 5 | 85.0% | 60.0% |  |  |
| 2Wiki | rankgpt_rank5_no_think | 5 | 85.0% | 60.0% | 100.0% | 5.00 |
| 2Wiki | rankgpt_select5_no_think | 5 | 85.0% | 60.0% | 100.0% | 5.00 |
| 2Wiki | setr_faithful | 5 | 90.0% | 80.0% |  |  |
| 2Wiki | source_order | 5 | 90.0% | 80.0% |  |  |
| HotpotQA | dbec_selective | 5 | 100.0% | 100.0% |  |  |
| HotpotQA | rankgpt_rank5_no_think | 5 | 100.0% | 100.0% | 100.0% | 5.00 |
| HotpotQA | rankgpt_select5_no_think | 5 | 90.0% | 80.0% | 100.0% | 5.00 |
| HotpotQA | setr_faithful | 5 | 100.0% | 100.0% |  |  |
| HotpotQA | source_order | 5 | 100.0% | 100.0% |  |  |
| MuSiQue | dbec_selective | 5 | 90.0% | 80.0% |  |  |
| MuSiQue | rankgpt_rank5_no_think | 5 | 73.3% | 60.0% | 100.0% | 5.00 |
| MuSiQue | rankgpt_select5_no_think | 5 | 83.3% | 60.0% | 100.0% | 5.00 |
| MuSiQue | setr_faithful | 5 | 83.3% | 60.0% |  |  |
| MuSiQue | source_order | 5 | 63.3% | 40.0% |  |  |

## Files

- Per-query rows: `reports/rankgpt_fixed_pool_baseline_smoke_20260508/selector_rows.csv`
- Method summary: `reports/rankgpt_fixed_pool_baseline_smoke_20260508/method_summary.csv`
- Prompt outputs: `reports/rankgpt_fixed_pool_baseline_smoke_20260508/prompt_outputs.jsonl`
- MuSiQue missing rows: `reports/rankgpt_fixed_pool_baseline_smoke_20260508/musique_missing_rows.csv`
- Full summary: `reports/rankgpt_fixed_pool_baseline_smoke_20260508/summary.json`

