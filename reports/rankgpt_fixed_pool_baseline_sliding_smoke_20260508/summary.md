# RankGPT Fixed-Pool Baseline

This report evaluates RankGPT-style no-thinking listwise selection over the fixed PropRAG pool100. It does not retrieve new documents and does not run the reader.

## Setup

- Datasets: `['2wikimultihopqa', 'hotpotqa', 'musique']`
- Limit: `2`
- Variants: `['sliding20_step10_no_think']`
- Passage id convention: `1-based`
- Run LLM: `True`
- LLM base URLs: `['http://localhost:8041/v1', 'http://localhost:8042/v1', 'http://localhost:8043/v1']`
- Workers: `6`
- All prompts use `/no_think`: `True`

## Selector Support Metrics

| Dataset | Method | Q | Support R@5 | Support Complete@5 | Parse ok | selected count |
|---|---|---:|---:|---:|---:|---:|
| 2Wiki | dbec_selective | 2 | 100.0% | 100.0% |  |  |
| 2Wiki | rankgpt_sliding20_step10_no_think | 2 | 100.0% | 100.0% | 100.0% | 5.00 |
| 2Wiki | setr_faithful | 2 | 100.0% | 100.0% |  |  |
| 2Wiki | source_order | 2 | 100.0% | 100.0% |  |  |
| HotpotQA | dbec_selective | 2 | 100.0% | 100.0% |  |  |
| HotpotQA | rankgpt_sliding20_step10_no_think | 2 | 100.0% | 100.0% | 100.0% | 5.00 |
| HotpotQA | setr_faithful | 2 | 100.0% | 100.0% |  |  |
| HotpotQA | source_order | 2 | 100.0% | 100.0% |  |  |
| MuSiQue | dbec_selective | 2 | 100.0% | 100.0% |  |  |
| MuSiQue | rankgpt_sliding20_step10_no_think | 2 | 83.3% | 50.0% | 100.0% | 5.00 |
| MuSiQue | setr_faithful | 2 | 83.3% | 50.0% |  |  |
| MuSiQue | source_order | 2 | 83.3% | 50.0% |  |  |

## Files

- Per-query rows: `reports/rankgpt_fixed_pool_baseline_sliding_smoke_20260508/selector_rows.csv`
- Method summary: `reports/rankgpt_fixed_pool_baseline_sliding_smoke_20260508/method_summary.csv`
- Prompt outputs: `reports/rankgpt_fixed_pool_baseline_sliding_smoke_20260508/prompt_outputs.jsonl`
- MuSiQue missing rows: `reports/rankgpt_fixed_pool_baseline_sliding_smoke_20260508/musique_missing_rows.csv`
- Full summary: `reports/rankgpt_fixed_pool_baseline_sliding_smoke_20260508/summary.json`
