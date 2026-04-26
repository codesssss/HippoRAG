# D-PathRAG Reader Failure Analysis

- Input: `data/dpathrag/reader_baselines/2wiki_proprag_rank_topk_eval200.jsonl`
- Predictions: `reports/dpathrag/reader_proprag_rank_topk_eval200.predictions.jsonl`
- Rows: 200

| Metric | Value |
|---|---:|
| answer_in_context_rate | 0.8000 |
| answer_failure_rate | 0.5150 |
| answer_in_context_but_fail_rate | 0.3600 |
| support_complete_but_fail_rate | 0.3600 |
| incomplete_but_answer_in_context_rate | 0.1050 |
| avg_f1_answer_in_context | 0.6141 |
| avg_f1_answer_absent | 0.2567 |
