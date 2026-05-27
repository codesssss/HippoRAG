# AREC-RAG Day 1 Closure Smoke

- Status: `completed`
- Dataset: `musique`
- Rows: `100`
- Verifier: `cross-encoder/nli-deberta-v3-base`

| Metric | Value |
|---|---:|
| closure_score_initial | 0.8342 |
| oracle_closure_score_initial | 2.1344 |
| initial_support_complete | 0.2900 |
| active_obligation_count | 2.7600 |

## By Hop

| Hop | Rows | Closure | Support Complete | Active Obligations |
|---|---:|---:|---:|---:|
| 2 | 48 | 0.8072 | 0.4583 | 2.3542 |
| 3 | 30 | 0.9589 | 0.2000 | 3.0000 |
| 4 | 22 | 0.7231 | 0.0455 | 3.3182 |
