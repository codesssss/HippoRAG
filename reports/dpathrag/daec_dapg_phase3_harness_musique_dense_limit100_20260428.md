# DAEC-DAPG Phase 3 Full Retrieval Harness

- Dataset: `musique`
- Frontier: `run_logs/dense_pool_exports_full1000_20260424/musique_dense_pool100.json`
- Frontier source: `supplied_dense_pool_frontier_schema_check`

| Hop | Rows | Support Recall | Support Complete | Noise Rate |
|---|---:|---:|---:|---:|
| 2 | 48 | 0.3854 | 0.1250 | 0.8542 |
| 3 | 30 | 0.2722 | 0.0000 | 0.8400 |
| 4 | 22 | 0.2462 | 0.0000 | 0.8091 |

This is a schema-compatible harness. Final paper runs must use a full-corpus index-graph frontier, not a fixed-pool rerank.
