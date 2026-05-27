# DAEC-DAPG Phase 2.5 Dense-Grounding Mechanism Test

- Dataset: `2wikimultihopqa`
- Backend: `api`
- Phi transform: `linear_clip`
- Phi scales: `[1.0]`
- Rows: `4000`

| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate | Objective |
|---|---:|---:|---:|---:|---:|
| 2:scale=1:demand_source_union | 765 | 0.7000 | 0.4170 | 0.7200 | 0.0004 |
| 2:scale=1:dense_demand_union_cosine | 765 | 0.7235 | 0.4523 | 0.7106 | 2.0508 |
| 2:scale=1:dense_query_cosine | 765 | 0.7771 | 0.5542 | 0.6892 | 1.8091 |
| 2:scale=1:query_level_ppr | 765 | 0.7346 | 0.4810 | 0.7061 | 0.0004 |
| 4:scale=1:demand_source_union | 235 | 0.5277 | 0.0128 | 0.5779 | 0.0004 |
| 4:scale=1:dense_demand_union_cosine | 235 | 0.5436 | 0.0213 | 0.5651 | 2.1009 |
| 4:scale=1:dense_query_cosine | 235 | 0.5500 | 0.0213 | 0.5600 | 1.6905 |
| 4:scale=1:query_level_ppr | 235 | 0.5330 | 0.0213 | 0.5736 | 0.0003 |

## Sum vs Noisy-OR

| Hop:Scale | Rows | Selection Diff Rate | Selection Jaccard | Δ Recall | Δ Complete | Δ Noise |
|---|---:|---:|---:|---:|---:|---:|
