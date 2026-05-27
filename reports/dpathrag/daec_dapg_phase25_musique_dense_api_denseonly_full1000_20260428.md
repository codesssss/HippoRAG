# DAEC-DAPG Phase 2.5 Dense-Grounding Mechanism Test

- Dataset: `musique`
- Backend: `api`
- Phi transform: `linear_clip`
- Phi scales: `[1.0]`
- Rows: `4000`

| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate | Objective |
|---|---:|---:|---:|---:|---:|
| 2:scale=1:demand_source_union | 518 | 0.6737 | 0.3764 | 0.7390 | 0.0003 |
| 2:scale=1:dense_demand_union_cosine | 518 | 0.6950 | 0.4189 | 0.7305 | 1.8537 |
| 2:scale=1:dense_query_cosine | 518 | 0.7751 | 0.5695 | 0.6985 | 1.6775 |
| 2:scale=1:query_level_ppr | 518 | 0.7336 | 0.4942 | 0.7151 | 0.0003 |
| 3:scale=1:demand_source_union | 316 | 0.6097 | 0.1804 | 0.6418 | 0.0003 |
| 3:scale=1:dense_demand_union_cosine | 316 | 0.6350 | 0.1741 | 0.6266 | 1.9146 |
| 3:scale=1:dense_query_cosine | 316 | 0.6772 | 0.2658 | 0.6019 | 1.5504 |
| 3:scale=1:query_level_ppr | 316 | 0.6456 | 0.2373 | 0.6209 | 0.0003 |
| 4:scale=1:demand_source_union | 166 | 0.4061 | 0.0241 | 0.6843 | 0.0003 |
| 4:scale=1:dense_demand_union_cosine | 166 | 0.4332 | 0.0301 | 0.6627 | 1.8954 |
| 4:scale=1:dense_query_cosine | 166 | 0.4418 | 0.0422 | 0.6554 | 1.4751 |
| 4:scale=1:query_level_ppr | 166 | 0.3991 | 0.0301 | 0.6892 | 0.0002 |

## Sum vs Noisy-OR

| Hop:Scale | Rows | Selection Diff Rate | Selection Jaccard | Δ Recall | Δ Complete | Δ Noise |
|---|---:|---:|---:|---:|---:|---:|
