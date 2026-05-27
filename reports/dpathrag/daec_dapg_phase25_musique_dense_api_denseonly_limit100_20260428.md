# DAEC-DAPG Phase 2.5 Dense-Grounding Mechanism Test

- Dataset: `musique`
- Backend: `api`
- Phi transform: `linear_clip`
- Phi scales: `[1.0]`
- Rows: `800`

| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate | Objective |
|---|---:|---:|---:|---:|---:|
| 2:scale=1:channel_cross_talk | 48 | 0.6979 | 0.3958 | 0.7333 | 0.0002 |
| 2:scale=1:demand_source_union | 48 | 0.6875 | 0.3958 | 0.7375 | 0.0003 |
| 2:scale=1:dense_demand_union_cosine | 48 | 0.6875 | 0.3750 | 0.7375 | 1.8934 |
| 2:scale=1:dense_query_cosine | 48 | 0.7292 | 0.4583 | 0.7208 | 1.6837 |
| 2:scale=1:mixed_source_absorption | 48 | 0.6875 | 0.3750 | 0.7375 | 0.0003 |
| 2:scale=1:multi_channel_noisy_or | 48 | 0.6562 | 0.3750 | 0.7458 | 0.0002 |
| 2:scale=1:multi_channel_sum | 48 | 0.6562 | 0.3750 | 0.7458 | 0.0002 |
| 2:scale=1:query_level_ppr | 48 | 0.6979 | 0.3958 | 0.7333 | 0.0003 |
| 3:scale=1:channel_cross_talk | 30 | 0.5333 | 0.1333 | 0.6867 | 0.0002 |
| 3:scale=1:demand_source_union | 30 | 0.5333 | 0.1333 | 0.6867 | 0.0003 |
| 3:scale=1:dense_demand_union_cosine | 30 | 0.5667 | 0.1333 | 0.6667 | 1.9221 |
| 3:scale=1:dense_query_cosine | 30 | 0.6444 | 0.2000 | 0.6200 | 1.5629 |
| 3:scale=1:mixed_source_absorption | 30 | 0.5667 | 0.1667 | 0.6667 | 0.0003 |
| 3:scale=1:multi_channel_noisy_or | 30 | 0.4722 | 0.0667 | 0.7200 | 0.0002 |
| 3:scale=1:multi_channel_sum | 30 | 0.4722 | 0.0667 | 0.7200 | 0.0002 |
| 3:scale=1:query_level_ppr | 30 | 0.6333 | 0.3000 | 0.6267 | 0.0002 |
| 4:scale=1:channel_cross_talk | 22 | 0.4167 | 0.0000 | 0.6727 | 0.0002 |
| 4:scale=1:demand_source_union | 22 | 0.4053 | 0.0000 | 0.6818 | 0.0003 |
| 4:scale=1:dense_demand_union_cosine | 22 | 0.4545 | 0.0455 | 0.6455 | 1.9484 |
| 4:scale=1:dense_query_cosine | 22 | 0.4659 | 0.0455 | 0.6364 | 1.4526 |
| 4:scale=1:mixed_source_absorption | 22 | 0.4053 | 0.0000 | 0.6818 | 0.0003 |
| 4:scale=1:multi_channel_noisy_or | 22 | 0.3712 | 0.0000 | 0.7091 | 0.0002 |
| 4:scale=1:multi_channel_sum | 22 | 0.3712 | 0.0000 | 0.7091 | 0.0002 |
| 4:scale=1:query_level_ppr | 22 | 0.4508 | 0.0000 | 0.6455 | 0.0002 |

## Sum vs Noisy-OR

| Hop:Scale | Rows | Selection Diff Rate | Selection Jaccard | Δ Recall | Δ Complete | Δ Noise |
|---|---:|---:|---:|---:|---:|---:|
| 2:scale=1 | 48 | 0.0208 | 0.9931 | 0.0000 | 0.0000 | 0.0000 |
| 3:scale=1 | 30 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4:scale=1 | 22 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
