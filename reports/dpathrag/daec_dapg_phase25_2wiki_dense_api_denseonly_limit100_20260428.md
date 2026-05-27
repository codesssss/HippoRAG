# DAEC-DAPG Phase 2.5 Dense-Grounding Mechanism Test

- Dataset: `2wikimultihopqa`
- Backend: `api`
- Phi transform: `linear_clip`
- Phi scales: `[1.0]`
- Rows: `800`

| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate | Objective |
|---|---:|---:|---:|---:|---:|
| 2:scale=1:channel_cross_talk | 77 | 0.6883 | 0.4026 | 0.7247 | 0.0004 |
| 2:scale=1:demand_source_union | 77 | 0.6883 | 0.4026 | 0.7247 | 0.0004 |
| 2:scale=1:dense_demand_union_cosine | 77 | 0.6948 | 0.4026 | 0.7221 | 1.9962 |
| 2:scale=1:dense_query_cosine | 77 | 0.7403 | 0.4805 | 0.7039 | 1.7600 |
| 2:scale=1:mixed_source_absorption | 77 | 0.5909 | 0.2857 | 0.7636 | 0.0004 |
| 2:scale=1:multi_channel_noisy_or | 77 | 0.4740 | 0.2208 | 0.8104 | 0.0003 |
| 2:scale=1:multi_channel_sum | 77 | 0.4740 | 0.2208 | 0.8104 | 0.0003 |
| 2:scale=1:query_level_ppr | 77 | 0.7143 | 0.4416 | 0.7143 | 0.0004 |
| 4:scale=1:channel_cross_talk | 23 | 0.5109 | 0.0000 | 0.5913 | 0.0003 |
| 4:scale=1:demand_source_union | 23 | 0.5217 | 0.0000 | 0.5826 | 0.0004 |
| 4:scale=1:dense_demand_union_cosine | 23 | 0.5435 | 0.0000 | 0.5652 | 2.0895 |
| 4:scale=1:dense_query_cosine | 23 | 0.5435 | 0.0000 | 0.5652 | 1.6775 |
| 4:scale=1:mixed_source_absorption | 23 | 0.5000 | 0.0000 | 0.6000 | 0.0004 |
| 4:scale=1:multi_channel_noisy_or | 23 | 0.4239 | 0.0000 | 0.6609 | 0.0002 |
| 4:scale=1:multi_channel_sum | 23 | 0.4239 | 0.0000 | 0.6609 | 0.0002 |
| 4:scale=1:query_level_ppr | 23 | 0.5326 | 0.0000 | 0.5739 | 0.0003 |

## Sum vs Noisy-OR

| Hop:Scale | Rows | Selection Diff Rate | Selection Jaccard | Δ Recall | Δ Complete | Δ Noise |
|---|---:|---:|---:|---:|---:|---:|
| 2:scale=1 | 77 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4:scale=1 | 23 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
