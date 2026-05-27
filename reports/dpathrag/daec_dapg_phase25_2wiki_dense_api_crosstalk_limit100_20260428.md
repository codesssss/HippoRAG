# DAEC-DAPG Phase 2.5 Dense-Grounding Mechanism Test

- Dataset: `2wikimultihopqa`
- Backend: `api`
- Phi transform: `linear_clip`
- Phi scales: `[1.0]`
- Rows: `600`

| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate | Objective |
|---|---:|---:|---:|---:|---:|
| 2:scale=1:channel_cross_talk | 77 | 0.6818 | 0.3636 | 0.7273 | 0.0004 |
| 2:scale=1:demand_source_union | 77 | 0.6883 | 0.3896 | 0.7247 | 0.0005 |
| 2:scale=1:mixed_source_absorption | 77 | 0.6039 | 0.2987 | 0.7584 | 0.0004 |
| 2:scale=1:multi_channel_noisy_or | 77 | 0.4870 | 0.2338 | 0.8052 | 0.0003 |
| 2:scale=1:multi_channel_sum | 77 | 0.4870 | 0.2338 | 0.8052 | 0.0003 |
| 2:scale=1:query_level_ppr | 77 | 0.7208 | 0.4416 | 0.7117 | 0.0004 |
| 4:scale=1:channel_cross_talk | 23 | 0.5109 | 0.0000 | 0.5913 | 0.0003 |
| 4:scale=1:demand_source_union | 23 | 0.5217 | 0.0000 | 0.5826 | 0.0004 |
| 4:scale=1:mixed_source_absorption | 23 | 0.4891 | 0.0000 | 0.6087 | 0.0004 |
| 4:scale=1:multi_channel_noisy_or | 23 | 0.4783 | 0.0000 | 0.6174 | 0.0003 |
| 4:scale=1:multi_channel_sum | 23 | 0.4783 | 0.0000 | 0.6174 | 0.0003 |
| 4:scale=1:query_level_ppr | 23 | 0.5326 | 0.0000 | 0.5739 | 0.0004 |

## Sum vs Noisy-OR

| Hop:Scale | Rows | Selection Diff Rate | Selection Jaccard | Δ Recall | Δ Complete | Δ Noise |
|---|---:|---:|---:|---:|---:|---:|
| 2:scale=1 | 77 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4:scale=1 | 23 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
