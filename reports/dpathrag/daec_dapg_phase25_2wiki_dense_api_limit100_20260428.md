# DAEC-DAPG Phase 2.5 Dense-Grounding Mechanism Test

- Dataset: `2wikimultihopqa`
- Backend: `api`
- Rows: `400`

| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate | Objective |
|---|---:|---:|---:|---:|---:|
| 2:mixed_source_absorption | 77 | 0.6234 | 0.3117 | 0.7506 | 0.0004 |
| 2:multi_channel_noisy_or | 77 | 0.4740 | 0.2078 | 0.8104 | 0.0003 |
| 2:multi_channel_sum | 77 | 0.4740 | 0.2078 | 0.8104 | 0.0003 |
| 2:query_level_ppr | 77 | 0.7208 | 0.4545 | 0.7117 | 0.0004 |
| 4:mixed_source_absorption | 23 | 0.4891 | 0.0000 | 0.6087 | 0.0004 |
| 4:multi_channel_noisy_or | 23 | 0.4565 | 0.0000 | 0.6348 | 0.0003 |
| 4:multi_channel_sum | 23 | 0.4565 | 0.0000 | 0.6348 | 0.0003 |
| 4:query_level_ppr | 23 | 0.5217 | 0.0000 | 0.5826 | 0.0003 |

## Sum vs Noisy-OR

| Hop | Rows | Selection Diff Rate | Selection Jaccard | Δ Recall | Δ Complete | Δ Noise |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 77 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 23 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
