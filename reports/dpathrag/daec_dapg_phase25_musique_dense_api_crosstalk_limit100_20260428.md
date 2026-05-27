# DAEC-DAPG Phase 2.5 Dense-Grounding Mechanism Test

- Dataset: `musique`
- Backend: `api`
- Phi transform: `linear_clip`
- Phi scales: `[1.0]`
- Rows: `600`

| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate | Objective |
|---|---:|---:|---:|---:|---:|
| 2:scale=1:channel_cross_talk | 48 | 0.7083 | 0.4167 | 0.7292 | 0.0002 |
| 2:scale=1:demand_source_union | 48 | 0.6771 | 0.3542 | 0.7417 | 0.0003 |
| 2:scale=1:mixed_source_absorption | 48 | 0.6979 | 0.3958 | 0.7333 | 0.0003 |
| 2:scale=1:multi_channel_noisy_or | 48 | 0.6562 | 0.3750 | 0.7458 | 0.0002 |
| 2:scale=1:multi_channel_sum | 48 | 0.6562 | 0.3750 | 0.7458 | 0.0002 |
| 2:scale=1:query_level_ppr | 48 | 0.7083 | 0.4167 | 0.7292 | 0.0003 |
| 3:scale=1:channel_cross_talk | 30 | 0.5611 | 0.2000 | 0.6667 | 0.0002 |
| 3:scale=1:demand_source_union | 30 | 0.5667 | 0.1667 | 0.6667 | 0.0003 |
| 3:scale=1:mixed_source_absorption | 30 | 0.5389 | 0.1333 | 0.6800 | 0.0003 |
| 3:scale=1:multi_channel_noisy_or | 30 | 0.4833 | 0.1000 | 0.7133 | 0.0002 |
| 3:scale=1:multi_channel_sum | 30 | 0.4833 | 0.1000 | 0.7133 | 0.0002 |
| 3:scale=1:query_level_ppr | 30 | 0.6056 | 0.2333 | 0.6400 | 0.0002 |
| 4:scale=1:channel_cross_talk | 22 | 0.4432 | 0.0455 | 0.6545 | 0.0002 |
| 4:scale=1:demand_source_union | 22 | 0.4545 | 0.0455 | 0.6455 | 0.0003 |
| 4:scale=1:mixed_source_absorption | 22 | 0.3977 | 0.0455 | 0.6909 | 0.0003 |
| 4:scale=1:multi_channel_noisy_or | 22 | 0.3447 | 0.0000 | 0.7273 | 0.0002 |
| 4:scale=1:multi_channel_sum | 22 | 0.3447 | 0.0000 | 0.7273 | 0.0002 |
| 4:scale=1:query_level_ppr | 22 | 0.4432 | 0.0455 | 0.6545 | 0.0002 |

## Sum vs Noisy-OR

| Hop:Scale | Rows | Selection Diff Rate | Selection Jaccard | Δ Recall | Δ Complete | Δ Noise |
|---|---:|---:|---:|---:|---:|---:|
| 2:scale=1 | 48 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3:scale=1 | 30 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4:scale=1 | 22 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
