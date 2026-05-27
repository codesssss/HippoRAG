# DAEC-DAPG Phase 2.5 Dense-Grounding Mechanism Test

- Dataset: `musique`
- Backend: `api`
- Phi transform: `exp`
- Phi scales: `[1.0, 1000.0, 10000.0]`
- Rows: `12`

| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate | Objective |
|---|---:|---:|---:|---:|---:|
| 2:scale=1:mixed_source_absorption | 1 | 1.0000 | 1.0000 | 0.8000 | 0.0003 |
| 2:scale=1:multi_channel_noisy_or | 1 | 1.0000 | 1.0000 | 0.8000 | 0.0003 |
| 2:scale=1:multi_channel_sum | 1 | 1.0000 | 1.0000 | 0.8000 | 0.0003 |
| 2:scale=1:query_level_ppr | 1 | 1.0000 | 1.0000 | 0.8000 | 0.0003 |
| 2:scale=1000:mixed_source_absorption | 1 | 1.0000 | 1.0000 | 0.8000 | 0.2938 |
| 2:scale=1000:multi_channel_noisy_or | 1 | 1.0000 | 1.0000 | 0.8000 | 0.2348 |
| 2:scale=1000:multi_channel_sum | 1 | 1.0000 | 1.0000 | 0.8000 | 0.2603 |
| 2:scale=1000:query_level_ppr | 1 | 1.0000 | 1.0000 | 0.8000 | 0.2868 |
| 2:scale=10000:mixed_source_absorption | 1 | 1.0000 | 1.0000 | 0.8000 | 2.2662 |
| 2:scale=10000:multi_channel_noisy_or | 1 | 1.0000 | 1.0000 | 0.8000 | 0.9302 |
| 2:scale=10000:multi_channel_sum | 1 | 1.0000 | 1.0000 | 0.8000 | 2.0445 |
| 2:scale=10000:query_level_ppr | 1 | 1.0000 | 1.0000 | 0.8000 | 2.2267 |

## Sum vs Noisy-OR

| Hop:Scale | Rows | Selection Diff Rate | Selection Jaccard | Δ Recall | Δ Complete | Δ Noise |
|---|---:|---:|---:|---:|---:|---:|
| 2:scale=1 | 1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2:scale=1000 | 1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2:scale=10000 | 1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
