# DAEC-DAPG Phase 2.5 Dense-Grounding Mechanism Test

- Dataset: `musique`
- Backend: `api`
- Rows: `400`

| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate | Objective |
|---|---:|---:|---:|---:|---:|
| 2:mixed_source_absorption | 48 | 0.6875 | 0.3750 | 0.7375 | 0.0002 |
| 2:multi_channel_noisy_or | 48 | 0.6458 | 0.3542 | 0.7542 | 0.0002 |
| 2:multi_channel_sum | 48 | 0.6458 | 0.3542 | 0.7542 | 0.0002 |
| 2:query_level_ppr | 48 | 0.6771 | 0.3750 | 0.7417 | 0.0002 |
| 3:mixed_source_absorption | 30 | 0.5778 | 0.2000 | 0.6600 | 0.0003 |
| 3:multi_channel_noisy_or | 30 | 0.5056 | 0.1000 | 0.7000 | 0.0002 |
| 3:multi_channel_sum | 30 | 0.5056 | 0.1000 | 0.7000 | 0.0002 |
| 3:query_level_ppr | 30 | 0.6000 | 0.2333 | 0.6467 | 0.0002 |
| 4:mixed_source_absorption | 22 | 0.3371 | 0.0000 | 0.7364 | 0.0003 |
| 4:multi_channel_noisy_or | 22 | 0.3712 | 0.0000 | 0.7091 | 0.0002 |
| 4:multi_channel_sum | 22 | 0.3712 | 0.0000 | 0.7091 | 0.0002 |
| 4:query_level_ppr | 22 | 0.4280 | 0.0000 | 0.6636 | 0.0002 |

## Sum vs Noisy-OR

| Hop | Rows | Selection Diff Rate | Selection Jaccard | Δ Recall | Δ Complete | Δ Noise |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 48 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 30 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 22 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
