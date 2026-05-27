# DAEC-DAPG Phase 2.5 Dense-Grounding Mechanism Test

- Dataset: `musique`
- Backend: `api`
- Rows: `4`

| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate | Objective |
|---|---:|---:|---:|---:|---:|
| 2:mixed_source_absorption | 1 | 1.0000 | 1.0000 | 0.8000 | 0.0003 |
| 2:multi_channel_noisy_or | 1 | 1.0000 | 1.0000 | 0.8000 | 0.0002 |
| 2:multi_channel_sum | 1 | 1.0000 | 1.0000 | 0.8000 | 0.0002 |
| 2:query_level_ppr | 1 | 1.0000 | 1.0000 | 0.8000 | 0.0003 |

## Sum vs Noisy-OR

| Hop | Rows | Selection Diff Rate | Selection Jaccard | Δ Recall | Δ Complete | Δ Noise |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
