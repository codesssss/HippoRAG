# DAEC-DAPG Phase 2.5 Dense-Grounding Mechanism Test

- Dataset: `2wikimultihopqa`
- Backend: `api`
- Phi transform: `exp`
- Phi scales: `[1.0, 1000.0, 10000.0, 50000.0]`
- Rows: `1600`

| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate | Objective |
|---|---:|---:|---:|---:|---:|
| 2:scale=1:mixed_source_absorption | 77 | 0.6364 | 0.3247 | 0.7455 | 0.0004 |
| 2:scale=1:multi_channel_noisy_or | 77 | 0.4481 | 0.1818 | 0.8208 | 0.0003 |
| 2:scale=1:multi_channel_sum | 77 | 0.4481 | 0.1818 | 0.8208 | 0.0003 |
| 2:scale=1:query_level_ppr | 77 | 0.7208 | 0.4416 | 0.7117 | 0.0004 |
| 2:scale=1000:mixed_source_absorption | 77 | 0.6364 | 0.3247 | 0.7455 | 0.3821 |
| 2:scale=1000:multi_channel_noisy_or | 77 | 0.4416 | 0.1948 | 0.8234 | 0.2698 |
| 2:scale=1000:multi_channel_sum | 77 | 0.4351 | 0.1818 | 0.8260 | 0.3085 |
| 2:scale=1000:query_level_ppr | 77 | 0.7208 | 0.4416 | 0.7117 | 0.3768 |
| 2:scale=10000:mixed_source_absorption | 77 | 0.6364 | 0.3247 | 0.7455 | 2.6720 |
| 2:scale=10000:multi_channel_noisy_or | 77 | 0.3377 | 0.1169 | 0.8649 | 0.9373 |
| 2:scale=10000:multi_channel_sum | 77 | 0.2922 | 0.1039 | 0.8831 | 2.2191 |
| 2:scale=10000:query_level_ppr | 77 | 0.7208 | 0.4416 | 0.7117 | 2.6341 |
| 2:scale=50000:mixed_source_absorption | 77 | 0.6364 | 0.3247 | 0.7455 | 4.8325 |
| 2:scale=50000:multi_channel_noisy_or | 77 | 0.1883 | 0.0649 | 0.9240 | 1.0000 |
| 2:scale=50000:multi_channel_sum | 77 | 0.0909 | 0.0260 | 0.9636 | 4.6059 |
| 2:scale=50000:query_level_ppr | 77 | 0.7208 | 0.4416 | 0.7117 | 4.8049 |
| 4:scale=1:mixed_source_absorption | 23 | 0.5000 | 0.0000 | 0.6000 | 0.0004 |
| 4:scale=1:multi_channel_noisy_or | 23 | 0.4457 | 0.0000 | 0.6435 | 0.0002 |
| 4:scale=1:multi_channel_sum | 23 | 0.4457 | 0.0000 | 0.6435 | 0.0002 |
| 4:scale=1:query_level_ppr | 23 | 0.5217 | 0.0000 | 0.5826 | 0.0003 |
| 4:scale=1000:mixed_source_absorption | 23 | 0.5000 | 0.0000 | 0.6000 | 0.3526 |
| 4:scale=1000:multi_channel_noisy_or | 23 | 0.4457 | 0.0000 | 0.6435 | 0.2148 |
| 4:scale=1000:multi_channel_sum | 23 | 0.4457 | 0.0000 | 0.6435 | 0.2349 |
| 4:scale=1000:query_level_ppr | 23 | 0.5217 | 0.0000 | 0.5826 | 0.3038 |
| 4:scale=10000:mixed_source_absorption | 23 | 0.5000 | 0.0000 | 0.6000 | 2.4772 |
| 4:scale=10000:multi_channel_noisy_or | 23 | 0.4022 | 0.0000 | 0.6783 | 0.9044 |
| 4:scale=10000:multi_channel_sum | 23 | 0.3370 | 0.0000 | 0.7304 | 1.8046 |
| 4:scale=10000:query_level_ppr | 23 | 0.5217 | 0.0000 | 0.5826 | 2.2686 |
| 4:scale=50000:mixed_source_absorption | 23 | 0.5000 | 0.0000 | 0.6000 | 4.7453 |
| 4:scale=50000:multi_channel_noisy_or | 23 | 0.1087 | 0.0000 | 0.9130 | 1.0000 |
| 4:scale=50000:multi_channel_sum | 23 | 0.0109 | 0.0000 | 0.9913 | 4.3540 |
| 4:scale=50000:query_level_ppr | 23 | 0.5217 | 0.0000 | 0.5826 | 4.6766 |

## Sum vs Noisy-OR

| Hop:Scale | Rows | Selection Diff Rate | Selection Jaccard | Δ Recall | Δ Complete | Δ Noise |
|---|---:|---:|---:|---:|---:|---:|
| 2:scale=1 | 77 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2:scale=1000 | 77 | 0.2468 | 0.9116 | 0.0065 | 0.0130 | -0.0026 |
| 2:scale=10000 | 77 | 0.6234 | 0.7286 | 0.0455 | 0.0130 | -0.0182 |
| 2:scale=50000 | 77 | 0.7922 | 0.6851 | 0.0974 | 0.0390 | -0.0396 |
| 4:scale=1 | 23 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4:scale=1000 | 23 | 0.0435 | 0.9855 | 0.0000 | 0.0000 | 0.0000 |
| 4:scale=10000 | 23 | 0.6087 | 0.7660 | 0.0652 | 0.0000 | -0.0522 |
| 4:scale=50000 | 23 | 0.7391 | 0.6708 | 0.0978 | 0.0000 | -0.0783 |
