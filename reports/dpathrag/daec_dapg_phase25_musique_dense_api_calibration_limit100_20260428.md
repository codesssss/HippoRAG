# DAEC-DAPG Phase 2.5 Dense-Grounding Mechanism Test

- Dataset: `musique`
- Backend: `api`
- Phi transform: `exp`
- Phi scales: `[1.0, 1000.0, 10000.0, 50000.0]`
- Rows: `1600`

| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate | Objective |
|---|---:|---:|---:|---:|---:|
| 2:scale=1:mixed_source_absorption | 48 | 0.6771 | 0.3542 | 0.7417 | 0.0003 |
| 2:scale=1:multi_channel_noisy_or | 48 | 0.6667 | 0.3750 | 0.7458 | 0.0002 |
| 2:scale=1:multi_channel_sum | 48 | 0.6667 | 0.3750 | 0.7458 | 0.0002 |
| 2:scale=1:query_level_ppr | 48 | 0.6667 | 0.3333 | 0.7458 | 0.0003 |
| 2:scale=1000:mixed_source_absorption | 48 | 0.6771 | 0.3542 | 0.7417 | 0.2580 |
| 2:scale=1000:multi_channel_noisy_or | 48 | 0.6667 | 0.3750 | 0.7458 | 0.1901 |
| 2:scale=1000:multi_channel_sum | 48 | 0.6667 | 0.3750 | 0.7458 | 0.2076 |
| 2:scale=1000:query_level_ppr | 48 | 0.6667 | 0.3333 | 0.7458 | 0.2601 |
| 2:scale=10000:mixed_source_absorption | 48 | 0.6771 | 0.3542 | 0.7417 | 2.0178 |
| 2:scale=10000:multi_channel_noisy_or | 48 | 0.6250 | 0.2917 | 0.7583 | 0.8560 |
| 2:scale=10000:multi_channel_sum | 48 | 0.6458 | 0.3542 | 0.7542 | 1.6209 |
| 2:scale=10000:query_level_ppr | 48 | 0.6667 | 0.3333 | 0.7458 | 2.0333 |
| 2:scale=50000:mixed_source_absorption | 48 | 0.6771 | 0.3542 | 0.7417 | 4.5360 |
| 2:scale=50000:multi_channel_noisy_or | 48 | 0.4688 | 0.2292 | 0.8208 | 0.9997 |
| 2:scale=50000:multi_channel_sum | 48 | 0.3333 | 0.1667 | 0.8750 | 4.0176 |
| 2:scale=50000:query_level_ppr | 48 | 0.6667 | 0.3333 | 0.7458 | 4.5507 |
| 3:scale=1:mixed_source_absorption | 30 | 0.5389 | 0.1333 | 0.6800 | 0.0003 |
| 3:scale=1:multi_channel_noisy_or | 30 | 0.4944 | 0.1000 | 0.7067 | 0.0002 |
| 3:scale=1:multi_channel_sum | 30 | 0.4944 | 0.1000 | 0.7067 | 0.0002 |
| 3:scale=1:query_level_ppr | 30 | 0.6111 | 0.2667 | 0.6400 | 0.0002 |
| 3:scale=1000:mixed_source_absorption | 30 | 0.5389 | 0.1333 | 0.6800 | 0.2790 |
| 3:scale=1000:multi_channel_noisy_or | 30 | 0.4944 | 0.1000 | 0.7067 | 0.1734 |
| 3:scale=1000:multi_channel_sum | 30 | 0.4944 | 0.1000 | 0.7067 | 0.1878 |
| 3:scale=1000:query_level_ppr | 30 | 0.6111 | 0.2667 | 0.6400 | 0.2423 |
| 3:scale=10000:mixed_source_absorption | 30 | 0.5389 | 0.1333 | 0.6800 | 2.1124 |
| 3:scale=10000:multi_channel_noisy_or | 30 | 0.5056 | 0.1000 | 0.7000 | 0.8257 |
| 3:scale=10000:multi_channel_sum | 30 | 0.4500 | 0.1000 | 0.7333 | 1.4913 |
| 3:scale=10000:query_level_ppr | 30 | 0.6111 | 0.2667 | 0.6400 | 1.9318 |
| 3:scale=50000:mixed_source_absorption | 30 | 0.5389 | 0.1333 | 0.6800 | 4.5733 |
| 3:scale=50000:multi_channel_noisy_or | 30 | 0.3944 | 0.1000 | 0.7667 | 0.9991 |
| 3:scale=50000:multi_channel_sum | 30 | 0.2389 | 0.0667 | 0.8600 | 3.8378 |
| 3:scale=50000:query_level_ppr | 30 | 0.6111 | 0.2667 | 0.6400 | 4.4943 |
| 4:scale=1:mixed_source_absorption | 22 | 0.3826 | 0.0000 | 0.7000 | 0.0003 |
| 4:scale=1:multi_channel_noisy_or | 22 | 0.3561 | 0.0000 | 0.7182 | 0.0002 |
| 4:scale=1:multi_channel_sum | 22 | 0.3561 | 0.0000 | 0.7182 | 0.0002 |
| 4:scale=1:query_level_ppr | 22 | 0.4280 | 0.0000 | 0.6636 | 0.0002 |
| 4:scale=1000:mixed_source_absorption | 22 | 0.3826 | 0.0000 | 0.7000 | 0.2988 |
| 4:scale=1000:multi_channel_noisy_or | 22 | 0.3712 | 0.0000 | 0.7091 | 0.1573 |
| 4:scale=1000:multi_channel_sum | 22 | 0.3447 | 0.0000 | 0.7273 | 0.1692 |
| 4:scale=1000:query_level_ppr | 22 | 0.4280 | 0.0000 | 0.6636 | 0.2281 |
| 4:scale=10000:mixed_source_absorption | 22 | 0.3826 | 0.0000 | 0.7000 | 2.2209 |
| 4:scale=10000:multi_channel_noisy_or | 22 | 0.3826 | 0.0000 | 0.7000 | 0.7869 |
| 4:scale=10000:multi_channel_sum | 22 | 0.3220 | 0.0000 | 0.7455 | 1.3408 |
| 4:scale=10000:query_level_ppr | 22 | 0.4280 | 0.0000 | 0.6636 | 1.8392 |
| 4:scale=50000:mixed_source_absorption | 22 | 0.3826 | 0.0000 | 0.7000 | 4.6280 |
| 4:scale=50000:multi_channel_noisy_or | 22 | 0.2424 | 0.0000 | 0.8091 | 0.9976 |
| 4:scale=50000:multi_channel_sum | 22 | 0.1856 | 0.0000 | 0.8545 | 3.5552 |
| 4:scale=50000:query_level_ppr | 22 | 0.4280 | 0.0000 | 0.6636 | 4.4207 |

## Sum vs Noisy-OR

| Hop:Scale | Rows | Selection Diff Rate | Selection Jaccard | Δ Recall | Δ Complete | Δ Noise |
|---|---:|---:|---:|---:|---:|---:|
| 2:scale=1 | 48 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2:scale=1000 | 48 | 0.2083 | 0.9292 | 0.0000 | 0.0000 | 0.0000 |
| 2:scale=10000 | 48 | 0.6875 | 0.7000 | -0.0208 | -0.0625 | 0.0042 |
| 2:scale=50000 | 48 | 0.7917 | 0.6331 | 0.1354 | 0.0625 | -0.0542 |
| 3:scale=1 | 30 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3:scale=1000 | 30 | 0.1000 | 0.9711 | 0.0000 | 0.0000 | 0.0000 |
| 3:scale=10000 | 30 | 0.7333 | 0.6811 | 0.0556 | 0.0000 | -0.0333 |
| 3:scale=50000 | 30 | 0.9333 | 0.4960 | 0.1556 | 0.0333 | -0.0933 |
| 4:scale=1 | 22 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4:scale=1000 | 22 | 0.3636 | 0.8788 | 0.0265 | 0.0000 | -0.0182 |
| 4:scale=10000 | 22 | 0.8182 | 0.6409 | 0.0606 | 0.0000 | -0.0455 |
| 4:scale=50000 | 22 | 0.8636 | 0.5442 | 0.0568 | 0.0000 | -0.0455 |
