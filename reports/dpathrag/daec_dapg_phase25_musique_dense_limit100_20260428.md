# DAEC-DAPG Phase 2.5 Dense-Grounding Mechanism Test

- Dataset: `musique`
- Backend: `tfidf_dense`
- Rows: `400`

| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate | Objective |
|---|---:|---:|---:|---:|---:|
| 2:mixed_source_absorption | 48 | 0.4896 | 0.1667 | 0.8125 | 0.0007 |
| 2:multi_channel_noisy_or | 48 | 0.4896 | 0.1667 | 0.8125 | 0.0007 |
| 2:multi_channel_sum | 48 | 0.4896 | 0.1667 | 0.8125 | 0.0007 |
| 2:query_level_ppr | 48 | 0.4896 | 0.1667 | 0.8125 | 0.0007 |
| 3:mixed_source_absorption | 30 | 0.3889 | 0.0667 | 0.7733 | 0.0006 |
| 3:multi_channel_noisy_or | 30 | 0.3889 | 0.0667 | 0.7733 | 0.0006 |
| 3:multi_channel_sum | 30 | 0.3889 | 0.0667 | 0.7733 | 0.0006 |
| 3:query_level_ppr | 30 | 0.3889 | 0.0667 | 0.7733 | 0.0006 |
| 4:mixed_source_absorption | 22 | 0.2576 | 0.0000 | 0.8000 | 0.0006 |
| 4:multi_channel_noisy_or | 22 | 0.2576 | 0.0000 | 0.8000 | 0.0006 |
| 4:multi_channel_sum | 22 | 0.2576 | 0.0000 | 0.8000 | 0.0006 |
| 4:query_level_ppr | 22 | 0.2576 | 0.0000 | 0.8000 | 0.0006 |
