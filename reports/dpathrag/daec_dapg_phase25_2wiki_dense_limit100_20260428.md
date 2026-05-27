# DAEC-DAPG Phase 2.5 Dense-Grounding Mechanism Test

- Dataset: `2wikimultihopqa`
- Backend: `tfidf_dense`
- Rows: `400`

| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate | Objective |
|---|---:|---:|---:|---:|---:|
| 2:mixed_source_absorption | 77 | 0.6688 | 0.4026 | 0.7325 | 0.0012 |
| 2:multi_channel_noisy_or | 77 | 0.6688 | 0.4026 | 0.7307 | 0.0012 |
| 2:multi_channel_sum | 77 | 0.6688 | 0.4026 | 0.7325 | 0.0012 |
| 2:query_level_ppr | 77 | 0.6688 | 0.4026 | 0.7325 | 0.0012 |
| 4:mixed_source_absorption | 23 | 0.5109 | 0.0000 | 0.5913 | 0.0014 |
| 4:multi_channel_noisy_or | 23 | 0.5109 | 0.0000 | 0.5913 | 0.0014 |
| 4:multi_channel_sum | 23 | 0.5109 | 0.0000 | 0.5913 | 0.0014 |
| 4:query_level_ppr | 23 | 0.5109 | 0.0000 | 0.5913 | 0.0014 |
