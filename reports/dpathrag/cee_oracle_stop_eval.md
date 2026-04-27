# CEE Oracle STOP Evaluation

- Rows: `1000`
- Folds: `5`
- Candidate pool size: `20`

| Variant | Support Recall | Support Complete | Selected Gold | Bridge Recall | Selection Overlap | Edits | Stop Rate | Added Gold | Added Non-Gold | Non-Gold/Gold | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank_topk | 0.9028 | 0.7720 | 2.2270 | 0.9365 | 0.0000 |  |  |  |  |  |  |
| selector_v1 | 0.9233 | 0.8050 | 2.2660 | 0.9416 | 0.6192 |  |  | 76 | 1205 | 15.8553 |  |
| oracle_edit1 | 0.9593 | 0.8950 | 2.3630 | 0.9510 | 0.0000 | 136 | 0.864 | 136 | 0 | 0.0 |  |
| none_margin15 | 0.9107 | 0.7830 | 2.2410 | 0.9417 | 0.9087 | 274 | 0.726 | 35 | 239 | 6.8286 | 2/3 |
| none_margin20 | 0.9090 | 0.7820 | 2.2390 | 0.9413 | 0.9337 | 199 | 0.801 | 27 | 172 | 6.3704 | 2/3 |
| objective_oracle_margin15 | 0.9193 | 0.8030 | 2.2610 | 0.9437 | 0.9773 | 68 | 0.932 | 35 | 33 | 0.9429 | 3/3 |
| objective_oracle_margin20 | 0.9160 | 0.7970 | 2.2540 | 0.9430 | 0.9833 | 50 | 0.95 | 27 | 23 | 0.8519 | 3/3 |
| rank_complete_margin15 | 0.9193 | 0.8030 | 2.2610 | 0.9437 | 0.9703 | 89 | 0.911 | 35 | 54 | 1.5429 | 3/3 |
| rank_complete_margin20 | 0.9160 | 0.7970 | 2.2540 | 0.9430 | 0.9787 | 64 | 0.936 | 27 | 37 | 1.3704 | 3/3 |
