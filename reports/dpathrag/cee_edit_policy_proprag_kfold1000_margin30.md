# D-PathRAG Conservative Evidence Editing Pilot

- Rows: `1000`
- Folds: `5`
- Top-k: `5`
- Candidate pool size: `20`

## Summary

| Variant | Support Recall | Support Complete | Selected Gold | Bridge Recall | Selection Overlap | Queries With Edit | Stop Rate | Added Gold | Added Non-Gold | Non-Gold/Gold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank_topk | 0.9028 | 0.7720 | 2.2270 | 0.9365 | 0.0000 |  |  |  |  |  |
| selector_v1 | 0.9233 | 0.8050 | 2.2660 | 0.9416 | 0.6192 |  |  | 76 | 1205 | 15.8553 |
| oracle_edit1 | 0.9593 | 0.8950 | 2.3630 | 0.9510 | 0.0000 | 136 | 0.864 | 136 | 0 | 0.0 |
| oracle_edit2 | 0.9607 | 0.9000 | 2.3680 | 0.9510 | 0.0000 | 136 | 0.864 | 141 | 0 | 0.0 |
| learned_edit1 | 0.9042 | 0.7740 | 2.2300 | 0.9365 | 0.9773 | 68 | 0.932 | 8 | 60 | 7.5 |
| learned_edit2 | 0.9032 | 0.7720 | 2.2280 | 0.9355 | 0.9805 | 68 | 0.932 | 11 | 86 | 7.8182 |

## Pilot Gates

| Variant | support +1pp | non-gold/gold <= 5 | added non-gold <= 50% v1 | Passed |
|---|---:|---:|---:|---:|
| learned_edit1 | False | False | True | 1/3 |
| learned_edit2 | False | False | True | 1/3 |
