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
| learned_edit1 | 0.9090 | 0.7820 | 2.2390 | 0.9413 | 0.9337 | 199 | 0.801 | 27 | 172 | 6.3704 |
| learned_edit2 | 0.9050 | 0.7740 | 2.2310 | 0.9403 | 0.9456 | 199 | 0.801 | 35 | 286 | 8.1714 |

## Pilot Gates

| Variant | support +1pp | non-gold/gold <= 5 | added non-gold <= 50% v1 | Passed |
|---|---:|---:|---:|---:|
| learned_edit1 | True | False | True | 2/3 |
| learned_edit2 | False | False | True | 1/3 |
