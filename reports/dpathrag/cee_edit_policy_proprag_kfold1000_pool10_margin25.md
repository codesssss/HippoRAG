# D-PathRAG Conservative Evidence Editing Pilot

- Rows: `1000`
- Folds: `5`
- Top-k: `5`
- Candidate pool size: `10`

## Summary

| Variant | Support Recall | Support Complete | Selected Gold | Bridge Recall | Selection Overlap | Queries With Edit | Stop Rate | Added Gold | Added Non-Gold | Non-Gold/Gold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank_topk | 0.9028 | 0.7720 | 2.2270 | 0.9365 | 0.0000 |  |  |  |  |  |
| selector_v1 | 0.9233 | 0.8050 | 2.2660 | 0.9416 | 0.6192 |  |  | 76 | 1205 | 15.8553 |
| oracle_edit1 | 0.9387 | 0.8480 | 2.3130 | 0.9483 | 0.0000 | 87 | 0.913 | 86 | 1 | 0.0116 |
| oracle_edit2 | 0.9393 | 0.8500 | 2.3150 | 0.9483 | 0.0000 | 87 | 0.913 | 88 | 1 | 0.0114 |
| learned_edit1 | 0.9048 | 0.7750 | 2.2310 | 0.9365 | 0.9850 | 45 | 0.955 | 8 | 37 | 4.625 |
| learned_edit2 | 0.9028 | 0.7710 | 2.2270 | 0.9365 | 0.9896 | 45 | 0.955 | 9 | 67 | 7.4444 |

## Pilot Gates

| Variant | support +1pp | non-gold/gold <= 5 | added non-gold <= 50% v1 | Passed |
|---|---:|---:|---:|---:|
| learned_edit1 | False | True | True | 2/3 |
| learned_edit2 | False | False | True | 1/3 |
