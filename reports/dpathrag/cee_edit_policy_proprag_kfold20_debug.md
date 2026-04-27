# D-PathRAG Conservative Evidence Editing Pilot

- Rows: `20`
- Folds: `5`
- Top-k: `5`
- Candidate pool size: `20`

## Summary

| Variant | Support Recall | Support Complete | Selected Gold | Bridge Recall | Selection Overlap | Queries With Edit | Stop Rate | Added Gold | Added Non-Gold | Non-Gold/Gold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank_topk | 0.9375 | 0.8500 | 2.3500 | 0.9500 | 0.0000 |  |  |  |  |  |
| selector_v1 | 0.9500 | 0.8500 | 2.3500 | 0.9500 | 0.6238 |  |  | 1 | 23 | 23.0 |
| oracle_edit1 | 0.9750 | 0.9500 | 2.4500 | 0.9500 | 0.0000 | 2 | 0.9 | 2 | 0 | 0.0 |
| oracle_edit2 | 0.9750 | 0.9500 | 2.4500 | 0.9500 | 0.0000 | 2 | 0.9 | 2 | 0 | 0.0 |
| learned_edit1 | 0.9375 | 0.8500 | 2.3500 | 0.9500 | 1.0000 | 0 | 1.0 | 0 | 0 | 0.0 |
| learned_edit2 | 0.9375 | 0.8500 | 2.3500 | 0.9500 | 1.0000 | 0 | 1.0 | 0 | 0 | 0.0 |

## Pilot Gates

| Variant | support +1pp | non-gold/gold <= 5 | added non-gold <= 50% v1 | Passed |
|---|---:|---:|---:|---:|
| learned_edit1 | False | False | True | 1/3 |
| learned_edit2 | False | False | True | 1/3 |
