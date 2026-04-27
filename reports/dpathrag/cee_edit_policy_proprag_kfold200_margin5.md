# D-PathRAG Conservative Evidence Editing Pilot

- Rows: `200`
- Folds: `5`
- Top-k: `5`
- Candidate pool size: `20`

## Summary

| Variant | Support Recall | Support Complete | Selected Gold | Bridge Recall | Selection Overlap | Queries With Edit | Stop Rate | Added Gold | Added Non-Gold | Non-Gold/Gold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank_topk | 0.9175 | 0.8150 | 2.2650 | 0.9383 | 0.0000 |  |  |  |  |  |
| selector_v1 | 0.9237 | 0.8150 | 2.2750 | 0.9467 | 0.6230 |  |  | 6 | 246 | 41.0 |
| oracle_edit1 | 0.9587 | 0.9000 | 2.3650 | 0.9500 | 0.0000 | 20 | 0.9 | 20 | 0 | 0.0 |
| oracle_edit2 | 0.9637 | 0.9150 | 2.3800 | 0.9500 | 0.0000 | 20 | 0.9 | 23 | 0 | 0.0 |
| learned_edit1 | 0.9025 | 0.7650 | 2.2100 | 0.9267 | 0.9317 | 41 | 0.795 | 0 | 41 | 41.0 |
| learned_edit2 | 0.8962 | 0.7550 | 2.1950 | 0.9267 | 0.9260 | 41 | 0.795 | 0 | 59 | 59.0 |

## Pilot Gates

| Variant | support +1pp | non-gold/gold <= 5 | added non-gold <= 50% v1 | Passed |
|---|---:|---:|---:|---:|
| learned_edit1 | False | False | True | 1/3 |
| learned_edit2 | False | False | True | 1/3 |
