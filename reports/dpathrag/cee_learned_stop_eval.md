# CEE Learned STOP Evaluation

- Rows: `1000`
- Folds: `5`
- Candidate pool size: `20`

| Variant | Support Recall | Support Complete | Selected Gold | Bridge Recall | Selection Overlap | Edits | Stop Rate | Added Gold | Added Non-Gold | Non-Gold/Gold | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank_topk | 0.9028 | 0.7720 | 2.2270 | 0.9365 | 0.0000 |  |  |  |  |  |  |
| selector_v1 | 0.9233 | 0.8050 | 2.2660 | 0.9416 | 0.6192 |  |  | 76 | 1205 | 15.8553 |  |
| oracle_edit1 | 0.9593 | 0.8950 | 2.3630 | 0.9510 | 0.0000 | 136 | 0.864 | 136 | 0 | 0.0 |  |
| learned_stop_margin15 | 0.9070 | 0.7780 | 2.2350 | 0.9390 | 0.9510 | 147 | 0.853 | 19 | 128 | 6.7368 | 1/3 |
| learned_stop_margin20 | 0.9060 | 0.7780 | 2.2340 | 0.9390 | 0.9603 | 119 | 0.881 | 16 | 103 | 6.4375 | 1/3 |

## Fold Thresholds

| Fold | Variant | Threshold | Dev Gates | Eval AUC | Eval Precision | Eval Recall |
|---|---|---:|---:|---:|---:|---:|
| 0 | learned_stop_margin15 | 96.4397 | 3/3 | 0.7131 | 0.0000 | 0.0000 |
| 0 | learned_stop_margin20 | 96.4397 | 3/3 | 0.7131 | 0.0000 | 0.0000 |
| 1 | learned_stop_margin15 | 32.0056 | 3/3 | 0.6834 | 0.2903 | 0.3913 |
| 1 | learned_stop_margin20 | -23.1723 | 3/3 | 0.6834 | 0.1455 | 0.6957 |
| 2 | learned_stop_margin15 | -50.4284 | 1/3 | 0.6508 | 0.1859 | 0.9667 |
| 2 | learned_stop_margin20 | -50.4284 | 1/3 | 0.6508 | 0.1859 | 0.9667 |
| 3 | learned_stop_margin15 | 143.5467 | 3/3 | 0.6087 | 0.0000 | 0.0000 |
| 3 | learned_stop_margin20 | 143.5467 | 3/3 | 0.6087 | 0.0000 | 0.0000 |
| 4 | learned_stop_margin15 | -97.8461 | 3/3 | 0.7266 | 0.1600 | 1.0000 |
| 4 | learned_stop_margin20 | -97.8461 | 3/3 | 0.7266 | 0.1600 | 1.0000 |
