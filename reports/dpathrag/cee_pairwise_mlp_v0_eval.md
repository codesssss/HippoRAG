# CEE Pairwise v0 Eval

- Rows: `1000`
- Folds: `5`
- Candidate pool: `top20`

## Main Selection Metrics

| Variant | Support Complete | Support Recall | Added Gold | Added Non-Gold | Non-Gold/Gold | Stop Rate | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rank_top5` | 0.7720 | 0.9028 | - | - | - | - | - |
| `oracle_edit1_top20` | 0.8950 | 0.9593 | 136 | 0 | 0.0 | 0.864 | - |
| `selector_v1` | 0.8050 | 0.9233 | 76 | 1205 | 15.8553 | - | - |
| `shallow_cee_margin20` | 0.7780 | 0.9070 | 54 | 344 | 6.3704 | 0.801 | - |
| `cee_pairwise_mlp_v0` | 0.7590 | 0.8990 | 22 | 273 | 12.4091 | 0.705 | 1/3 |

## Admission Diagnostics

| Variant | Rank-Complete Over-Edit | Oracle Beneficial Recall | Oracle Non-Beneficial FP | Lexical HN Admission |
|---|---:|---:|---:|---:|
| `cee_pairwise_mlp_v0` | 0.2210 | 0.1544 | 0.2836 | 0.5797 |
