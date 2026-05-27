# ETv3 Pool + Stable DBEC Residual Audit

Date: 2026-05-10

This is an offline diagnostic over the completed full1000 JSON outputs.
It does not call any model and does not change ETv3 or DBEC code.

## Overall By Dataset

| Dataset | Count | Changed | Base F1 | DBEC F1 | Delta F1 | Base title-all@5 | DBEC title-all@5 | Pool title-all@100 | Swaps | Swap-in gold | Swap-out gold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 1000 | 487 | 0.6516 | 0.6867 | 0.0351 | 0.7060 | 0.8200 | 0.9720 | 539 | 186 | 94 |
| hotpotqa | 1000 | 209 | 0.7324 | 0.7368 | 0.0043 | 0.9050 | 0.8880 | 0.9940 | 226 | 33 | 57 |
| musique | 1000 | 553 | 0.4332 | 0.3989 | -0.0343 | 0.4600 | 0.4520 | 0.8260 | 740 | 143 | 228 |

## MuSiQue Depth Breakdown

| Gold docs | Count | Changed | Base F1 | DBEC F1 | Delta F1 | Base title-all@5 | DBEC title-all@5 | Pool title-all@100 | Rescue | Regression | Avg swaps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 518 | 219 | 0.5173 | 0.4788 | -0.0384 | 0.6737 | 0.6525 | 0.9305 | 24 | 35 | 0.4846 |
| 3 | 316 | 199 | 0.3978 | 0.3594 | -0.0384 | 0.3323 | 0.3323 | 0.8228 | 25 | 25 | 0.8892 |
| 4 | 166 | 135 | 0.2382 | 0.2245 | -0.0137 | 0.0361 | 0.0542 | 0.5060 | 6 | 3 | 1.2530 |

## 2Wiki Gain Source

| Gold docs | Count | Changed | Base F1 | DBEC F1 | Delta F1 | Base title-all@5 | DBEC title-all@5 | Rescue | Regression |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 765 | 289 | 0.6235 | 0.6337 | 0.0101 | 0.8784 | 0.8954 | 44 | 31 |
| 4 | 235 | 198 | 0.7431 | 0.8596 | 0.1165 | 0.1447 | 0.5745 | 110 | 9 |

2Wiki contribution to the overall F1 delta:
- `2`-doc: absolute contribution `0.0078`, share `22.1%`.
- `4`-doc: absolute contribution `0.0274`, share `77.9%`.

## MuSiQue Residual Rank Buckets

Bucket is the maximum title-rank needed to cover all gold titles in the ETv3 pool100.

| Bucket | All queries | Pool title-complete but DBEC top5 title-incomplete |
|---|---:|---:|
| `top5` | 546 | 63 |
| `rank6_10` | 124 | 85 |
| `rank11_20` | 113 | 84 |
| `rank21_50` | 159 | 101 |
| `rank51_100` | 58 | 41 |
| `missing_100` | 0 | 0 |

## MuSiQue Harm Summary

| Metric | Count |
|---|---:|
| Worsened queries | 92 |
| Worsened and changed queries | 92 |
| Worsened with a gold title swapped out | 75 |
| Worsened with a gold title swapped in | 14 |
| Worsened without a gold title swapped out | 17 |
| Pool title-complete but DBEC top5 title-incomplete | 374 |

Breakdown of worsened queries without a gold-title swap-out:

| Category | Count |
|---|---:|
| `gold_complete_preserved_order_or_context_sensitivity` | 6 |
| `gold_recall_same_non_gold_reorder_or_context` | 9 |
| `title_recall_increased_but_reader_worse` | 2 |

## Interpretation

- The MuSiQue regression is not a missing-binding-call artifact: many swaps are made, but title-level set completeness does not improve enough and F1 drops.
- The 2-doc harm is especially important: the selector changes many shallow cases while the pre-registered regression gate required preserving them.
- The local-edit objective is positive on 2Wiki but miscalibrated on MuSiQue; this supports a residual-audit path rather than immediately naming ETv4 as state-binding.
- A clean next diagnostic is query-level manual inspection of the MuSiQue changed-and-worsened rows, especially cases with `swapped_out_gold_title_count > 0` or pool title-complete but selector title-incomplete.

## Files

- JSON: `reports/etv3_dbec_latest_full1000_residual_audit_20260510/etv3_dbec_full1000_residual_audit.json`
- Query CSV: `reports/etv3_dbec_latest_full1000_residual_audit_20260510/etv3_dbec_full1000_residual_queries.csv`
- Swap CSV: `reports/etv3_dbec_latest_full1000_residual_audit_20260510/etv3_dbec_full1000_residual_swaps.csv`
