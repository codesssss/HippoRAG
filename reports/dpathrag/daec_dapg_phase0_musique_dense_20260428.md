# DAEC-DAPG Phase 0 Diagnostics

- Rows: `1000`

## Overall

| Metric | Value |
|---|---:|
| baseline_em | 0.2980 |
| baseline_f1 | 0.3896 |
| daec_l1_em | 0.2900 |
| daec_l1_f1 | 0.3818 |
| baseline_support_recall | 0.6888 |
| baseline_support_complete | 0.3860 |
| daec_l1_support_recall | 0.6854 |
| daec_l1_support_complete | 0.3720 |
| noise_rate | 0.6632 |
| reader_wrong_despite_support_complete | 0.1720 |

## Failure Categories

| Category | Count |
|---|---:|
| already_solved_by_daec_l1 | 57 |
| destructive_projection | 27 |
| mixed_or_unclear | 38 |
| no_headroom | 233 |
| reader_bottleneck | 182 |
| retrieval_or_composition_bottleneck | 463 |

## Hop Buckets

| Hop | Rows | DAEC EM | DAEC F1 | DAEC Support Complete | Reader Wrong Despite Complete |
|---|---:|---:|---:|---:|---:|
| 2 | 518 | 0.3668 | 0.4635 | 0.5849 | 0.2568 |
| 3 | 316 | 0.2437 | 0.3344 | 0.1994 | 0.1108 |
| 4 | 166 | 0.1386 | 0.2173 | 0.0361 | 0.0241 |
