# DAEC-DAPG Phase 0 Diagnostics

- Rows: `1000`

## Overall

| Metric | Value |
|---|---:|
| baseline_em | 0.5790 |
| baseline_f1 | 0.6503 |
| daec_l1_em | 0.6130 |
| daec_l1_f1 | 0.6828 |
| baseline_support_recall | 0.9028 |
| baseline_support_complete | 0.7720 |
| daec_l1_support_recall | 0.9215 |
| daec_l1_support_complete | 0.8100 |
| noise_rate | 0.5478 |
| reader_wrong_despite_support_complete | 0.2660 |

## Failure Categories

| Category | Count |
|---|---:|
| already_solved_by_daec_l1 | 68 |
| destructive_projection | 10 |
| mixed_or_unclear | 24 |
| no_headroom | 545 |
| reader_bottleneck | 262 |
| retrieval_or_composition_bottleneck | 91 |

## Hop Buckets

| Hop | Rows | DAEC EM | DAEC F1 | DAEC Support Complete | Reader Wrong Despite Complete |
|---|---:|---:|---:|---:|---:|
| 2 | 765 | 0.5464 | 0.6376 | 0.8706 | 0.3373 |
| 4 | 235 | 0.8298 | 0.8298 | 0.6128 | 0.0340 |
