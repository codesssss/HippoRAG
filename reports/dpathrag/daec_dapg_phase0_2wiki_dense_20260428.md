# DAEC-DAPG Phase 0 Diagnostics

- Rows: `1000`

## Overall

| Metric | Value |
|---|---:|
| baseline_em | 0.4550 |
| baseline_f1 | 0.4991 |
| daec_l1_em | 0.4980 |
| daec_l1_f1 | 0.5541 |
| baseline_support_recall | 0.7238 |
| baseline_support_complete | 0.4290 |
| daec_l1_support_recall | 0.7855 |
| daec_l1_support_complete | 0.5440 |
| noise_rate | 0.6296 |
| reader_wrong_despite_support_complete | 0.1840 |

## Failure Categories

| Category | Count |
|---|---:|
| already_solved_by_daec_l1 | 95 |
| destructive_projection | 10 |
| mixed_or_unclear | 42 |
| no_headroom | 403 |
| reader_bottleneck | 175 |
| retrieval_or_composition_bottleneck | 275 |

## Hop Buckets

| Hop | Rows | DAEC EM | DAEC F1 | DAEC Support Complete | Reader Wrong Despite Complete |
|---|---:|---:|---:|---:|---:|
| 2 | 765 | 0.4706 | 0.5432 | 0.6967 | 0.2392 |
| 4 | 235 | 0.5872 | 0.5897 | 0.0468 | 0.0043 |
