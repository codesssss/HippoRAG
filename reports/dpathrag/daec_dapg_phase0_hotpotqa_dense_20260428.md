# DAEC-DAPG Phase 0 Diagnostics

- Rows: `1000`

## Overall

| Metric | Value |
|---|---:|
| baseline_em | 0.5950 |
| baseline_f1 | 0.7106 |
| daec_l1_em | 0.6120 |
| daec_l1_f1 | 0.7267 |
| baseline_support_recall | 0.9305 |
| baseline_support_complete | 0.8640 |
| daec_l1_support_recall | 0.9360 |
| daec_l1_support_complete | 0.8810 |
| noise_rate | 0.6258 |
| reader_wrong_despite_support_complete | 0.2980 |

## Failure Categories

| Category | Count |
|---|---:|
| already_solved_by_daec_l1 | 45 |
| destructive_projection | 11 |
| mixed_or_unclear | 17 |
| no_headroom | 567 |
| reader_bottleneck | 294 |
| retrieval_or_composition_bottleneck | 66 |

## Hop Buckets

| Hop | Rows | DAEC EM | DAEC F1 | DAEC Support Complete | Reader Wrong Despite Complete |
|---|---:|---:|---:|---:|---:|
| 2 | 1000 | 0.6120 | 0.7267 | 0.8810 | 0.2980 |
