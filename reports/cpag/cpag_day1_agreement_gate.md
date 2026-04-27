# CPAG Day-1 Agreement Gate

## Summary

| Variant | Rows | Support Recall | Support Complete | Selected Gold | Avg Pool | Avg Cross-Pool Docs |
|---|---:|---:|---:|---:|---:|---:|
| proprag_rank | 200 | 0.8888 | 0.705 | 2.155 | 31.15 | 8.835 |
| dense_rank | 200 | 0.7712 | 0.49 | 1.84 | 31.15 | 8.835 |
| rrf | 200 | 0.765 | 0.49 | 1.82 | 31.15 | 8.835 |
| cpag_pure | 200 | 0.5687 | 0.31 | 1.305 | 31.15 | 8.835 |
| cpag | 200 | 0.7412 | 0.47 | 1.735 | 31.15 | 8.835 |

## CPAG Signal

- Cross-pool gold-vs-non-gold AUC: `0.772391`
- CPAG support-complete delta vs PropRAG rank: `-0.235`
- CPAG added gold vs PropRAG: `3`
- CPAG added non-gold vs PropRAG: `87`
- CPAG non-gold/gold vs PropRAG: `29.0`

## Per-Type Support Complete

| Variant | Type | Rows | Support Complete | Support Recall |
|---|---|---:|---:|---:|
| proprag_rank | bridge_comparison | 47 | 0.2766 | 0.8032 |
| proprag_rank | comparison | 51 | 1.0 | 1.0 |
| proprag_rank | compositional | 79 | 0.7595 | 0.8797 |
| proprag_rank | inference | 23 | 0.7391 | 0.8478 |
| dense_rank | bridge_comparison | 47 | 0.0426 | 0.633 |
| dense_rank | comparison | 51 | 1.0 | 1.0 |
| dense_rank | compositional | 79 | 0.443 | 0.7215 |
| dense_rank | inference | 23 | 0.4348 | 0.7174 |
| rrf | bridge_comparison | 47 | 0.0426 | 0.617 |
| rrf | comparison | 51 | 1.0 | 1.0 |
| rrf | compositional | 79 | 0.4177 | 0.7089 |
| rrf | inference | 23 | 0.5217 | 0.7391 |
| cpag_pure | bridge_comparison | 47 | 0.0 | 0.3564 |
| cpag_pure | comparison | 51 | 0.6863 | 0.8039 |
| cpag_pure | compositional | 79 | 0.2911 | 0.5949 |
| cpag_pure | inference | 23 | 0.1739 | 0.3913 |
| cpag | bridge_comparison | 47 | 0.0 | 0.5372 |
| cpag | comparison | 51 | 0.9608 | 0.9804 |
| cpag | compositional | 79 | 0.4557 | 0.7278 |
| cpag | inference | 23 | 0.3913 | 0.6739 |

## Reader Rows

Reader JSONL files are written per variant in this directory and should be evaluated with `scripts/dpathrag_eval_reader_baseline.py`.

## Reader Evaluation

| Variant | EM | F1 | Support Recall | Support Complete |
|---|---:|---:|---:|---:|
| proprag_rank | 0.4100 | 0.4720 | 0.8888 | 0.7050 |
| rrf | 0.3650 | 0.4162 | 0.7650 | 0.4900 |
| cpag_pure | 0.3200 | 0.3580 | 0.5687 | 0.3100 |
| cpag | 0.3500 | 0.3897 | 0.7412 | 0.4700 |

## Decision

`STOP_CPAG_AGREEMENT_FAIL`

Cross-pool document presence is discriminative in isolation (`AUC=0.772391`), but the agreement-closure operator does not translate that signal into a better evidence set. Pure agreement selection collapses to high-degree hubs, while anchor-first CPAG still introduces many non-gold documents and substantially hurts both support completeness and reader F1.

## Implementation Audit

Audit report: `reports/cpag/audit200/cpag_implementation_audit.md`

The CPAG/RRF implementation was audited after observing that RRF dropped far below PropRAG rank. The audit checked doc identity alignment, RRF formula/tie-break consistency, support-complete computation, and cross-pool graph counts.

Audit summary:

```text
sample_queries = 200
avg_doc_id_overlap@20 = 8.84
avg_title_overlap@20 = 8.835
all_checks_pass = true
failure_count = 0
warning_count = 15
prop_top1_rrf_rank_gt3_count = 12
```

Interpretation: the RRF drop is not an implementation artifact. PropRAG/Dense overlap is reasonable and RRF formula/support/cross-pool counts are consistent. The issue is methodological: standard RRF and CPAG both over-reward cross-pool shared distractors, which can demote PropRAG-only gold supports.
