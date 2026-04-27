# CPAG Day-1 Agreement Gate

## Summary

| Variant | Rows | Support Recall | Support Complete | Selected Gold | Avg Pool | Avg Cross-Pool Docs |
|---|---:|---:|---:|---:|---:|---:|
| proprag_rank | 5 | 0.9 | 0.8 | 2.2 | 30.6 | 9.4 |
| dense_rank | 5 | 0.85 | 0.6 | 2.0 | 30.6 | 9.4 |
| rrf | 5 | 0.85 | 0.6 | 2.0 | 30.6 | 9.4 |
| cpag_pure | 5 | 0.55 | 0.0 | 1.4 | 30.6 | 9.4 |
| cpag | 5 | 0.85 | 0.6 | 2.0 | 30.6 | 9.4 |

## CPAG Signal

- Cross-pool gold-vs-non-gold AUC: `0.824264`
- CPAG support-complete delta vs PropRAG rank: `-0.2`
- CPAG added gold vs PropRAG: `0`
- CPAG added non-gold vs PropRAG: `1`
- CPAG non-gold/gold vs PropRAG: `None`

## Per-Type Support Complete

| Variant | Type | Rows | Support Complete | Support Recall |
|---|---|---:|---:|---:|
| proprag_rank | bridge_comparison | 1 | 1.0 | 1.0 |
| proprag_rank | comparison | 2 | 1.0 | 1.0 |
| proprag_rank | compositional | 2 | 0.5 | 0.75 |
| dense_rank | bridge_comparison | 1 | 0.0 | 0.75 |
| dense_rank | comparison | 2 | 1.0 | 1.0 |
| dense_rank | compositional | 2 | 0.5 | 0.75 |
| rrf | bridge_comparison | 1 | 0.0 | 0.75 |
| rrf | comparison | 2 | 1.0 | 1.0 |
| rrf | compositional | 2 | 0.5 | 0.75 |
| cpag_pure | bridge_comparison | 1 | 0.0 | 0.75 |
| cpag_pure | comparison | 2 | 0.0 | 0.5 |
| cpag_pure | compositional | 2 | 0.0 | 0.5 |
| cpag | bridge_comparison | 1 | 0.0 | 0.75 |
| cpag | comparison | 2 | 1.0 | 1.0 |
| cpag | compositional | 2 | 0.5 | 0.75 |

## Reader Rows

Reader JSONL files are written per variant in this directory and should be evaluated with `scripts/dpathrag_eval_reader_baseline.py`.
