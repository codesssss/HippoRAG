# Minimal Connected Evidence Cover: 2wikimultihopqa

This is a diagnostic-only source-grounded evidence cover. It does not modify the V13B selector.

## Summary

| Metric | Value |
|---|---:|
| query_count | 100 |
| retrieval_critical_obligation_count | 276 |
| covered_obligation_count | 150 |
| covered_obligation_rate | 0.5435 |
| full_cover_feasible_count | 21 |
| full_cover_feasible_rate | 0.2100 |
| connected_cover_count | 41 |
| connected_cover_rate | 0.4100 |
| uses_source_grounded_match_count | 22 |
| minimal_cover_recall_at5 | 0.9600 |
| source_recall_at5 | 0.9350 |
| selector_recall_at5 | 0.9550 |
| minimal_cover_all_gold_at5 | 0.9100 |
| source_all_gold_at5 | 0.8500 |
| selector_all_gold_at5 | 0.9200 |
| gold_count_gains_vs_source_top5 | 7 |
| gold_count_losses_vs_source_top5 | 0 |
| gold_count_gains_vs_selector_top5 | 2 |
| gold_count_losses_vs_selector_top5 | 2 |
| certified_admission_count | 20 |
| certified_admission_recall_at5 | 0.9600 |
| certified_admission_all_gold_at5 | 0.9300 |
| certified_admission_gains_vs_selector_top5 | 1 |
| certified_admission_losses_vs_selector_top5 | 0 |

## Interpretation

A useful signal requires both higher demand coverage and non-negative retrieval behavior versus the source top5.
If recall drops, the cover objective is not selector-ready even if source-grounded demand coverage is high.
