# Minimal Connected Evidence Cover: hotpotqa

This is a diagnostic-only source-grounded evidence cover. It does not modify the V13B selector.

## Summary

| Metric | Value |
|---|---:|
| query_count | 100 |
| retrieval_critical_obligation_count | 216 |
| covered_obligation_count | 104 |
| covered_obligation_rate | 0.4815 |
| full_cover_feasible_count | 29 |
| full_cover_feasible_rate | 0.2900 |
| connected_cover_count | 63 |
| connected_cover_rate | 0.6300 |
| uses_source_grounded_match_count | 52 |
| minimal_cover_recall_at5 | 0.9400 |
| source_recall_at5 | 0.9300 |
| selector_recall_at5 | 0.9300 |
| minimal_cover_all_gold_at5 | 0.8900 |
| source_all_gold_at5 | 0.8700 |
| selector_all_gold_at5 | 0.8700 |
| gold_count_gains_vs_source_top5 | 2 |
| gold_count_losses_vs_source_top5 | 0 |
| gold_count_gains_vs_selector_top5 | 2 |
| gold_count_losses_vs_selector_top5 | 0 |
| certified_admission_count | 25 |
| certified_admission_recall_at5 | 0.9350 |
| certified_admission_all_gold_at5 | 0.8800 |
| certified_admission_gains_vs_selector_top5 | 1 |
| certified_admission_losses_vs_selector_top5 | 0 |

## Interpretation

A useful signal requires both higher demand coverage and non-negative retrieval behavior versus the source top5.
If recall drops, the cover objective is not selector-ready even if source-grounded demand coverage is high.
