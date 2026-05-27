# Minimal Connected Evidence Cover: musique

This is a diagnostic-only source-grounded evidence cover. It does not modify the V13B selector.

## Summary

| Metric | Value |
|---|---:|
| query_count | 100 |
| retrieval_critical_obligation_count | 312 |
| covered_obligation_count | 91 |
| covered_obligation_rate | 0.2917 |
| full_cover_feasible_count | 16 |
| full_cover_feasible_rate | 0.1600 |
| connected_cover_count | 54 |
| connected_cover_rate | 0.5400 |
| uses_source_grounded_match_count | 47 |
| minimal_cover_recall_at5 | 0.6883 |
| source_recall_at5 | 0.6775 |
| selector_recall_at5 | 0.6825 |
| minimal_cover_all_gold_at5 | 0.4100 |
| source_all_gold_at5 | 0.3900 |
| selector_all_gold_at5 | 0.4000 |
| gold_count_gains_vs_source_top5 | 5 |
| gold_count_losses_vs_source_top5 | 2 |
| gold_count_gains_vs_selector_top5 | 4 |
| gold_count_losses_vs_selector_top5 | 2 |
| certified_admission_count | 16 |
| certified_admission_recall_at5 | 0.6792 |
| certified_admission_all_gold_at5 | 0.3900 |
| certified_admission_gains_vs_selector_top5 | 0 |
| certified_admission_losses_vs_selector_top5 | 1 |

## Interpretation

A useful signal requires both higher demand coverage and non-negative retrieval behavior versus the source top5.
If recall drops, the cover objective is not selector-ready even if source-grounded demand coverage is high.
