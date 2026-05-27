# Source-Grounded Demand Matcher: musique

This is a diagnostic-only probe. It does not change retrieval or selector outputs.

## Summary

| Metric | Value |
|---|---:|
| query_count | 100 |
| retrieval_critical_obligation_count | 253 |
| openie_exact_count | 26 |
| source_grounded_match_count | 58 |
| new_matches_over_openie_exact | 39 |
| exact_or_source_grounded_count | 65 |
| openie_exact_rate | 0.1028 |
| source_grounded_new_rate | 0.1542 |
| exact_or_source_rate | 0.2569 |

## Status Counts

| Status | Count |
|---|---:|
| openie_exact_available | 26 |
| source_grounded_new_match | 39 |
| unmatched | 188 |

## Interpretation

If `new_matches_over_openie_exact` is non-trivial, the next clean step is a minimal connected evidence cover over source-grounded units, not OpenIE prompt/schema tuning.
If it is small, this line should be downgraded because source text matching does not recover the OpenIE grounding gap.
