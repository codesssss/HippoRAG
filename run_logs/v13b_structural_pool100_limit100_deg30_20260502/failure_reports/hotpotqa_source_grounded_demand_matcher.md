# Source-Grounded Demand Matcher: hotpotqa

This is a diagnostic-only probe. It does not change retrieval or selector outputs.

## Summary

| Metric | Value |
|---|---:|
| query_count | 100 |
| retrieval_critical_obligation_count | 216 |
| openie_exact_count | 38 |
| source_grounded_match_count | 88 |
| new_matches_over_openie_exact | 56 |
| exact_or_source_grounded_count | 94 |
| openie_exact_rate | 0.1759 |
| source_grounded_new_rate | 0.2593 |
| exact_or_source_rate | 0.4352 |

## Status Counts

| Status | Count |
|---|---:|
| openie_exact_available | 38 |
| source_grounded_new_match | 56 |
| unmatched | 122 |

## Interpretation

If `new_matches_over_openie_exact` is non-trivial, the next clean step is a minimal connected evidence cover over source-grounded units, not OpenIE prompt/schema tuning.
If it is small, this line should be downgraded because source text matching does not recover the OpenIE grounding gap.
