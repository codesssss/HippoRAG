# Source-Grounded Demand Matcher: 2wikimultihopqa

This is a diagnostic-only probe. It does not change retrieval or selector outputs.

## Summary

| Metric | Value |
|---|---:|
| query_count | 100 |
| retrieval_critical_obligation_count | 220 |
| openie_exact_count | 127 |
| source_grounded_match_count | 89 |
| new_matches_over_openie_exact | 19 |
| exact_or_source_grounded_count | 146 |
| openie_exact_rate | 0.5773 |
| source_grounded_new_rate | 0.0864 |
| exact_or_source_rate | 0.6636 |

## Status Counts

| Status | Count |
|---|---:|
| openie_exact_available | 127 |
| source_grounded_new_match | 19 |
| unmatched | 74 |

## Interpretation

If `new_matches_over_openie_exact` is non-trivial, the next clean step is a minimal connected evidence cover over source-grounded units, not OpenIE prompt/schema tuning.
If it is small, this line should be downgraded because source text matching does not recover the OpenIE grounding gap.
