# Source-Grounded Demand Matcher: hotpotqa

This is a diagnostic-only probe. It does not change retrieval or selector outputs.

## Summary

| Metric | Value |
|---|---:|
| query_count | 100 |
| retrieval_critical_obligation_count | 216 |
| openie_exact_count | 41 |
| source_grounded_match_count | 98 |
| new_matches_over_openie_exact | 65 |
| exact_or_source_grounded_count | 106 |
| openie_exact_rate | 0.1898 |
| source_grounded_new_rate | 0.3009 |
| exact_or_source_rate | 0.4907 |

## Status Counts

| Status | Count |
|---|---:|
| openie_exact_available | 41 |
| source_grounded_new_match | 65 |
| unmatched | 110 |

## Interpretation

If `new_matches_over_openie_exact` is non-trivial, the next clean step is a minimal connected evidence cover over source-grounded units, not OpenIE prompt/schema tuning.
If it is small, this line should be downgraded because source text matching does not recover the OpenIE grounding gap.
