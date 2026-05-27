# V13B Grounding Failure Report: 2wikimultihopqa

## Summary

| metric | value |
|---|---:|
| retrieval_critical_ungrounded | 5 |
| classified_failure_count | 7 |
| unknown_fraction | 0.0 |

## Failure Types

| failure_type | count |
|---|---:|
| bridge_missing | 1 |
| overconstrained_query_demand | 2 |
| relation_mismatch | 4 |

## Source Prior vs V13B

| metric | value |
|---|---:|
| count | 20 |
| source_r5 | 0.9375 |
| v13b_r5 | 0.975 |
| source_all_gold_at5 | 0.85 |
| v13b_all_gold_at5 | 0.95 |
| gold_count_improved_queries | 2 |
| gold_count_regressed_queries | 0 |

## Representative Cases

### relation_mismatch
- q=1 triple=['Aas Ka Panchhi', 'release date', '?date1'] reason=relation_mismatch
- q=1 triple=['Phoolwari', 'release date', '?date2'] reason=relation_mismatch
- q=7 triple=['Raghnall Mac Ruaidhrí', 'paternal grandfather', '?x1'] reason=relation_mismatch
- q=15 triple=['Aleksander Koniecpolski (1620–1659)', 'father', '?x1'] reason=relation_mismatch

### bridge_missing
- q=3 triple=['Nasamkhrali', 'located in', '?country'] reason=bridge_missing

### overconstrained_query_demand
- q=14 triple=['?x1', 'died in', '?place'] reason=overconstrained_query_demand
- q=15 triple=['?x1', 'is from', '?country'] reason=overconstrained_query_demand
