# V13B Grounding Failure Report: hotpotqa

## Summary

| metric | value |
|---|---:|
| retrieval_critical_ungrounded | 4 |
| classified_failure_count | 14 |
| unknown_fraction | 0.0 |

## Failure Types

| failure_type | count |
|---|---:|
| bridge_missing | 4 |
| endpoint_missing | 4 |
| overconstrained_query_demand | 1 |
| relation_mismatch | 5 |

## Source Prior vs V13B

| metric | value |
|---|---:|
| count | 20 |
| source_r5 | 0.95 |
| v13b_r5 | 0.95 |
| source_all_gold_at5 | 0.9 |
| v13b_all_gold_at5 | 0.9 |
| gold_count_improved_queries | 0 |
| gold_count_regressed_queries | 0 |

## Representative Cases

### relation_mismatch
- q=3 triple=['Marian civil war', 'sparked by', '?x1'] reason=relation_mismatch
- q=9 triple=['?x1', 'is from', 'Jamaica'] reason=relation_mismatch
- q=12 triple=['No Animals Were Harmed', 'focus on', '?x2'] reason=relation_mismatch
- q=14 triple=['Summer Wars', 'country of origin', '?x1'] reason=relation_mismatch
- q=14 triple=['The Secret of Kells', 'country of origin', '?x2'] reason=relation_mismatch

### endpoint_missing
- q=3 triple=['?x1', 'helped', 'recently abdicated queen'] reason=endpoint_missing
- q=3 triple=['recently abdicated queen', 'imprisoned by', '?x2'] reason=endpoint_missing
- q=5 triple=['Miklos Rozsa', 'worked on', '?screenplay'] reason=endpoint_missing
- q=9 triple=['The 2000 ICC KnockOut Trophy', 'debut of', '?x1'] reason=endpoint_missing

### bridge_missing
- q=12 triple=['Peter Laufer', 'wrote', 'Forbidden Creatures'] reason=bridge_missing
- q=12 triple=['Peter Laufer', 'wrote', 'No Animals Were Harmed'] reason=bridge_missing
- q=14 triple=['Summer Wars', 'year', '2009'] reason=bridge_missing
- q=14 triple=['The Secret of Kells', 'year', '2009'] reason=bridge_missing

### overconstrained_query_demand
- q=19 triple=['?team', 'based in', '?place'] reason=overconstrained_query_demand
