# V13B Grounding Failure Report: musique

## Summary

| metric | value |
|---|---:|
| retrieval_critical_ungrounded | 13 |
| classified_failure_count | 25 |
| unknown_fraction | 0.0 |

## Failure Types

| failure_type | count |
|---|---:|
| candidate_missing | 6 |
| endpoint_missing | 7 |
| overconstrained_query_demand | 8 |
| relation_mismatch | 4 |

## Source Prior vs V13B

| metric | value |
|---|---:|
| count | 20 |
| source_r5 | 0.625 |
| v13b_r5 | 0.65 |
| source_all_gold_at5 | 0.4 |
| v13b_all_gold_at5 | 0.45 |
| gold_count_improved_queries | 1 |
| gold_count_regressed_queries | 0 |

## Representative Cases

### relation_mismatch
- q=1 triple=['France', 'participated in', '?x1'] reason=relation_mismatch
- q=9 triple=['Till dom ensamma', 'performer', '?x1'] reason=relation_mismatch
- q=10 triple=['The Bag or the Bat', 'is a series of', '?series'] reason=relation_mismatch
- q=16 triple=['Palau de la Generalitat', 'constructed in', '?city'] reason=relation_mismatch

### overconstrained_query_demand
- q=1 triple=['?x1', 'involved', '?country'] reason=overconstrained_query_demand
- q=1 triple=['?country', 'originated', '?x2'] reason=overconstrained_query_demand
- q=6 triple=['?x1', 'died in', '?city'] reason=overconstrained_query_demand
- q=6 triple=['?body', 'located in', '?city'] reason=overconstrained_query_demand
- q=9 triple=['?x1', 'born on', '?date'] reason=overconstrained_query_demand

### endpoint_missing
- q=1 triple=['?country', 'headquartered in', 'the nobilities commonwealth'] reason=endpoint_missing
- q=1 triple=['?x2', 'top-ranking', 'Warsaw Pact operatives'] reason=endpoint_missing
- q=4 triple=['Lady Godiva', 'birthplace', '?x1'] reason=endpoint_missing
- q=11 triple=['new coins', 'proclamation of independence by', 'Somali Muslim Ajuran Empire'] reason=endpoint_missing
- q=11 triple=['?x1', 'includes', "A Lim's country"] reason=endpoint_missing

### candidate_missing
- q=5 triple=['?region1', 'location of', 'Battle of Qurah'] reason=candidate_missing
- q=5 triple=['?region1', 'location of', 'Umm al Maradim'] reason=candidate_missing
- q=7 triple=['?x2', 'headquarters located in', '?city'] reason=candidate_missing
- q=7 triple=['?x1', 'headquarters located in', '?city'] reason=candidate_missing
- q=8 triple=['?x1', 'was born in', '?place'] reason=candidate_missing
