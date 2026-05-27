# V13B Grounding Failure Report: 2wikimultihopqa

## Summary

| metric | value |
|---|---:|
| retrieval_critical_ungrounded | 43 |
| classified_failure_count | 71 |
| unknown_fraction | 0.0 |

## Failure Types

| failure_type | count |
|---|---:|
| bridge_missing | 5 |
| candidate_missing | 1 |
| endpoint_missing | 3 |
| relation_mismatch | 40 |
| upstream_binding_missing | 22 |

## Relation Mismatch Pairs

| query_relation -> fact_relation | count |
|---|---:|
| composer -> cast_includ | 13 |
| establish_in -> <empty> | 11 |
| released_on -> <empty> | 9 |
| establish_in -> collection | 8 |
| father -> son | 7 |
| country_origin -> starr | 7 |
| composer -> starr | 7 |
| establish_in -> dedicat | 7 |
| released_on -> starr | 6 |
| released_on -> support_rol | 6 |
| country_origin -> <empty> | 6 |
| country_origin -> about | 6 |
| country_origin -> star | 6 |
| nationality -> <empty> | 6 |
| place_death -> <empty> | 5 |
| composer -> collaborat_with | 5 |
| work_at -> produc | 5 |
| born_in -> <empty> | 5 |
| establish_in -> locat_in | 5 |
| nationality -> play_for | 5 |
| released_on -> direct | 4 |
| released_on -> produc | 4 |
| father -> <empty> | 4 |
| wife -> associat_with | 4 |
| composer -> <empty> | 4 |
| composer -> produc | 4 |
| composer -> star | 4 |
| husband -> <empty> | 4 |
| composer -> prais_cast | 4 |
| husband -> won_in | 4 |
| country_origin -> direct | 4 |
| country_origin -> written | 4 |
| parent -> marriage | 4 |
| released_on -> star | 3 |
| wife -> produc | 3 |
| composer -> co_produc | 3 |
| composer -> direct | 3 |
| composer -> open_in | 3 |
| composer -> released_on | 3 |
| composer -> written | 3 |

## Source Prior vs V13B

| metric | value |
|---|---:|
| count | 100 |
| source_r5 | 0.935 |
| v13b_r5 | 0.955 |
| source_all_gold_at5 | 0.85 |
| v13b_all_gold_at5 | 0.92 |
| gold_count_improved_queries | 7 |
| gold_count_regressed_queries | 0 |

## Representative Cases

### relation_mismatch
- q=1 triple=['Aas Ka Panchhi', 'release date', '?date1'] reason=relation_mismatch
- q=1 triple=['Phoolwari', 'release date', '?date2'] reason=relation_mismatch
- q=7 triple=['Raghnall Mac Ruaidhrí', 'paternal grandfather', '?x1'] reason=relation_mismatch
- q=15 triple=['Aleksander Koniecpolski (1620–1659)', 'father', '?x1'] reason=relation_mismatch
- q=21 triple=['John Middleton Murry', 'wife', '?x1'] reason=relation_mismatch

### bridge_missing
- q=3 triple=['Nasamkhrali', 'located in', '?country'] reason=bridge_missing
- q=30 triple=['?x2', 'born on', '?date2'] reason=bridge_missing
- q=41 triple=['?x2', 'born on', '?date2'] reason=bridge_missing
- q=50 triple=['Domenico Ravenna', 'born in', '?place'] reason=bridge_missing
- q=71 triple=['?x1', 'died in', '?place'] reason=bridge_missing

### upstream_binding_missing
- q=14 triple=['?x1', 'died in', '?place'] reason=upstream_binding_missing
- q=15 triple=['?x1', 'is from', '?country'] reason=upstream_binding_missing
- q=20 triple=['?x1', 'studied at', '?place'] reason=upstream_binding_missing
- q=21 triple=['?x1', 'died', '?reason'] reason=upstream_binding_missing
- q=22 triple=['?x1', 'was born in', '?place'] reason=upstream_binding_missing

### endpoint_missing
- q=23 triple=['Lisbeth Palme', 'husband', '?x1'] reason=endpoint_missing
- q=34 triple=['?x1', 'works at', '?place'] reason=endpoint_missing
- q=43 triple=['?x2', 'born on', '?date2'] reason=endpoint_missing

### candidate_missing
- q=33 triple=['?x1', 'was born in', '?place'] reason=candidate_missing
