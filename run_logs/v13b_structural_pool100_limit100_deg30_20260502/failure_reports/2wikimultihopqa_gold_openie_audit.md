# V13B Gold OpenIE Audit: 2wikimultihopqa

## Summary

| metric | value |
|---|---:|
| audited_ungrounded_obligation_count | 71 |

## Gold OpenIE Status

| status | count |
|---|---:|
| candidate_missing | 1 |
| gold_endpoint_missing | 2 |
| gold_endpoint_relation_mismatch | 65 |
| gold_exact_openie_match_available | 1 |
| gold_relation_present_without_endpoint_binding | 2 |

## Relation Mismatch Pairs On Gold Docs

| query_relation -> gold_fact_relation | count |
|---|---:|
| born_in -> <empty> | 32 |
| born_on -> <empty> | 27 |
| died_in -> <empty> | 16 |
| from -> <empty> | 16 |
| born_on -> direct | 16 |
| born_in -> criticiz | 13 |
| composer -> cast_includ | 13 |
| born_in -> cast_includ | 13 |
| born_on -> died_on | 11 |
| establish_in -> <empty> | 11 |
| born_on -> born_on | 10 |
| born_on -> starr | 10 |
| born_on -> released_on | 10 |
| died_in -> made | 10 |
| released_on -> <empty> | 9 |
| born_on -> known_as | 9 |
| died_in -> died_on | 8 |
| born_on -> work_as | 8 |
| born_on -> born_in | 8 |
| born_on -> specialis_in | 8 |
| establish_in -> collection | 8 |
| born_in -> collaborat_with | 7 |
| detain_in -> <empty> | 7 |
| country_origin -> starr | 7 |
| composer -> starr | 7 |
| from -> starr | 7 |
| establish_in -> dedicat | 7 |
| released_on -> starr | 6 |
| died_in -> born_on | 6 |
| from -> defeat | 6 |
| study_at -> inherit_from | 6 |
| study_at -> died_on | 6 |
| born_in -> released_on | 6 |
| born_in -> born_on | 6 |
| born_in -> died_on | 6 |
| released_on -> support_rol | 6 |
| born_on -> work_for | 6 |
| detain_in -> released_on | 6 |
| country_origin -> <empty> | 6 |
| country_origin -> about | 6 |

## Representative Cases

### gold_endpoint_relation_mismatch
- q=1 triple=['Aas Ka Panchhi', 'release date', '?date1'] gold=[17, 19]
- q=1 triple=['Phoolwari', 'release date', '?date2'] gold=[17, 19]
- q=7 triple=['Raghnall Mac Ruaidhrí', 'paternal grandfather', '?x1'] gold=[73, 75]
- q=14 triple=['?x1', 'died in', '?place'] gold=[131, 133]
- q=15 triple=['Aleksander Koniecpolski (1620–1659)', 'father', '?x1'] gold=[138, 140]

### gold_exact_openie_match_available
- q=3 triple=['Nasamkhrali', 'located in', '?country'] gold=[39, 31]

### gold_endpoint_missing
- q=23 triple=['Lisbeth Palme', 'husband', '?x1'] gold=[213, 211]
- q=34 triple=['?x1', 'works at', '?place'] gold=[304, 306]

### candidate_missing
- q=33 triple=['?x1', 'was born in', '?place'] gold=[296, 297]

### gold_relation_present_without_endpoint_binding
- q=43 triple=['?x2', 'born on', '?date2'] gold=[372, 373, 374, 371]
- q=50 triple=['Domenico Ravenna', 'born in', '?place'] gold=[420, 419]
