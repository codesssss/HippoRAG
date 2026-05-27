# V13B Grounding Failure Report: musique

## Summary

| metric | value |
|---|---:|
| retrieval_critical_ungrounded | 72 |
| classified_failure_count | 161 |
| unknown_fraction | 0.037267 |

## Failure Types

| failure_type | count |
|---|---:|
| bridge_missing | 6 |
| candidate_missing | 12 |
| endpoint_missing | 36 |
| relation_mismatch | 37 |
| unknown | 6 |
| upstream_binding_missing | 64 |

## Relation Mismatch Pairs

| query_relation -> fact_relation | count |
|---|---:|
| majority_religion -> population_practic | 12 |
| locat_in -> flow_through | 10 |
| underwater -> fraction | 9 |
| character -> voice_role | 9 |
| majority_religion -> participat_in | 6 |
| majority_religion -> produc | 5 |
| from -> border | 5 |
| performer -> released_on | 4 |
| performer -> written | 4 |
| performer -> contain | 4 |
| headquarter_location -> rais_money_assist_with | 4 |
| locat_in -> includ | 4 |
| underwater -> locat_in | 4 |
| participat_in -> agre | 3 |
| perform -> releas_in | 3 |
| perform -> released_on | 3 |
| majority_religion -> population | 3 |
| performer -> <empty> | 3 |
| from -> locat_in | 3 |
| from -> resident | 3 |
| underwater -> renown_for | 3 |
| character -> direct | 3 |
| character -> sequel | 3 |
| distribut -> bas_on | 3 |
| distribut -> starr | 3 |
| died_in -> <empty> | 2 |
| educat_at -> born | 2 |
| educat_at -> obtain_degree_through | 2 |
| educat_at -> receiv | 2 |
| in -> mean | 2 |
| in -> used_for | 2 |
| in -> used_in | 2 |
| majority_religion -> appear_in | 2 |
| majority_religion -> censu_year | 2 |
| majority_religion -> found_member | 2 |
| majority_religion -> will_largest_population | 2 |
| performer -> from | 2 |
| performer -> produc | 2 |
| performer -> receiv_praise_for | 2 |
| religion -> <empty> | 2 |

## Source Prior vs V13B

| metric | value |
|---|---:|
| count | 100 |
| source_r5 | 0.6775 |
| v13b_r5 | 0.6825 |
| source_all_gold_at5 | 0.39 |
| v13b_all_gold_at5 | 0.4 |
| gold_count_improved_queries | 1 |
| gold_count_regressed_queries | 0 |

## Representative Cases

### relation_mismatch
- q=1 triple=['Britain', 'participated in', '?x1'] reason=relation_mismatch
- q=1 triple=['France', 'participated in', '?x1'] reason=relation_mismatch
- q=6 triple=['?body', 'located in', '?city'] reason=relation_mismatch
- q=9 triple=['Till dom ensamma', 'performer', '?x1'] reason=relation_mismatch
- q=26 triple=['?x1', 'died in', '?city'] reason=relation_mismatch

### upstream_binding_missing
- q=1 triple=['?x1', 'involved', '?country'] reason=upstream_binding_missing
- q=1 triple=['?country', 'originated', '?x2'] reason=upstream_binding_missing
- q=9 triple=['?x1', 'born on', '?date'] reason=upstream_binding_missing
- q=13 triple=['?x1', 'died in', '?place'] reason=upstream_binding_missing
- q=14 triple=['?basilica', 'located in', '?city'] reason=upstream_binding_missing

### endpoint_missing
- q=1 triple=['?country', 'headquartered in', 'the nobilities commonwealth'] reason=endpoint_missing
- q=1 triple=['?x2', 'top-ranking', 'Warsaw Pact operatives'] reason=endpoint_missing
- q=4 triple=['Lady Godiva', 'birthplace', '?x1'] reason=endpoint_missing
- q=6 triple=['?x1', 'died in', '?city'] reason=endpoint_missing
- q=11 triple=['new coins', 'proclamation of independence by', 'Somali Muslim Ajuran Empire'] reason=endpoint_missing

### candidate_missing
- q=5 triple=['?region1', 'location of', 'Battle of Qurah'] reason=candidate_missing
- q=5 triple=['?region1', 'location of', 'Umm al Maradim'] reason=candidate_missing
- q=7 triple=['?x2', 'headquarters located in', '?city'] reason=candidate_missing
- q=7 triple=['?x1', 'headquarters located in', '?city'] reason=candidate_missing
- q=8 triple=['?x1', 'was born in', '?place'] reason=candidate_missing

### unknown
- q=10 triple=['The Bag or the Bat', 'is a series of', '?series'] reason=unknown
- q=37 triple=['?feature', 'lets the interface replace', 'FireWire'] reason=unknown
- q=38 triple=['Come Away with Me', 'singer', '?x1'] reason=unknown
- q=40 triple=['Bustami', 'birth country', '?country1'] reason=unknown
- q=48 triple=['House of Representatives', 'approves members of', 'Cabinet'] reason=unknown

### bridge_missing
- q=16 triple=['Palau de la Generalitat', 'constructed in', '?city'] reason=bridge_missing
- q=46 triple=['Mizraab', 'country origin', '?x1'] reason=bridge_missing
- q=59 triple=['?x2', 'preached a sermon on Marian devotion', '?x3'] reason=bridge_missing
- q=79 triple=['Darren Carter', 'team', '?x1'] reason=bridge_missing
- q=93 triple=['Nissan', 'manufacturer of', 'Acura Legend'] reason=bridge_missing
