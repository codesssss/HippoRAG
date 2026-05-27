# V13B Grounding Failure Report: hotpotqa

## Summary

| metric | value |
|---|---:|
| retrieval_critical_ungrounded | 49 |
| classified_failure_count | 98 |
| unknown_fraction | 0.030612 |

## Failure Types

| failure_type | count |
|---|---:|
| bridge_missing | 8 |
| endpoint_missing | 37 |
| relation_mismatch | 28 |
| unknown | 3 |
| upstream_binding_missing | 22 |

## Relation Mismatch Pairs

| query_relation -> fact_relation | count |
|---|---:|
| profession -> <empty> | 9 |
| spark -> involv | 6 |
| screen_debut_in -> appear_in | 6 |
| country_origin -> voice_cast_includ | 5 |
| launch_for_first_time_in -> includ | 5 |
| sampl_from -> topp_in | 5 |
| launch_for_first_time_in -> assign_for | 4 |
| profession -> direct | 4 |
| spark -> key_event_in | 3 |
| focu_on -> publish_in | 3 |
| country_origin -> won | 3 |
| country_origin -> premier_on | 3 |
| country_origin -> releas_in | 3 |
| subsidiary -> operat | 3 |
| screen_debut_in -> <empty> | 3 |
| profession -> born_on | 3 |
| producer -> starr | 3 |
| focu_on -> <empty> | 2 |
| focu_on -> follow | 2 |
| focu_on -> part | 2 |
| country_origin -> <empty> | 2 |
| country_origin -> animat | 2 |
| country_origin -> nominat_for | 2 |
| country_origin -> set_in | 2 |
| subsidiary -> bas_in | 2 |
| launch_for_first_time_in -> <empty> | 2 |
| start_sing_in -> <empty> | 2 |
| start_sing_in -> open_in_concert_for | 2 |
| start_sing_in -> sang_at | 2 |
| start_sing_in -> sold_recording | 2 |
| screen_debut_in -> title_character | 2 |
| profession -> name | 2 |
| born_on -> tune | 2 |
| profession -> father | 2 |
| profession -> part | 2 |
| sampl_from -> featur_sample_from | 2 |
| producer -> produc | 2 |
| father -> <empty> | 2 |
| father -> lead_play_play_announcer_for | 2 |
| father -> not_serv_as_play_play_announcer_for | 2 |

## Source Prior vs V13B

| metric | value |
|---|---:|
| count | 100 |
| source_r5 | 0.93 |
| v13b_r5 | 0.93 |
| source_all_gold_at5 | 0.87 |
| v13b_all_gold_at5 | 0.87 |
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
- q=24 triple=['This Experts Network sports analysts', 'inducted into', 'Pro Football Hall of Fame'] reason=endpoint_missing

### bridge_missing
- q=19 triple=['?team', 'based in', '?place'] reason=bridge_missing
- q=37 triple=['Eski Imaret Mosque', 'located in', '?city'] reason=bridge_missing
- q=50 triple=['?x1', 'edited by', 'Andrew Anglin'] reason=bridge_missing
- q=65 triple=['?x1', 'acted in', '?film'] reason=bridge_missing
- q=81 triple=['Johnny Majors', 'defeated by', '?x1'] reason=bridge_missing

### upstream_binding_missing
- q=20 triple=['?chain', 'located in', '?part'] reason=upstream_binding_missing
- q=21 triple=['?x1', 'former home', '?location'] reason=upstream_binding_missing
- q=21 triple=['?location', 'is located at', '?intersection'] reason=upstream_binding_missing
- q=29 triple=['?x1', 'notable for having', '?x2'] reason=upstream_binding_missing
- q=29 triple=['?x2', 'hacking organization with a user base of over 1,800,000', '?x3'] reason=upstream_binding_missing

### unknown
- q=33 triple=['?x1', 'is a former', 'MGM Grand Garden Special Events Center'] reason=unknown
- q=65 triple=['?film', 'genre', 'Bollywood'] reason=unknown
- q=78 triple=['?x2', 'shares its title with', '?x1'] reason=unknown
