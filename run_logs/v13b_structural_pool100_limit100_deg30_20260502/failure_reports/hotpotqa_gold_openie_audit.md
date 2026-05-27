# V13B Gold OpenIE Audit: hotpotqa

## Summary

| metric | value |
|---|---:|
| audited_ungrounded_obligation_count | 98 |

## Gold OpenIE Status

| status | count |
|---|---:|
| gold_endpoint_missing | 40 |
| gold_endpoint_relation_mismatch | 45 |
| gold_exact_openie_match_available | 1 |
| gold_relation_present_without_endpoint_binding | 12 |

## Relation Mismatch Pairs On Gold Docs

| query_relation -> gold_fact_relation | count |
|---|---:|
| second_season -> <empty> | 9 |
| population_rank -> <empty> | 9 |
| profession -> <empty> | 9 |
| attend -> <empty> | 9 |
| 53rd_unit_stat_secretary_treasury -> serv_in | 7 |
| 13th_chief_justice_unit_stat -> serv_in | 7 |
| border_with -> <empty> | 7 |
| screen_debut_in -> appear_in | 6 |
| hous -> <empty> | 6 |
| hous -> hous | 6 |
| company -> <empty> | 6 |
| rank -> <empty> | 6 |
| broadcast_for -> <empty> | 6 |
| country_origin -> voice_cast_includ | 5 |
| bas_in -> original_six_franchise | 5 |
| locat_in -> locat_in | 5 |
| hous -> work_are_part | 5 |
| sampl_from -> topp_in | 5 |
| 53rd_unit_stat_secretary_treasury -> serv_as | 5 |
| 13th_chief_justice_unit_stat -> serv_as | 5 |
| spark -> involv | 4 |
| profession -> direct | 4 |
| inspir -> appear_in | 4 |
| attend -> releas | 4 |
| design -> locat_in | 4 |
| start_with -> known_as | 4 |
| 53rd_unit_stat_secretary_treasury -> appoint | 4 |
| 53rd_unit_stat_secretary_treasury -> <empty> | 4 |
| 13th_chief_justice_unit_stat -> appoint | 4 |
| 13th_chief_justice_unit_stat -> <empty> | 4 |
| spark -> key_event_in | 3 |
| country_origin -> releas_in | 3 |
| locat_in -> former_elevation | 3 |
| locat_in -> part | 3 |
| former_home -> locat_in | 3 |
| locat_at -> locat_in | 3 |
| subsidiary -> operat | 3 |
| count -> <empty> | 3 |
| count -> contribut | 3 |
| second_season -> develop | 3 |

## Representative Cases

### gold_endpoint_relation_mismatch
- q=3 triple=['Marian civil war', 'sparked by', '?x1'] gold=[35, 39]
- q=12 triple=['No Animals Were Harmed', 'focus on', '?x2'] gold=[119, 120]
- q=14 triple=['Summer Wars', 'country of origin', '?x1'] gold=[142, 139]
- q=14 triple=['The Secret of Kells', 'country of origin', '?x2'] gold=[142, 139]
- q=19 triple=['?team', 'based in', '?place'] gold=[192, 188]

### gold_endpoint_missing
- q=3 triple=['?x1', 'helped', 'recently abdicated queen'] gold=[35, 39]
- q=3 triple=['recently abdicated queen', 'imprisoned by', '?x2'] gold=[35, 39]
- q=9 triple=['The 2000 ICC KnockOut Trophy', 'debut of', '?x1'] gold=[85, 91]
- q=9 triple=['?x1', 'is from', 'Jamaica'] gold=[85, 91]
- q=24 triple=['Pro Football Hall of Fame', 'induction year', '2000'] gold=[235, 242]

### gold_relation_present_without_endpoint_binding
- q=5 triple=['Miklos Rozsa', 'worked on', '?screenplay'] gold=[53, 56]
- q=24 triple=['This Experts Network sports analysts', 'inducted into', 'Pro Football Hall of Fame'] gold=[235, 242]
- q=24 triple=['This Experts Network sports analysts', 'played in', 'NFL'] gold=[235, 242]
- q=39 triple=['?x1', 'author', 'Semyon Aranovich Gershgorin'] gold=[386, 390]
- q=40 triple=['?x2', 'originally made for', '?x1'] gold=[400, 395]

### gold_exact_openie_match_available
- q=37 triple=['Eski Imaret Mosque', 'located in', '?city'] gold=[372, 367]
