# V13B Gold OpenIE Audit: musique

## Summary

| metric | value |
|---|---:|
| audited_ungrounded_obligation_count | 161 |

## Gold OpenIE Status

| status | count |
|---|---:|
| candidate_missing | 12 |
| gold_endpoint_missing | 55 |
| gold_endpoint_relation_mismatch | 90 |
| gold_relation_present_without_endpoint_binding | 4 |

## Relation Mismatch Pairs On Gold Docs

| query_relation -> gold_fact_relation | count |
|---|---:|
| died_on -> <empty> | 43 |
| locat_in -> <empty> | 30 |
| competition -> includ | 30 |
| competition -> draft | 30 |
| competition -> attend | 29 |
| died_in -> occurr_dur | 20 |
| occurr_in -> occurr_dur | 20 |
| creat -> <empty> | 20 |
| born_in -> occurr_dur | 20 |
| network -> <empty> | 19 |
| radio_division -> <empty> | 19 |
| agre -> <empty> | 18 |
| involv -> <empty> | 17 |
| originat -> <empty> | 17 |
| locat_in -> locat_in | 15 |
| competition -> locat_in | 15 |
| born_in -> predecessor | 14 |
| died_on -> neighbor | 14 |
| died_on -> discuss | 12 |
| born_in -> compos | 12 |
| co_official_language -> spoken_in | 12 |
| tallest_build -> <empty> | 11 |
| died_in -> occurr_in | 10 |
| occurr_in -> occurr_in | 10 |
| locat_in -> includ | 10 |
| died_on -> influenc | 10 |
| location -> part | 10 |
| location -> includ | 10 |
| born_in -> occurr_in | 10 |
| born_in -> ancestor | 9 |
| held_in -> ancestor | 9 |
| underwater -> fraction | 9 |
| character -> voice_role | 9 |
| play -> voice_role | 9 |
| competition -> position | 8 |
| locat_in -> part | 8 |
| born_in -> <empty> | 8 |
| born_in -> locat_in | 8 |
| location -> rais_money_assist_with | 8 |
| governor -> locat_in | 7 |

## Representative Cases

### gold_endpoint_relation_mismatch
- q=1 triple=['Britain', 'participated in', '?x1'] gold=[35, 29, 22]
- q=1 triple=['?x1', 'involved', '?country'] gold=[35, 29, 22]
- q=1 triple=['?country', 'originated', '?x2'] gold=[35, 29, 22]
- q=9 triple=['Till dom ensamma', 'performer', '?x1'] gold=[184, 192]
- q=9 triple=['?x1', 'born on', '?date'] gold=[184, 192]

### gold_endpoint_missing
- q=1 triple=['France', 'participated in', '?x1'] gold=[35, 29, 22]
- q=1 triple=['?x2', 'top-ranking', 'Warsaw Pact operatives'] gold=[35, 29, 22]
- q=4 triple=['Lady Godiva', 'birthplace', '?x1'] gold=[83, 84]
- q=6 triple=['?x1', 'died in', '?city'] gold=[133, 124, 119, 120]
- q=6 triple=['?body', 'located in', '?city'] gold=[133, 124, 119, 120]

### gold_relation_present_without_endpoint_binding
- q=1 triple=['?country', 'headquartered in', 'the nobilities commonwealth'] gold=[35, 29, 22]
- q=52 triple=['the person who married their half sister in the bible', 'married', '?x1'] gold=[975, 968]
- q=62 triple=['Auctor comes', 'from', '?x1'] gold=[801, 1136, 794]
- q=85 triple=['?place', 'located in', '?entity'] gold=[1506, 1495]

### candidate_missing
- q=5 triple=['?region1', 'location of', 'Battle of Qurah'] gold=[110, 102, 112, 104]
- q=5 triple=['?region1', 'location of', 'Umm al Maradim'] gold=[110, 102, 112, 104]
- q=7 triple=['?x2', 'headquarters located in', '?city'] gold=[148, 158, 146, 147]
- q=7 triple=['?x1', 'headquarters located in', '?city'] gold=[148, 158, 146, 147]
- q=8 triple=['?x1', 'was born in', '?place'] gold=[168, 178, 161]
