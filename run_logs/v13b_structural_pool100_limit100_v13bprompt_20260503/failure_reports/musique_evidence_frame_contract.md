# V13B Evidence Frame Contract Audit: musique

This report is diagnostic-only. It separates candidate coverage from gold-document OpenIE frame compliance.

## Summary

| metric | value |
|---|---:|
| retrieval_critical_obligation_count | 253 |
| candidate_coverage:gold_docs_covered | 237 |
| candidate_coverage:candidate_missing | 16 |
| gold_openie_frame:gold_endpoint_relation_mismatch | 125 |
| gold_openie_frame:gold_endpoint_missing | 93 |
| gold_openie_frame:gold_exact_openie_match_available | 26 |
| gold_openie_frame:gold_relation_present_without_endpoint_binding | 9 |

## Interpretation

| item | value |
|---|---|
| gold_openie_exact_frame_rate | 0.102767 |
| endpoint_relation_or_endpoint_missing_rate | 0.86166 |
| main_bottleneck | gold_openie_schema_contract |
| recommended_next_step | Create and test a Qwen OpenIE evidence-frame schema contract before tuning selector parameters. |

## Status By Relation Family

| relation_family | total | gold_endpoint_missing | gold_endpoint_relation_mismatch | gold_exact_openie_match_available | gold_relation_present_without_endpoint_binding |
|---|---:|---:|---:|---:|---:|
| event_or_competition_role | 7 | 3 | 1 | 1 | 2 |
| office_or_position_role | 1 | 0 | 1 | 0 | 0 |
| other_relation | 123 | 55 | 62 | 6 | 0 |
| person_relation_role | 5 | 2 | 2 | 1 | 0 |
| place_or_origin_attribute | 80 | 24 | 36 | 14 | 6 |
| temporal_attribute | 24 | 5 | 16 | 2 | 1 |
| work_metadata_role | 13 | 4 | 7 | 2 | 0 |

## Status By Endpoint Shape

| endpoint_shape | total | gold_endpoint_missing | gold_endpoint_relation_mismatch | gold_exact_openie_match_available | gold_relation_present_without_endpoint_binding |
|---|---:|---:|---:|---:|---:|
| all_variable_obligation | 105 | 9 | 91 | 0 | 5 |
| descriptive_bound_endpoint | 20 | 13 | 4 | 2 | 1 |
| long_bound_endpoint | 6 | 4 | 1 | 0 | 1 |
| named_endpoint | 122 | 67 | 29 | 24 | 2 |

## Top Query Relation Keys

| relation_key | count |
|---|---:|
| locat_in | 30 |
| born_in | 13 |
| location | 4 |
| died_in | 4 |
| album | 4 |
| died_on | 4 |
| won | 4 |
| headquarter_in | 3 |
| headquarter_locat_in | 3 |
| performer | 3 |
| written | 3 |
| creat | 3 |
| gain_control | 3 |
| held_in | 3 |
| composer | 3 |
| participat_in | 2 |
| proclamation_independence | 2 |
| includ | 2 |
| child | 2 |
| perform | 2 |
| form_in | 2 |
| used_in | 2 |
| era | 2 |
| later_known_as | 2 |
| set_in | 2 |
| husband | 2 |
| from | 2 |
| co_official_language | 2 |
| play | 2 |
| team | 2 |
| version | 2 |
| maximum_load_drawn | 2 |
| adjacent | 2 |
| determin_rul | 2 |
| manufacturer | 2 |
| open | 2 |
| scor_goal_in | 1 |
| get_sign | 1 |
| involv | 1 |
| originat | 1 |

## Representative Failures

### event_or_competition_role::gold_endpoint_missing
- q=10 relation=season endpoint_shape=named_endpoint triple=['?series', 'has season', '5'] coverage=gold_docs_covered
- q=79 relation=won endpoint_shape=named_endpoint triple=['?x2', 'is the winner of', '1894-95 FA Cup'] coverage=gold_docs_covered
- q=79 relation=won endpoint_shape=named_endpoint triple=['?x2', 'won the', 'FA Cup'] coverage=gold_docs_covered

### event_or_competition_role::gold_endpoint_relation_mismatch
- q=39 relation=competition endpoint_shape=all_variable_obligation triple=['?league', 'has competition', '?competition'] coverage=gold_docs_covered

### event_or_competition_role::gold_relation_present_without_endpoint_binding
- q=39 relation=draft endpoint_shape=named_endpoint triple=['1999 draft', 'drafted by', '?team'] coverage=gold_docs_covered
- q=63 relation=won endpoint_shape=long_bound_endpoint triple=['1979-80 European Cup winner', 'won', '?team'] coverage=gold_docs_covered

### office_or_position_role::gold_endpoint_relation_mismatch
- q=40 relation=president endpoint_shape=all_variable_obligation triple=['?x1', 'was president of', '?country2'] coverage=gold_docs_covered

### other_relation::gold_endpoint_missing
- q=0 relation=get_sign endpoint_shape=named_endpoint triple=['?x2', 'get signed by', 'Barcelona'] coverage=gold_docs_covered
- q=1 relation=top_rank endpoint_shape=named_endpoint triple=['?x2', 'top-ranking', 'Warsaw Pact operatives'] coverage=gold_docs_covered
- q=6 relation=empti_into endpoint_shape=named_endpoint triple=['?body', 'empties into', 'Gulf of Mexico'] coverage=gold_docs_covered
- q=10 relation=seri endpoint_shape=descriptive_bound_endpoint triple=['The Bag or the Bat', 'is a series of', '?series'] coverage=gold_docs_covered
- q=11 relation=proclamation_independence endpoint_shape=named_endpoint triple=['new coins', 'proclamation of independence by', 'Somali Muslim Ajuran Empire'] coverage=gold_docs_covered

### other_relation::gold_endpoint_relation_mismatch
- q=1 relation=involv endpoint_shape=all_variable_obligation triple=['?x1', 'involved', '?country'] coverage=gold_docs_covered
- q=1 relation=originat endpoint_shape=all_variable_obligation triple=['?country', 'originated', '?x2'] coverage=gold_docs_covered
- q=8 relation=album endpoint_shape=named_endpoint triple=['III', 'is an album by', '?x1'] coverage=candidate_missing
- q=9 relation=performer endpoint_shape=named_endpoint triple=['Till dom ensamma', 'performer', '?x1'] coverage=gold_docs_covered
- q=14 relation=governor endpoint_shape=all_variable_obligation triple=['?city', 'governor of', '?governor'] coverage=gold_docs_covered

### person_relation_role::gold_endpoint_missing
- q=36 relation=child endpoint_shape=descriptive_bound_endpoint triple=['the president', 'child', '?x1'] coverage=gold_docs_covered
- q=86 relation=daughter endpoint_shape=named_endpoint triple=['Marty MCFly', 'has daughter', '?x1'] coverage=gold_docs_covered

### person_relation_role::gold_endpoint_relation_mismatch
- q=24 relation=child endpoint_shape=all_variable_obligation triple=['?navigator', 'child of', '?child'] coverage=gold_docs_covered
- q=58 relation=spouse endpoint_shape=all_variable_obligation triple=['?x1', 'spouse', '?x2'] coverage=gold_docs_covered

### place_or_origin_attribute::gold_endpoint_missing
- q=0 relation=scor_goal_in endpoint_shape=named_endpoint triple=['Messi', 'scored goals in', 'Copa del Rey'] coverage=gold_docs_covered
- q=1 relation=participat_in endpoint_shape=named_endpoint triple=['Britain', 'participated in', '?x1'] coverage=gold_docs_covered
- q=1 relation=participat_in endpoint_shape=named_endpoint triple=['France', 'participated in', '?x1'] coverage=gold_docs_covered
- q=5 relation=location endpoint_shape=named_endpoint triple=['?region1', 'location of', 'Battle of Qurah'] coverage=candidate_missing
- q=5 relation=location endpoint_shape=named_endpoint triple=['?region1', 'location of', 'Umm al Maradim'] coverage=candidate_missing

### place_or_origin_attribute::gold_endpoint_relation_mismatch
- q=2 relation=locat_in endpoint_shape=all_variable_obligation triple=['?x1', 'located in', '?county'] coverage=gold_docs_covered
- q=7 relation=headquarter_locat_in endpoint_shape=all_variable_obligation triple=['?x2', 'headquarters located in', '?city'] coverage=candidate_missing
- q=7 relation=headquarter_locat_in endpoint_shape=all_variable_obligation triple=['?x1', 'headquarters located in', '?city'] coverage=candidate_missing
- q=14 relation=locat_in endpoint_shape=all_variable_obligation triple=['?basilica', 'located in', '?city'] coverage=gold_docs_covered
- q=16 relation=locat_in endpoint_shape=all_variable_obligation triple=['?region', 'located in', '?state'] coverage=gold_docs_covered

### place_or_origin_attribute::gold_relation_present_without_endpoint_binding
- q=1 relation=headquarter_in endpoint_shape=descriptive_bound_endpoint triple=['?country', 'headquartered in', 'the nobilities commonwealth'] coverage=gold_docs_covered
- q=6 relation=locat_in endpoint_shape=all_variable_obligation triple=['?body', 'located in', '?city'] coverage=gold_docs_covered
- q=30 relation=locat_in endpoint_shape=all_variable_obligation triple=['?city', 'located in', '?state'] coverage=gold_docs_covered
- q=50 relation=headquarter_in endpoint_shape=named_endpoint triple=['Warsaw Pact', 'headquartered in', '?country1'] coverage=gold_docs_covered
- q=72 relation=locat_in endpoint_shape=all_variable_obligation triple=['?city', 'located in', '?county'] coverage=candidate_missing

### temporal_attribute::gold_endpoint_missing
- q=4 relation=born_in endpoint_shape=named_endpoint triple=['Lady Godiva', 'birthplace', '?x1'] coverage=gold_docs_covered
- q=6 relation=died_in endpoint_shape=all_variable_obligation triple=['?x1', 'died in', '?city'] coverage=gold_docs_covered
- q=16 relation=died_in endpoint_shape=named_endpoint triple=['Martin', 'died in', '?city'] coverage=gold_docs_covered
- q=40 relation=birth_country endpoint_shape=named_endpoint triple=['Bustami', 'birth country', '?country1'] coverage=gold_docs_covered
- q=52 relation=died_on endpoint_shape=named_endpoint triple=['sarah', 'died on', '?date2'] coverage=gold_docs_covered

### temporal_attribute::gold_endpoint_relation_mismatch
- q=8 relation=born_in endpoint_shape=all_variable_obligation triple=['?x1', 'was born in', '?place'] coverage=candidate_missing
- q=9 relation=born_on endpoint_shape=all_variable_obligation triple=['?x1', 'born on', '?date'] coverage=gold_docs_covered
- q=12 relation=born_in endpoint_shape=all_variable_obligation triple=['?x1', 'was born in', '?place'] coverage=candidate_missing
- q=13 relation=died_in endpoint_shape=all_variable_obligation triple=['?x1', 'died in', '?place'] coverage=gold_docs_covered
- q=26 relation=died_in endpoint_shape=all_variable_obligation triple=['?x1', 'died in', '?city'] coverage=gold_docs_covered

### temporal_attribute::gold_relation_present_without_endpoint_binding
- q=30 relation=born_in endpoint_shape=all_variable_obligation triple=['?x1', 'was born in', '?state'] coverage=gold_docs_covered

### work_metadata_role::gold_endpoint_missing
- q=22 relation=written_about endpoint_shape=descriptive_bound_endpoint triple=['the rioting being a dividing factor in Birmingham', 'written about by', '?x1'] coverage=gold_docs_covered
- q=38 relation=written endpoint_shape=named_endpoint triple=['?x1', 'wrote', 'Turn Me On'] coverage=gold_docs_covered
- q=81 relation=actor_who_play_jarvi endpoint_shape=named_endpoint triple=['Avengers Age of Ultron', 'actor who plays Jarvis', '?x1'] coverage=gold_docs_covered
- q=98 relation=composer endpoint_shape=long_bound_endpoint triple=['Concerto in C Major Op 3 6', 'composer', '?x1'] coverage=gold_docs_covered

### work_metadata_role::gold_endpoint_relation_mismatch
- q=27 relation=perform endpoint_shape=named_endpoint triple=['Attics To Eden', 'performed by', '?x1'] coverage=gold_docs_covered
- q=45 relation=star endpoint_shape=long_bound_endpoint triple=['Sous les pieds des femmes', 'star', '?x1'] coverage=gold_docs_covered
- q=57 relation=composer endpoint_shape=named_endpoint triple=['Scanderbeg', 'composer of', '?x1'] coverage=gold_docs_covered
- q=78 relation=character endpoint_shape=named_endpoint triple=['Shrek 2', 'has character', '?x1'] coverage=gold_docs_covered
- q=80 relation=composer endpoint_shape=named_endpoint triple=['La Silvia', 'composer', '?x1'] coverage=gold_docs_covered
