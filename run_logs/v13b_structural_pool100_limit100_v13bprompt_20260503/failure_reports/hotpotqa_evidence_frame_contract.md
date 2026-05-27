# V13B Evidence Frame Contract Audit: hotpotqa

This report is diagnostic-only. It separates candidate coverage from gold-document OpenIE frame compliance.

## Summary

| metric | value |
|---|---:|
| retrieval_critical_obligation_count | 216 |
| candidate_coverage:gold_docs_covered | 216 |
| gold_openie_frame:gold_endpoint_relation_mismatch | 102 |
| gold_openie_frame:gold_endpoint_missing | 66 |
| gold_openie_frame:gold_exact_openie_match_available | 28 |
| gold_openie_frame:gold_relation_present_without_endpoint_binding | 20 |

## Interpretation

| item | value |
|---|---|
| gold_openie_exact_frame_rate | 0.12963 |
| endpoint_relation_or_endpoint_missing_rate | 0.777778 |
| main_bottleneck | gold_openie_schema_contract |
| recommended_next_step | Create and test a Qwen OpenIE evidence-frame schema contract before tuning selector parameters. |

## Status By Relation Family

| relation_family | total | gold_endpoint_missing | gold_endpoint_relation_mismatch | gold_exact_openie_match_available | gold_relation_present_without_endpoint_binding |
|---|---:|---:|---:|---:|---:|
| event_or_competition_role | 5 | 0 | 2 | 2 | 1 |
| office_or_position_role | 4 | 2 | 2 | 0 | 0 |
| other_relation | 120 | 41 | 68 | 3 | 8 |
| person_relation_role | 2 | 0 | 1 | 0 | 1 |
| place_or_origin_attribute | 48 | 10 | 21 | 14 | 3 |
| temporal_attribute | 17 | 5 | 4 | 4 | 4 |
| work_metadata_role | 20 | 8 | 4 | 5 | 3 |

## Status By Endpoint Shape

| endpoint_shape | total | gold_endpoint_missing | gold_endpoint_relation_mismatch | gold_exact_openie_match_available | gold_relation_present_without_endpoint_binding |
|---|---:|---:|---:|---:|---:|
| all_variable_obligation | 49 | 7 | 39 | 0 | 3 |
| descriptive_bound_endpoint | 26 | 12 | 9 | 2 | 3 |
| long_bound_endpoint | 12 | 1 | 6 | 3 | 2 |
| named_endpoint | 129 | 46 | 48 | 23 | 12 |

## Top Query Relation Keys

| relation_key | count |
|---|---:|
| written | 12 |
| locat_in | 12 |
| year | 6 |
| born_on | 6 |
| profession | 4 |
| born_in | 3 |
| work_on | 3 |
| won | 3 |
| in | 3 |
| appear_in | 3 |
| genre | 3 |
| produc_in | 2 |
| focu_on | 2 |
| country_origin | 2 |
| tallest_in | 2 |
| found_as | 2 |
| number_act | 2 |
| count | 2 |
| headquarter_in | 2 |
| character_in | 2 |
| direct | 2 |
| company | 2 |
| form_in | 2 |
| shar_its_title_with | 2 |
| design | 2 |
| production_company | 2 |
| starr_in | 1 |
| pass | 1 |
| spark | 1 |
| help | 1 |
| imprison | 1 |
| begin_with_interchange_at | 1 |
| debut | 1 |
| from | 1 |
| civil_parish | 1 |
| nam_nba_final_most_valuable_player | 1 |
| star_as_mark_cohen_in | 1 |
| succeed | 1 |
| captain | 1 |
| bas_in | 1 |

## Representative Failures

### event_or_competition_role::gold_endpoint_relation_mismatch
- q=44 relation=second_season endpoint_shape=all_variable_obligation triple=['?series', 'second season', '?season'] coverage=gold_docs_covered
- q=82 relation=won endpoint_shape=named_endpoint triple=['Norm Coleman', 'won', '?election'] coverage=gold_docs_covered

### event_or_competition_role::gold_relation_present_without_endpoint_binding
- q=55 relation=season endpoint_shape=descriptive_bound_endpoint triple=['The Simpsons', 'season', 'seventh season'] coverage=gold_docs_covered

### office_or_position_role::gold_endpoint_missing
- q=46 relation=serv_as_residence_for endpoint_shape=named_endpoint triple=['?home', 'serves as the residence for', 'Mayor of New York'] coverage=gold_docs_covered
- q=73 relation=former_president endpoint_shape=named_endpoint triple=['ABC television', 'former president', '?x1'] coverage=gold_docs_covered

### office_or_position_role::gold_endpoint_relation_mismatch
- q=92 relation=53rd_unit_stat_secretary_treasury endpoint_shape=all_variable_obligation triple=['?x1', '53rd United States Secretary of the Treasury', '?x2'] coverage=gold_docs_covered
- q=92 relation=13th_chief_justice_unit_stat endpoint_shape=all_variable_obligation triple=['?x1', '13th Chief Justice of the United States', '?x3'] coverage=gold_docs_covered

### other_relation::gold_endpoint_missing
- q=3 relation=help endpoint_shape=descriptive_bound_endpoint triple=['?x1', 'helped', 'recently abdicated queen'] coverage=gold_docs_covered
- q=3 relation=imprison endpoint_shape=descriptive_bound_endpoint triple=['recently abdicated queen', 'imprisoned by', '?x2'] coverage=gold_docs_covered
- q=5 relation=work_on endpoint_shape=named_endpoint triple=['Miklos Rozsa', 'worked on', '?screenplay'] coverage=gold_docs_covered
- q=9 relation=debut endpoint_shape=descriptive_bound_endpoint triple=['The 2000 ICC KnockOut Trophy', 'debut of', '?x1'] coverage=gold_docs_covered
- q=16 relation=nam_nba_final_most_valuable_player endpoint_shape=named_endpoint triple=['Golden State NBA player', 'named NBA Finals Most Valuable Player', '?year'] coverage=gold_docs_covered

### other_relation::gold_endpoint_relation_mismatch
- q=3 relation=spark endpoint_shape=named_endpoint triple=['Marian civil war', 'sparked by', '?x1'] coverage=gold_docs_covered
- q=5 relation=work_on endpoint_shape=named_endpoint triple=['Edward Carfagno', 'worked on', '?screenplay'] coverage=gold_docs_covered
- q=7 relation=begin_with_interchange_at endpoint_shape=named_endpoint triple=['Bethpage State Parkway', 'begins with an interchange at', '?x1'] coverage=gold_docs_covered
- q=9 relation=from endpoint_shape=named_endpoint triple=['?x1', 'is from', 'Jamaica'] coverage=gold_docs_covered
- q=10 relation=civil_parish endpoint_shape=all_variable_obligation triple=['?x1', 'has civil parish of', '?x2'] coverage=gold_docs_covered

### other_relation::gold_relation_present_without_endpoint_binding
- q=2 relation=pass endpoint_shape=descriptive_bound_endpoint triple=['The Distribution of Industry act', 'passed by', '?x1'] coverage=gold_docs_covered
- q=24 relation=induct_into endpoint_shape=descriptive_bound_endpoint triple=['This Experts Network sports analysts', 'inducted into', 'Pro Football Hall of Fame'] coverage=gold_docs_covered
- q=40 relation=acquir endpoint_shape=named_endpoint triple=['RadioShack', 'acquired by', '?x1'] coverage=gold_docs_covered
- q=51 relation=play_for endpoint_shape=named_endpoint triple=['Chris Summers', 'plays for', '?team'] coverage=gold_docs_covered
- q=55 relation=episode endpoint_shape=named_endpoint triple=['seventh season', 'episode', 'fourth episode'] coverage=gold_docs_covered

### person_relation_role::gold_endpoint_relation_mismatch
- q=98 relation=father endpoint_shape=named_endpoint triple=['Joe Buck', 'father', '?x1'] coverage=gold_docs_covered

### person_relation_role::gold_relation_present_without_endpoint_binding
- q=65 relation=mother endpoint_shape=named_endpoint triple=['Govinda', 'mother', '?x1'] coverage=gold_docs_covered

### place_or_origin_attribute::gold_endpoint_missing
- q=18 relation=tallest_in endpoint_shape=named_endpoint triple=['One Raffles Place', 'tallest in', 'world outside North America'] coverage=gold_docs_covered
- q=18 relation=tallest_in endpoint_shape=named_endpoint triple=['One Raffles Place', 'tallest in', 'city of Singapore'] coverage=gold_docs_covered
- q=24 relation=play_in endpoint_shape=descriptive_bound_endpoint triple=['This Experts Network sports analysts', 'played in', 'NFL'] coverage=gold_docs_covered
- q=41 relation=wrote_in endpoint_shape=all_variable_obligation triple=['?x1', 'wrote in', '?place'] coverage=gold_docs_covered
- q=46 relation=painting_hang_in endpoint_shape=named_endpoint triple=['Stockely Webster', 'has paintings hanging in', '?home'] coverage=gold_docs_covered

### place_or_origin_attribute::gold_endpoint_relation_mismatch
- q=0 relation=starr_in endpoint_shape=descriptive_bound_endpoint triple=['The Newcomers', 'starred in', '?x1'] coverage=gold_docs_covered
- q=4 relation=locat_in endpoint_shape=all_variable_obligation triple=['?town', 'located in', '?county'] coverage=gold_docs_covered
- q=6 relation=produc_in endpoint_shape=descriptive_bound_endpoint triple=['The Apple Dumpling Gang', 'produced in', '?date1'] coverage=gold_docs_covered
- q=6 relation=produc_in endpoint_shape=long_bound_endpoint triple=['Something Wicked This Way Comes', 'produced in', '?date2'] coverage=gold_docs_covered
- q=14 relation=country_origin endpoint_shape=descriptive_bound_endpoint triple=['The Secret of Kells', 'country of origin', '?x2'] coverage=gold_docs_covered

### place_or_origin_attribute::gold_relation_present_without_endpoint_binding
- q=21 relation=locat_in endpoint_shape=named_endpoint triple=['Six Flags Great America', 'located in', 'Gurnee, Illinois'] coverage=gold_docs_covered
- q=27 relation=film_in endpoint_shape=long_bound_endpoint triple=['?x1', 'filmed in', 'Winter Palace of the Russian State Hermitage Museum'] coverage=gold_docs_covered
- q=33 relation=took_place_in endpoint_shape=named_endpoint triple=['Mayweather-Ortiz fight', 'took place in', '?x1'] coverage=gold_docs_covered

### temporal_attribute::gold_endpoint_missing
- q=65 relation=born_in endpoint_shape=all_variable_obligation triple=['?x1', 'was born in', '?place'] coverage=gold_docs_covered
- q=65 relation=year endpoint_shape=named_endpoint triple=['?film', 'year', '1944'] coverage=gold_docs_covered
- q=69 relation=year endpoint_shape=named_endpoint triple=['?film', 'year', '2006'] coverage=gold_docs_covered
- q=78 relation=year endpoint_shape=named_endpoint triple=['?x1', 'year', '1959'] coverage=gold_docs_covered
- q=82 relation=year endpoint_shape=named_endpoint triple=['?election', 'year', '2017'] coverage=gold_docs_covered

### temporal_attribute::gold_endpoint_relation_mismatch
- q=14 relation=year endpoint_shape=named_endpoint triple=['Summer Wars', 'year', '2009'] coverage=gold_docs_covered
- q=14 relation=year endpoint_shape=descriptive_bound_endpoint triple=['The Secret of Kells', 'year', '2009'] coverage=gold_docs_covered
- q=24 relation=induction_year endpoint_shape=long_bound_endpoint triple=['Pro Football Hall of Fame', 'induction year', '2000'] coverage=gold_docs_covered
- q=81 relation=born_on endpoint_shape=all_variable_obligation triple=['?x1', 'born on', '?year'] coverage=gold_docs_covered

### temporal_attribute::gold_relation_present_without_endpoint_binding
- q=58 relation=born_on endpoint_shape=all_variable_obligation triple=['?x1', 'born on', '?date'] coverage=gold_docs_covered
- q=65 relation=died_on endpoint_shape=all_variable_obligation triple=['?x1', 'died on', '?date'] coverage=gold_docs_covered
- q=93 relation=born_on endpoint_shape=all_variable_obligation triple=['?x1', 'born on', '?month'] coverage=gold_docs_covered
- q=97 relation=born_in endpoint_shape=named_endpoint triple=['Ronald Engert', 'born in', '?x1'] coverage=gold_docs_covered

### work_metadata_role::gold_endpoint_missing
- q=39 relation=written endpoint_shape=named_endpoint triple=['?x1', 'author', 'Semyon Aranovich Gershgorin'] coverage=gold_docs_covered
- q=39 relation=written endpoint_shape=named_endpoint triple=['?x2', 'author', 'Pavel Alexandrov'] coverage=gold_docs_covered
- q=41 relation=written endpoint_shape=named_endpoint triple=['Bring Me Sunshine', 'written by', '?x1'] coverage=gold_docs_covered
- q=41 relation=written endpoint_shape=named_endpoint triple=['?x1', 'wrote', 'Bring Me Sunshine'] coverage=gold_docs_covered
- q=66 relation=play_character endpoint_shape=named_endpoint triple=['Kristin Davis', 'played the character', 'Charlotte York Goldenblatt'] coverage=gold_docs_covered

### work_metadata_role::gold_endpoint_relation_mismatch
- q=39 relation=written endpoint_shape=named_endpoint triple=['Semyon Aranovich Gershgorin', 'wrote', '?x1'] coverage=gold_docs_covered
- q=39 relation=written endpoint_shape=named_endpoint triple=['Pavel Alexandrov', 'wrote', '?x2'] coverage=gold_docs_covered
- q=48 relation=title_character_play endpoint_shape=named_endpoint triple=['Ellie Parker', 'title character played by', '?x1'] coverage=gold_docs_covered
- q=53 relation=produc_for endpoint_shape=named_endpoint triple=['?series', 'produced for', 'Disney Channel'] coverage=gold_docs_covered

### work_metadata_role::gold_relation_present_without_endpoint_binding
- q=12 relation=written endpoint_shape=named_endpoint triple=['Peter Laufer', 'wrote', 'Forbidden Creatures'] coverage=gold_docs_covered
- q=17 relation=written endpoint_shape=long_bound_endpoint triple=['Without you: A memoir of love, loss, and the Musical Rent', 'written by', '?x1'] coverage=gold_docs_covered
- q=56 relation=written endpoint_shape=named_endpoint triple=['La Machine a ecirire', 'written by', '?x1'] coverage=gold_docs_covered
