# V13B Evidence Frame Contract Audit: 2wikimultihopqa

This report is diagnostic-only. It separates candidate coverage from gold-document OpenIE frame compliance.

## Summary

| metric | value |
|---|---:|
| retrieval_critical_obligation_count | 220 |
| candidate_coverage:gold_docs_covered | 216 |
| candidate_coverage:candidate_missing | 4 |
| gold_openie_frame:gold_exact_openie_match_available | 108 |
| gold_openie_frame:gold_endpoint_relation_mismatch | 91 |
| gold_openie_frame:gold_relation_present_without_endpoint_binding | 14 |
| gold_openie_frame:gold_endpoint_missing | 7 |

## Interpretation

| item | value |
|---|---|
| gold_openie_exact_frame_rate | 0.490909 |
| endpoint_relation_or_endpoint_missing_rate | 0.445455 |
| main_bottleneck | mixed |
| recommended_next_step | Split by relation family and endpoint shape before changing method behavior. |

## Status By Relation Family

| relation_family | total | gold_endpoint_missing | gold_endpoint_relation_mismatch | gold_exact_openie_match_available | gold_relation_present_without_endpoint_binding |
|---|---:|---:|---:|---:|---:|
| other_relation | 24 | 6 | 17 | 0 | 1 |
| person_relation_role | 16 | 0 | 8 | 7 | 1 |
| place_or_origin_attribute | 13 | 0 | 4 | 9 | 0 |
| temporal_attribute | 94 | 0 | 56 | 29 | 9 |
| work_metadata_role | 73 | 1 | 6 | 63 | 3 |

## Status By Endpoint Shape

| endpoint_shape | total | gold_endpoint_missing | gold_endpoint_relation_mismatch | gold_exact_openie_match_available | gold_relation_present_without_endpoint_binding |
|---|---:|---:|---:|---:|---:|
| all_variable_obligation | 73 | 1 | 66 | 0 | 6 |
| descriptive_bound_endpoint | 11 | 0 | 0 | 11 | 0 |
| long_bound_endpoint | 16 | 0 | 3 | 10 | 3 |
| named_endpoint | 82 | 1 | 14 | 63 | 4 |
| title_qualified_endpoint | 18 | 3 | 4 | 10 | 1 |
| typed_title_endpoint | 20 | 2 | 4 | 14 | 0 |

## Top Query Relation Keys

| relation_key | count |
|---|---:|
| direct | 63 |
| born_on | 47 |
| died_on | 17 |
| born_in | 15 |
| father | 7 |
| died_in | 7 |
| released_on | 6 |
| locat_in | 6 |
| perform | 5 |
| composer | 5 |
| husband | 5 |
| mother | 4 |
| country_origin | 4 |
| nationality | 4 |
| song | 3 |
| from | 3 |
| study_at | 2 |
| spouse | 2 |
| work_at | 2 |
| establish_in | 2 |
| originat_from | 2 |
| paternal_grandfather | 1 |
| wife | 1 |
| died | 1 |
| spouse_s_father | 1 |
| place_death | 1 |
| detain_in | 1 |
| marry | 1 |
| parent | 1 |
| stepmother | 1 |

## Representative Failures

### other_relation::gold_endpoint_missing
- q=2 relation=song endpoint_shape=typed_title_endpoint triple=['song Changed It', 'is a song by', '?x1'] coverage=candidate_missing
- q=13 relation=song endpoint_shape=typed_title_endpoint triple=['song Me And Bobby Mcgee', 'is an song by', '?x1'] coverage=gold_docs_covered
- q=33 relation=song endpoint_shape=title_qualified_endpoint triple=['song Come Dance With Me (Song)', 'is a song by', '?x1'] coverage=candidate_missing
- q=34 relation=work_at endpoint_shape=all_variable_obligation triple=['?x1', 'works at', '?place'] coverage=gold_docs_covered
- q=70 relation=nationality endpoint_shape=title_qualified_endpoint triple=['Christopher Newton (Criminal)', 'nationality', '?x1'] coverage=gold_docs_covered

### other_relation::gold_endpoint_relation_mismatch
- q=7 relation=paternal_grandfather endpoint_shape=named_endpoint triple=['Raghnall Mac Ruaidhrí', 'paternal grandfather', '?x1'] coverage=gold_docs_covered
- q=15 relation=from endpoint_shape=all_variable_obligation triple=['?x1', 'is from', '?country'] coverage=gold_docs_covered
- q=20 relation=study_at endpoint_shape=all_variable_obligation triple=['?x1', 'studied at', '?place'] coverage=gold_docs_covered
- q=23 relation=husband endpoint_shape=named_endpoint triple=['Lisbeth Palme', 'husband', '?x1'] coverage=gold_docs_covered
- q=27 relation=study_at endpoint_shape=all_variable_obligation triple=['?x1', 'studied at', '?place'] coverage=gold_docs_covered

### other_relation::gold_relation_present_without_endpoint_binding
- q=51 relation=marry endpoint_shape=long_bound_endpoint triple=['John Ernest, Duke Of Saxe-Eisenach', 'married to', '?x1'] coverage=gold_docs_covered

### person_relation_role::gold_endpoint_relation_mismatch
- q=21 relation=wife endpoint_shape=named_endpoint triple=['John Middleton Murry', 'wife', '?x1'] coverage=gold_docs_covered
- q=25 relation=spouse_s_father endpoint_shape=named_endpoint triple=['Sisowath Kossamak', "spouse's father", '?x1'] coverage=gold_docs_covered
- q=26 relation=mother endpoint_shape=all_variable_obligation triple=['?x1', 'mother', '?x2'] coverage=gold_docs_covered
- q=27 relation=father endpoint_shape=named_endpoint triple=['Theodore Salisbury Woolsey', 'father', '?x1'] coverage=gold_docs_covered
- q=51 relation=parent endpoint_shape=all_variable_obligation triple=['?x1', 'parent of', '?x2'] coverage=gold_docs_covered

### person_relation_role::gold_relation_present_without_endpoint_binding
- q=59 relation=mother endpoint_shape=named_endpoint triple=['Prince Ferdinand Of Bavaria', 'mother', '?x1'] coverage=gold_docs_covered

### place_or_origin_attribute::gold_endpoint_relation_mismatch
- q=44 relation=detain_in endpoint_shape=all_variable_obligation triple=['?x1', 'was detained in', '?place'] coverage=gold_docs_covered
- q=46 relation=country_origin endpoint_shape=title_qualified_endpoint triple=['Algiers (Film)', 'country of origin', '?x2'] coverage=gold_docs_covered
- q=69 relation=establish_in endpoint_shape=long_bound_endpoint triple=['Museum Of Croatian Archaeological Monuments', 'established in', '?date1'] coverage=gold_docs_covered
- q=69 relation=establish_in endpoint_shape=named_endpoint triple=['Bayernhof Music Museum', 'established in', '?date2'] coverage=gold_docs_covered

### temporal_attribute::gold_endpoint_relation_mismatch
- q=5 relation=born_on endpoint_shape=all_variable_obligation triple=['?x1', 'born on', '?date1'] coverage=gold_docs_covered
- q=5 relation=born_on endpoint_shape=all_variable_obligation triple=['?x2', 'born on', '?date2'] coverage=gold_docs_covered
- q=10 relation=born_in endpoint_shape=all_variable_obligation triple=['?x1', 'was born in', '?place'] coverage=gold_docs_covered
- q=14 relation=died_in endpoint_shape=all_variable_obligation triple=['?x1', 'died in', '?place'] coverage=gold_docs_covered
- q=16 relation=died_on endpoint_shape=all_variable_obligation triple=['?x1', 'died on', '?date'] coverage=gold_docs_covered

### temporal_attribute::gold_relation_present_without_endpoint_binding
- q=2 relation=born_in endpoint_shape=all_variable_obligation triple=['?x1', 'was born in', '?place'] coverage=candidate_missing
- q=6 relation=born_on endpoint_shape=named_endpoint triple=['Andy Summers', 'born on', '?date2'] coverage=gold_docs_covered
- q=18 relation=died_on endpoint_shape=all_variable_obligation triple=['?x1', 'died on', '?date1'] coverage=gold_docs_covered
- q=32 relation=born_on endpoint_shape=all_variable_obligation triple=['?x1', 'born on', '?date1'] coverage=gold_docs_covered
- q=33 relation=born_in endpoint_shape=all_variable_obligation triple=['?x1', 'was born in', '?place'] coverage=candidate_missing

### work_metadata_role::gold_endpoint_missing
- q=36 relation=composer endpoint_shape=title_qualified_endpoint triple=['Inherent Vice (Film)', 'composer', '?x1'] coverage=gold_docs_covered

### work_metadata_role::gold_endpoint_relation_mismatch
- q=22 relation=composer endpoint_shape=typed_title_endpoint triple=['film Billy Elliot', 'composer', '?x1'] coverage=gold_docs_covered
- q=44 relation=perform endpoint_shape=title_qualified_endpoint triple=['song B Boy (Song)', 'performed by', '?x1'] coverage=gold_docs_covered
- q=48 relation=composer endpoint_shape=typed_title_endpoint triple=['film The Straw Hat', 'composer of', '?x1'] coverage=gold_docs_covered
- q=56 relation=composer endpoint_shape=typed_title_endpoint triple=['film Thunder On The Hill', 'composer', '?x1'] coverage=gold_docs_covered
- q=78 relation=perform endpoint_shape=title_qualified_endpoint triple=['song Am I Wrong (Étienne De Crécy Song)', 'performed by', '?x1'] coverage=gold_docs_covered

### work_metadata_role::gold_relation_present_without_endpoint_binding
- q=4 relation=direct endpoint_shape=named_endpoint triple=['Aldri Annet Enn Bråk', 'director', '?x2'] coverage=gold_docs_covered
- q=30 relation=direct endpoint_shape=long_bound_endpoint triple=["I'Ll Be Going Now", 'director', '?x2'] coverage=gold_docs_covered
- q=71 relation=direct endpoint_shape=title_qualified_endpoint triple=['Dancing In The Rain (Film)', 'director', '?x1'] coverage=gold_docs_covered
