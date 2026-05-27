# V13B Evidence Frame Contract Audit: 2wikimultihopqa

This report is diagnostic-only. It separates candidate coverage from gold-document OpenIE frame compliance.

## Summary

| metric | value |
|---|---:|
| retrieval_critical_obligation_count | 220 |
| candidate_coverage:gold_docs_covered | 218 |
| candidate_coverage:candidate_missing | 2 |
| gold_openie_frame:gold_endpoint_relation_mismatch | 107 |
| gold_openie_frame:gold_exact_openie_match_available | 94 |
| gold_openie_frame:gold_relation_present_without_endpoint_binding | 14 |
| gold_openie_frame:gold_endpoint_missing | 5 |

## Interpretation

| item | value |
|---|---|
| gold_openie_exact_frame_rate | 0.427273 |
| endpoint_relation_or_endpoint_missing_rate | 0.509091 |
| main_bottleneck | gold_openie_schema_contract |
| recommended_next_step | Create and test a Qwen OpenIE evidence-frame schema contract before tuning selector parameters. |

## Status By Relation Family

| relation_family | total | gold_endpoint_missing | gold_endpoint_relation_mismatch | gold_exact_openie_match_available | gold_relation_present_without_endpoint_binding |
|---|---:|---:|---:|---:|---:|
| other_relation | 24 | 3 | 18 | 2 | 1 |
| person_relation_role | 16 | 2 | 12 | 1 | 1 |
| place_or_origin_attribute | 13 | 0 | 7 | 6 | 0 |
| temporal_attribute | 94 | 0 | 61 | 25 | 8 |
| work_metadata_role | 73 | 0 | 9 | 60 | 4 |

## Status By Endpoint Shape

| endpoint_shape | total | gold_endpoint_missing | gold_endpoint_relation_mismatch | gold_exact_openie_match_available | gold_relation_present_without_endpoint_binding |
|---|---:|---:|---:|---:|---:|
| all_variable_obligation | 73 | 1 | 66 | 0 | 6 |
| descriptive_bound_endpoint | 11 | 0 | 1 | 9 | 1 |
| long_bound_endpoint | 16 | 0 | 4 | 10 | 2 |
| named_endpoint | 82 | 3 | 21 | 54 | 4 |
| title_qualified_endpoint | 18 | 1 | 8 | 8 | 1 |
| typed_title_endpoint | 20 | 0 | 7 | 13 | 0 |

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
- q=23 relation=husband endpoint_shape=named_endpoint triple=['Lisbeth Palme', 'husband', '?x1'] coverage=gold_docs_covered
- q=33 relation=song endpoint_shape=title_qualified_endpoint triple=['song Come Dance With Me (Song)', 'is a song by', '?x1'] coverage=candidate_missing
- q=34 relation=work_at endpoint_shape=all_variable_obligation triple=['?x1', 'works at', '?place'] coverage=gold_docs_covered

### other_relation::gold_endpoint_relation_mismatch
- q=7 relation=paternal_grandfather endpoint_shape=named_endpoint triple=['Raghnall Mac Ruaidhrí', 'paternal grandfather', '?x1'] coverage=gold_docs_covered
- q=13 relation=song endpoint_shape=typed_title_endpoint triple=['song Me And Bobby Mcgee', 'is an song by', '?x1'] coverage=gold_docs_covered
- q=15 relation=from endpoint_shape=all_variable_obligation triple=['?x1', 'is from', '?country'] coverage=gold_docs_covered
- q=20 relation=study_at endpoint_shape=all_variable_obligation triple=['?x1', 'studied at', '?place'] coverage=gold_docs_covered
- q=27 relation=study_at endpoint_shape=all_variable_obligation triple=['?x1', 'studied at', '?place'] coverage=gold_docs_covered

### other_relation::gold_relation_present_without_endpoint_binding
- q=51 relation=marry endpoint_shape=long_bound_endpoint triple=['John Ernest, Duke Of Saxe-Eisenach', 'married to', '?x1'] coverage=gold_docs_covered

### person_relation_role::gold_endpoint_missing
- q=0 relation=mother endpoint_shape=named_endpoint triple=['Lothair Ii', 'mother', '?x1'] coverage=gold_docs_covered
- q=59 relation=mother endpoint_shape=named_endpoint triple=['Prince Ferdinand Of Bavaria', 'mother', '?x1'] coverage=gold_docs_covered

### person_relation_role::gold_endpoint_relation_mismatch
- q=14 relation=father endpoint_shape=named_endpoint triple=['Maurice, Prince Of Orange', 'father', '?x1'] coverage=gold_docs_covered
- q=15 relation=father endpoint_shape=title_qualified_endpoint triple=['Aleksander Koniecpolski (1620–1659)', 'father', '?x1'] coverage=gold_docs_covered
- q=20 relation=father endpoint_shape=named_endpoint triple=['Coulson Wallop', 'father', '?x1'] coverage=gold_docs_covered
- q=21 relation=wife endpoint_shape=named_endpoint triple=['John Middleton Murry', 'wife', '?x1'] coverage=gold_docs_covered
- q=25 relation=spouse_s_father endpoint_shape=named_endpoint triple=['Sisowath Kossamak', "spouse's father", '?x1'] coverage=gold_docs_covered

### person_relation_role::gold_relation_present_without_endpoint_binding
- q=65 relation=father endpoint_shape=named_endpoint triple=['Abdul-Aziz Bin Muhammad', 'father', '?x1'] coverage=gold_docs_covered

### place_or_origin_attribute::gold_endpoint_relation_mismatch
- q=44 relation=detain_in endpoint_shape=all_variable_obligation triple=['?x1', 'was detained in', '?place'] coverage=gold_docs_covered
- q=46 relation=country_origin endpoint_shape=named_endpoint triple=['Naked Tango', 'country of origin', '?x1'] coverage=gold_docs_covered
- q=46 relation=country_origin endpoint_shape=title_qualified_endpoint triple=['Algiers (Film)', 'country of origin', '?x2'] coverage=gold_docs_covered
- q=68 relation=country_origin endpoint_shape=long_bound_endpoint triple=['Wizards Of The Lost Kingdom', 'country of origin', '?x1'] coverage=gold_docs_covered
- q=68 relation=country_origin endpoint_shape=title_qualified_endpoint triple=['Final Exam (1981 Film)', 'country of origin', '?x2'] coverage=gold_docs_covered

### temporal_attribute::gold_endpoint_relation_mismatch
- q=1 relation=released_on endpoint_shape=named_endpoint triple=['Aas Ka Panchhi', 'release date', '?date1'] coverage=gold_docs_covered
- q=1 relation=released_on endpoint_shape=named_endpoint triple=['Phoolwari', 'release date', '?date2'] coverage=gold_docs_covered
- q=2 relation=born_in endpoint_shape=all_variable_obligation triple=['?x1', 'was born in', '?place'] coverage=gold_docs_covered
- q=5 relation=born_on endpoint_shape=all_variable_obligation triple=['?x2', 'born on', '?date2'] coverage=gold_docs_covered
- q=10 relation=born_in endpoint_shape=all_variable_obligation triple=['?x1', 'was born in', '?place'] coverage=gold_docs_covered

### temporal_attribute::gold_relation_present_without_endpoint_binding
- q=5 relation=born_on endpoint_shape=all_variable_obligation triple=['?x1', 'born on', '?date1'] coverage=gold_docs_covered
- q=6 relation=born_on endpoint_shape=named_endpoint triple=['Andy Summers', 'born on', '?date2'] coverage=gold_docs_covered
- q=16 relation=died_on endpoint_shape=all_variable_obligation triple=['?x1', 'died on', '?date'] coverage=gold_docs_covered
- q=18 relation=died_on endpoint_shape=all_variable_obligation triple=['?x1', 'died on', '?date1'] coverage=gold_docs_covered
- q=33 relation=born_in endpoint_shape=all_variable_obligation triple=['?x1', 'was born in', '?place'] coverage=candidate_missing

### work_metadata_role::gold_endpoint_relation_mismatch
- q=12 relation=perform endpoint_shape=typed_title_endpoint triple=['song When The Stars Go Blue', 'performed by', '?x1'] coverage=gold_docs_covered
- q=22 relation=composer endpoint_shape=typed_title_endpoint triple=['film Billy Elliot', 'composer', '?x1'] coverage=gold_docs_covered
- q=36 relation=composer endpoint_shape=title_qualified_endpoint triple=['Inherent Vice (Film)', 'composer', '?x1'] coverage=gold_docs_covered
- q=44 relation=perform endpoint_shape=title_qualified_endpoint triple=['song B Boy (Song)', 'performed by', '?x1'] coverage=gold_docs_covered
- q=48 relation=composer endpoint_shape=typed_title_endpoint triple=['film The Straw Hat', 'composer of', '?x1'] coverage=gold_docs_covered

### work_metadata_role::gold_relation_present_without_endpoint_binding
- q=4 relation=direct endpoint_shape=named_endpoint triple=['Aldri Annet Enn Bråk', 'director', '?x2'] coverage=gold_docs_covered
- q=30 relation=direct endpoint_shape=long_bound_endpoint triple=["I'Ll Be Going Now", 'director', '?x2'] coverage=gold_docs_covered
- q=41 relation=direct endpoint_shape=descriptive_bound_endpoint triple=['The Blue Collar Worker And The Hairdresser In A Whirl Of Sex And Politics', 'director', '?x2'] coverage=gold_docs_covered
- q=71 relation=direct endpoint_shape=title_qualified_endpoint triple=['Dancing In The Rain (Film)', 'director', '?x1'] coverage=gold_docs_covered
