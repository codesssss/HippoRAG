# V13B Source-Support Audit: 2wikimultihopqa

Diagnostic-only schema-light audit. It checks whether gold source text contains anchored predicate evidence for retrieval-critical demands.

## Summary

| metric | value |
|---|---:|
| retrieval_critical_obligation_count | 220 |
| candidate_coverage:gold_docs_covered | 218 |
| candidate_coverage:candidate_missing | 2 |
| source_support:openie_exact_available | 127 |
| source_support:source_support_missing | 53 |
| source_support:unresolved_variable_issue | 22 |
| source_support:openie_failed_but_source_supports | 18 |

## Interpretation

| item | value |
|---|---|
| openie_exact_rate | 0.577273 |
| openie_failed_but_source_supports_rate | 0.081818 |
| unresolved_or_descriptive_query_issue_rate | 0.1 |
| main_signal | openie_exact_often_available |

## Status By Endpoint Shape

| endpoint_shape | total | openie_exact_available | openie_failed_but_source_supports | source_support_missing | unresolved_variable_issue |
|---|---:|---:|---:|---:|---:|
| all_variable_obligation | 73 | 35 | 3 | 13 | 22 |
| descriptive_bound_endpoint | 11 | 9 | 2 | 0 | 0 |
| long_bound_endpoint | 16 | 10 | 2 | 4 | 0 |
| named_endpoint | 82 | 52 | 7 | 23 | 0 |
| title_qualified_endpoint | 18 | 8 | 2 | 8 | 0 |
| typed_title_endpoint | 20 | 13 | 2 | 5 | 0 |

## Relation Token Count Distribution

| bucket | count |
|---|---:|
| 1_content_relation_tokens | 210 |
| 2_content_relation_tokens | 6 |
| 0_content_relation_tokens | 3 |
| 3_content_relation_tokens | 1 |

## Representative Cases

### openie_failed_but_source_supports
- q=4 relation=direct endpoint_shape=named_endpoint triple=['Aldri Annet Enn Bråk', 'director', '?x2'] support_candidates=1
- q=5 relation=born_on endpoint_shape=all_variable_obligation triple=['?x1', 'born on', '?date1'] support_candidates=1
- q=6 relation=born_on endpoint_shape=named_endpoint triple=['Andy Summers', 'born on', '?date2'] support_candidates=1
- q=12 relation=perform endpoint_shape=typed_title_endpoint triple=['song When The Stars Go Blue', 'performed by', '?x1'] support_candidates=1
- q=13 relation=song endpoint_shape=typed_title_endpoint triple=['song Me And Bobby Mcgee', 'is an song by', '?x1'] support_candidates=1

### source_support_missing
- q=0 relation=mother endpoint_shape=named_endpoint triple=['Lothair Ii', 'mother', '?x1'] support_candidates=0
- q=1 relation=released_on endpoint_shape=named_endpoint triple=['Aas Ka Panchhi', 'release date', '?date1'] support_candidates=0
- q=1 relation=released_on endpoint_shape=named_endpoint triple=['Phoolwari', 'release date', '?date2'] support_candidates=0
- q=3 relation=locat_in endpoint_shape=named_endpoint triple=['Nasamkhrali', 'located in', '?country'] support_candidates=0
- q=7 relation=paternal_grandfather endpoint_shape=named_endpoint triple=['Raghnall Mac Ruaidhrí', 'paternal grandfather', '?x1'] support_candidates=0

### unresolved_variable_issue
- q=14 relation=died_in endpoint_shape=all_variable_obligation triple=['?x1', 'died in', '?place'] support_candidates=0
- q=15 relation=from endpoint_shape=all_variable_obligation triple=['?x1', 'is from', '?country'] support_candidates=0
- q=20 relation=study_at endpoint_shape=all_variable_obligation triple=['?x1', 'studied at', '?place'] support_candidates=0
- q=21 relation=died endpoint_shape=all_variable_obligation triple=['?x1', 'died', '?reason'] support_candidates=0
- q=22 relation=born_in endpoint_shape=all_variable_obligation triple=['?x1', 'was born in', '?place'] support_candidates=0
