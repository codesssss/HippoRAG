# V13B Source-Support Audit: hotpotqa

Diagnostic-only schema-light audit. It checks whether gold source text contains anchored predicate evidence for retrieval-critical demands.

## Summary

| metric | value |
|---|---:|
| retrieval_critical_obligation_count | 216 |
| candidate_coverage:gold_docs_covered | 216 |
| source_support:source_support_missing | 81 |
| source_support:openie_failed_but_source_supports | 58 |
| source_support:openie_exact_available | 38 |
| source_support:unresolved_variable_issue | 22 |
| source_support:descriptive_endpoint_query_issue | 17 |

## Interpretation

| item | value |
|---|---|
| openie_exact_rate | 0.175926 |
| openie_failed_but_source_supports_rate | 0.268519 |
| unresolved_or_descriptive_query_issue_rate | 0.180556 |
| main_signal | source_support_exceeds_openie_exact_gap |

## Status By Endpoint Shape

| endpoint_shape | total | descriptive_endpoint_query_issue | openie_exact_available | openie_failed_but_source_supports | source_support_missing | unresolved_variable_issue |
|---|---:|---:|---:|---:|---:|---:|
| all_variable_obligation | 49 | 0 | 3 | 10 | 14 | 22 |
| descriptive_bound_endpoint | 26 | 17 | 3 | 6 | 0 | 0 |
| long_bound_endpoint | 12 | 0 | 2 | 4 | 6 | 0 |
| named_endpoint | 129 | 0 | 30 | 38 | 61 | 0 |

## Relation Token Count Distribution

| bucket | count |
|---|---:|
| 1_content_relation_tokens | 150 |
| 2_content_relation_tokens | 40 |
| 3_content_relation_tokens | 11 |
| 0_content_relation_tokens | 5 |
| 6_content_relation_tokens | 3 |
| 4_content_relation_tokens | 3 |
| 8_content_relation_tokens | 2 |
| 5_content_relation_tokens | 2 |

## Representative Cases

### descriptive_endpoint_query_issue
- q=2 relation=pass endpoint_shape=descriptive_bound_endpoint triple=['The Distribution of Industry act', 'passed by', '?x1'] support_candidates=0
- q=3 relation=help endpoint_shape=descriptive_bound_endpoint triple=['?x1', 'helped', 'recently abdicated queen'] support_candidates=0
- q=3 relation=imprison endpoint_shape=descriptive_bound_endpoint triple=['recently abdicated queen', 'imprisoned by', '?x2'] support_candidates=0
- q=9 relation=debut endpoint_shape=descriptive_bound_endpoint triple=['The 2000 ICC KnockOut Trophy', 'debut of', '?x1'] support_candidates=0
- q=14 relation=year endpoint_shape=descriptive_bound_endpoint triple=['The Secret of Kells', 'year', '2009'] support_candidates=0

### openie_failed_but_source_supports
- q=0 relation=starr_in endpoint_shape=descriptive_bound_endpoint triple=['The Newcomers', 'starred in', '?x1'] support_candidates=1
- q=6 relation=produc_in endpoint_shape=descriptive_bound_endpoint triple=['The Apple Dumpling Gang', 'produced in', '?date1'] support_candidates=1
- q=6 relation=produc_in endpoint_shape=long_bound_endpoint triple=['Something Wicked This Way Comes', 'produced in', '?date2'] support_candidates=1
- q=7 relation=begin_with_interchange_at endpoint_shape=named_endpoint triple=['Bethpage State Parkway', 'begins with an interchange at', '?x1'] support_candidates=1
- q=10 relation=civil_parish endpoint_shape=all_variable_obligation triple=['?x1', 'has civil parish of', '?x2'] support_candidates=1

### source_support_missing
- q=3 relation=spark endpoint_shape=named_endpoint triple=['Marian civil war', 'sparked by', '?x1'] support_candidates=0
- q=5 relation=work_on endpoint_shape=named_endpoint triple=['Miklos Rozsa', 'worked on', '?screenplay'] support_candidates=0
- q=9 relation=from endpoint_shape=named_endpoint triple=['?x1', 'is from', 'Jamaica'] support_candidates=0
- q=11 relation=locat_in endpoint_shape=named_endpoint triple=['Truro Cathedral', 'located in', '?x1'] support_candidates=0
- q=12 relation=written endpoint_shape=named_endpoint triple=['Peter Laufer', 'wrote', 'Forbidden Creatures'] support_candidates=0

### unresolved_variable_issue
- q=20 relation=locat_in endpoint_shape=all_variable_obligation triple=['?chain', 'located in', '?part'] support_candidates=0
- q=21 relation=former_home endpoint_shape=all_variable_obligation triple=['?x1', 'former home', '?location'] support_candidates=0
- q=21 relation=locat_at endpoint_shape=all_variable_obligation triple=['?location', 'is located at', '?intersection'] support_candidates=0
- q=29 relation=notable_for_hav endpoint_shape=all_variable_obligation triple=['?x1', 'notable for having', '?x2'] support_candidates=0
- q=29 relation=hack_organization_with_user_base_over_1_800_000 endpoint_shape=all_variable_obligation triple=['?x2', 'hacking organization with a user base of over 1,800,000', '?x3'] support_candidates=0
