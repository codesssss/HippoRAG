# V13B Source-Support Audit: musique

Diagnostic-only schema-light audit. It checks whether gold source text contains anchored predicate evidence for retrieval-critical demands.

## Summary

| metric | value |
|---|---:|
| retrieval_critical_obligation_count | 253 |
| candidate_coverage:gold_docs_covered | 237 |
| candidate_coverage:candidate_missing | 16 |
| source_support:source_support_missing | 105 |
| source_support:unresolved_variable_issue | 70 |
| source_support:openie_failed_but_source_supports | 37 |
| source_support:openie_exact_available | 26 |
| source_support:descriptive_endpoint_query_issue | 15 |

## Interpretation

| item | value |
|---|---|
| openie_exact_rate | 0.102767 |
| openie_failed_but_source_supports_rate | 0.146245 |
| unresolved_or_descriptive_query_issue_rate | 0.335968 |
| main_signal | query_compiler_or_binding_bottleneck |

## Status By Endpoint Shape

| endpoint_shape | total | descriptive_endpoint_query_issue | openie_exact_available | openie_failed_but_source_supports | source_support_missing | unresolved_variable_issue |
|---|---:|---:|---:|---:|---:|---:|
| all_variable_obligation | 105 | 0 | 2 | 9 | 24 | 70 |
| descriptive_bound_endpoint | 20 | 15 | 3 | 2 | 0 | 0 |
| long_bound_endpoint | 6 | 0 | 1 | 0 | 5 | 0 |
| named_endpoint | 122 | 0 | 20 | 26 | 76 | 0 |

## Relation Token Count Distribution

| bucket | count |
|---|---:|
| 1_content_relation_tokens | 178 |
| 2_content_relation_tokens | 47 |
| 3_content_relation_tokens | 14 |
| 4_content_relation_tokens | 9 |
| 0_content_relation_tokens | 3 |
| 8_content_relation_tokens | 1 |
| 7_content_relation_tokens | 1 |

## Representative Cases

### descriptive_endpoint_query_issue
- q=1 relation=headquarter_in endpoint_shape=descriptive_bound_endpoint triple=['?country', 'headquartered in', 'the nobilities commonwealth'] support_candidates=0
- q=10 relation=seri endpoint_shape=descriptive_bound_endpoint triple=['The Bag or the Bat', 'is a series of', '?series'] support_candidates=0
- q=11 relation=includ endpoint_shape=descriptive_bound_endpoint triple=['?x1', 'includes', "A Lim's country"] support_candidates=0
- q=22 relation=written_about endpoint_shape=descriptive_bound_endpoint triple=['the rioting being a dividing factor in Birmingham', 'written about by', '?x1'] support_candidates=0
- q=30 relation=screenwriter endpoint_shape=descriptive_bound_endpoint triple=['The Poor Boob', 'screenwriter', '?x1'] support_candidates=0

### openie_failed_but_source_supports
- q=0 relation=scor_goal_in endpoint_shape=named_endpoint triple=['Messi', 'scored goals in', 'Copa del Rey'] support_candidates=1
- q=8 relation=album endpoint_shape=named_endpoint triple=['III', 'is an album by', '?x1'] support_candidates=3
- q=16 relation=locat_in endpoint_shape=all_variable_obligation triple=['?region', 'located in', '?state'] support_candidates=1
- q=16 relation=locat_in endpoint_shape=all_variable_obligation triple=['?city', 'located in', '?state'] support_candidates=1
- q=17 relation=actor_in endpoint_shape=named_endpoint triple=['Terminator', 'actor in', '?x1'] support_candidates=1

### source_support_missing
- q=0 relation=get_sign endpoint_shape=named_endpoint triple=['?x2', 'get signed by', 'Barcelona'] support_candidates=0
- q=1 relation=participat_in endpoint_shape=named_endpoint triple=['Britain', 'participated in', '?x1'] support_candidates=0
- q=1 relation=participat_in endpoint_shape=named_endpoint triple=['France', 'participated in', '?x1'] support_candidates=0
- q=1 relation=top_rank endpoint_shape=named_endpoint triple=['?x2', 'top-ranking', 'Warsaw Pact operatives'] support_candidates=0
- q=4 relation=born_in endpoint_shape=named_endpoint triple=['Lady Godiva', 'birthplace', '?x1'] support_candidates=0

### unresolved_variable_issue
- q=1 relation=involv endpoint_shape=all_variable_obligation triple=['?x1', 'involved', '?country'] support_candidates=0
- q=1 relation=originat endpoint_shape=all_variable_obligation triple=['?country', 'originated', '?x2'] support_candidates=0
- q=7 relation=headquarter_locat_in endpoint_shape=all_variable_obligation triple=['?x2', 'headquarters located in', '?city'] support_candidates=0
- q=7 relation=headquarter_locat_in endpoint_shape=all_variable_obligation triple=['?x1', 'headquarters located in', '?city'] support_candidates=0
- q=8 relation=born_in endpoint_shape=all_variable_obligation triple=['?x1', 'was born in', '?place'] support_candidates=0
