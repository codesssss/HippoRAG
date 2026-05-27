# Transition-Valid Edge Failure Audit

| field | value |
| --- | --- |
| dataset | musique |
| rows | 100 |
| source report | `run_logs/evidence_transition_graphragv4_fact_witnessed_sto_musique_limit100_20260510/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json` |
| comparison report | `` |

| method | R@5 | all-gold@5 | mean certified docs@5 |
| --- | ---: | ---: | ---: |
| source | 0.709167 | 0.390000 | 2.750000 |

| gold status | count |
| --- | ---: |
| admitted_no_edge_to_selected_context | 20 |
| already_selected | 183 |
| candidate_missing | 20 |
| fact_witnessed_but_not_transition_valid | 17 |
| transition_valid_but_no_new_query_endpoint | 21 |
| transition_valid_reverse_only | 13 |

| missing-gold status | count |
| --- | ---: |
| admitted_no_edge_to_selected_context | 20 |
| candidate_missing | 20 |
| fact_witnessed_but_not_transition_valid | 17 |
| transition_valid_but_no_new_query_endpoint | 21 |
| transition_valid_reverse_only | 13 |

| query primary status | count |
| --- | ---: |
| admitted_no_edge_to_selected_context | 10 |
| all_gold_selected | 39 |
| candidate_missing | 14 |
| fact_witnessed_but_not_transition_valid | 12 |
| transition_valid_but_no_new_query_endpoint | 15 |
| transition_valid_reverse_only | 10 |

| missing-gold status | selected->gold certificate docs | gold->selected certificate docs |
| --- | ---: | ---: |
| admitted_no_edge_to_selected_context | 6 | 0 |
| fact_witnessed_but_not_transition_valid | 5 | 6 |
| transition_valid_but_no_new_query_endpoint | 19 | 8 |
| transition_valid_reverse_only | 8 | 10 |

| certificate precision field | value |
| --- | ---: |
| selected->candidate certificate targets | 2184 |
| selected->gold certificate targets | 38 |
| selected->missing-gold certificate targets | 38 |

## Examples

### admitted_no_edge_to_selected_context

| query | gold doc | selected top5 | new endpoints | edge kinds |
| ---: | ---: | --- | --- | --- |
| 6 | 124 | `(133, 119, 128, 122, 1667)` | `[]` | `{}; s2g={'title_alias': 1}; g2s={}` |
| 8 | 161 | `(4485, 167, 2470, 1054, 9827)` | `['iii']` | `{}; s2g={}; g2s={}` |
| 8 | 178 | `(4485, 167, 2470, 1054, 9827)` | `[]` | `{}; s2g={}; g2s={}` |
| 11 | 225 | `(220, 217, 4330, 223, 4331)` | `[]` | `{}; s2g={}; g2s={}` |
| 13 | 259 | `(261, 265, 267, 9815, 262)` | `[]` | `{}; s2g={'title_alias': 2, 'endpoint_title': 2}; g2s={}` |

### already_selected

| query | gold doc | selected top5 | new endpoints | edge kinds |
| ---: | ---: | --- | --- | --- |
| 0 | 1 | `(1, 2, 0, 15, 4049)` | `[]` | `{}; s2g={}; g2s={}` |
| 0 | 2 | `(1, 2, 0, 15, 4049)` | `[]` | `{}; s2g={}; g2s={}` |
| 1 | 29 | `(932, 35, 29, 8419, 32)` | `[]` | `{}; s2g={}; g2s={}` |
| 1 | 35 | `(932, 35, 29, 8419, 32)` | `[]` | `{}; s2g={}; g2s={}` |
| 2 | 44 | `(59, 44, 3759, 2379, 628)` | `[]` | `{}; s2g={}; g2s={}` |

### candidate_missing

| query | gold doc | selected top5 | new endpoints | edge kinds |
| ---: | ---: | --- | --- | --- |
| 1 | 22 | `(932, 35, 29, 8419, 32)` | `[]` | `{}; s2g={}; g2s={}` |
| 5 | 102 | `(110, 1743, 5940, 104, 11588)` | `[]` | `{}; s2g={}; g2s={}` |
| 7 | 148 | `(147, 2628, 1123, 5545, 149)` | `[]` | `{}; s2g={}; g2s={}` |
| 7 | 158 | `(147, 2628, 1123, 5545, 149)` | `[]` | `{}; s2g={}; g2s={}` |
| 8 | 168 | `(4485, 167, 2470, 1054, 9827)` | `[]` | `{}; s2g={}; g2s={}` |

### fact_witnessed_but_not_transition_valid

| query | gold doc | selected top5 | new endpoints | edge kinds |
| ---: | ---: | --- | --- | --- |
| 5 | 112 | `(110, 1743, 5940, 104, 11588)` | `[]` | `{'same_object': 3}; s2g={}; g2s={}` |
| 11 | 235 | `(220, 217, 4330, 223, 4331)` | `[]` | `{'same_object': 2}; s2g={}; g2s={}` |
| 23 | 456 | `(451, 447, 463, 3117, 450)` | `[]` | `{'same_object': 1}; s2g={}; g2s={}` |
| 24 | 474 | `(470, 1290, 482, 4058, 476)` | `[]` | `{'same_object': 2}; s2g={}; g2s={'endpoint_title': 1, 'source_title_endpoint': 1}` |
| 30 | 584 | `(580, 7577, 583, 7980, 3013)` | `[]` | `{'same_object': 1}; s2g={'title_alias': 1, 'endpoint_title': 1, 'source_title_endpoint': 1}; g2s={}` |

### transition_valid_but_no_new_query_endpoint

| query | gold doc | selected top5 | new endpoints | edge kinds |
| ---: | ---: | --- | --- | --- |
| 7 | 146 | `(147, 2628, 1123, 5545, 149)` | `[]` | `{'sentence_grounded_transition': 1, 'role_bridge': 1, 'title_role_grounding': 1}; s2g={'title_alias': 2, 'endpoint_title': 4, 'source_title_endpoint': 2}; g2s={}` |
| 14 | 286 | `(8378, 11160, 296, 279, 9782)` | `[]` | `{'sentence_grounded_transition': 1, 'role_bridge': 1, 'same_subject': 1, 'title_role_grounding': 1}; s2g={'title_alias': 1, 'endpoint_title': 4}; g2s={'endpoint_title': 7, 'source_title_endpoint': 1}` |
| 30 | 582 | `(580, 7577, 583, 7980, 3013)` | `[]` | `{'role_bridge': 1}; s2g={'title_alias': 1, 'endpoint_title': 1, 'source_title_endpoint': 1}; g2s={}` |
| 33 | 102 | `(629, 5254, 619, 9460, 9413)` | `[]` | `{'role_bridge': 2}; s2g={'endpoint_title': 2, 'source_title_endpoint': 2}; g2s={}` |
| 39 | 734 | `(745, 5143, 746, 6935, 1660)` | `[]` | `{'role_bridge': 1, 'same_object': 2, 'title_role_grounding': 1}; s2g={'title_alias': 1, 'endpoint_title': 1}; g2s={'endpoint_title': 4, 'source_title_endpoint': 4}` |

### transition_valid_reverse_only

| query | gold doc | selected top5 | new endpoints | edge kinds |
| ---: | ---: | --- | --- | --- |
| 6 | 120 | `(133, 119, 128, 122, 1667)` | `[]` | `{'same_object': 2, 'sentence_grounded_transition': 1, 'role_bridge': 1, 'title_role_grounding': 1}; s2g={}; g2s={'title_alias': 1, 'endpoint_title': 1, 'source_title_endpoint': 1}` |
| 25 | 503 | `(496, 491, 501, 488, 6046)` | `[]` | `{'role_bridge': 1, 'same_object': 1}; s2g={'endpoint_title': 4, 'source_title_endpoint': 1}; g2s={}` |
| 26 | 273 | `(504, 2033, 267, 9815, 270)` | `[]` | `{'same_object': 3, 'role_bridge': 1}; s2g={}; g2s={}` |
| 36 | 680 | `(1287, 679, 3936, 686, 673)` | `[]` | `{'sentence_grounded_transition': 1}; s2g={'endpoint_title': 24, 'source_title_endpoint': 2}; g2s={'title_alias': 1, 'endpoint_title': 1}` |
| 70 | 1247 | `(1264, 52, 1251, 1259, 11392)` | `[]` | `{'sentence_grounded_transition': 1}; s2g={'title_alias': 1, 'endpoint_title': 1, 'source_title_endpoint': 1}; g2s={'title_alias': 1, 'endpoint_title': 1, 'source_title_endpoint': 1}` |

