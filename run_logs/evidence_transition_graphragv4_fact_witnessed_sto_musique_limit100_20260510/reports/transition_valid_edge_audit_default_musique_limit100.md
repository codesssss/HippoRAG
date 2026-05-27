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

## Examples

### admitted_no_edge_to_selected_context

| query | gold doc | selected top5 | new endpoints | edge kinds |
| ---: | ---: | --- | --- | --- |
| 6 | 124 | `(133, 119, 128, 122, 1667)` | `[]` | `{}` |
| 8 | 161 | `(4485, 167, 2470, 1054, 9827)` | `['iii']` | `{}` |
| 8 | 178 | `(4485, 167, 2470, 1054, 9827)` | `[]` | `{}` |
| 11 | 225 | `(220, 217, 4330, 223, 4331)` | `[]` | `{}` |
| 13 | 259 | `(261, 265, 267, 9815, 262)` | `[]` | `{}` |

### already_selected

| query | gold doc | selected top5 | new endpoints | edge kinds |
| ---: | ---: | --- | --- | --- |
| 0 | 1 | `(1, 2, 0, 15, 4049)` | `[]` | `{}` |
| 0 | 2 | `(1, 2, 0, 15, 4049)` | `[]` | `{}` |
| 1 | 29 | `(932, 35, 29, 8419, 32)` | `[]` | `{}` |
| 1 | 35 | `(932, 35, 29, 8419, 32)` | `[]` | `{}` |
| 2 | 44 | `(59, 44, 3759, 2379, 628)` | `[]` | `{}` |

### candidate_missing

| query | gold doc | selected top5 | new endpoints | edge kinds |
| ---: | ---: | --- | --- | --- |
| 1 | 22 | `(932, 35, 29, 8419, 32)` | `[]` | `{}` |
| 5 | 102 | `(110, 1743, 5940, 104, 11588)` | `[]` | `{}` |
| 7 | 148 | `(147, 2628, 1123, 5545, 149)` | `[]` | `{}` |
| 7 | 158 | `(147, 2628, 1123, 5545, 149)` | `[]` | `{}` |
| 8 | 168 | `(4485, 167, 2470, 1054, 9827)` | `[]` | `{}` |

### fact_witnessed_but_not_transition_valid

| query | gold doc | selected top5 | new endpoints | edge kinds |
| ---: | ---: | --- | --- | --- |
| 5 | 112 | `(110, 1743, 5940, 104, 11588)` | `[]` | `{'same_object': 3}` |
| 11 | 235 | `(220, 217, 4330, 223, 4331)` | `[]` | `{'same_object': 2}` |
| 23 | 456 | `(451, 447, 463, 3117, 450)` | `[]` | `{'same_object': 1}` |
| 24 | 474 | `(470, 1290, 482, 4058, 476)` | `[]` | `{'same_object': 2}` |
| 30 | 584 | `(580, 7577, 583, 7980, 3013)` | `[]` | `{'same_object': 1}` |

### transition_valid_but_no_new_query_endpoint

| query | gold doc | selected top5 | new endpoints | edge kinds |
| ---: | ---: | --- | --- | --- |
| 7 | 146 | `(147, 2628, 1123, 5545, 149)` | `[]` | `{'sentence_grounded_transition': 1, 'role_bridge': 1, 'title_role_grounding': 1}` |
| 14 | 286 | `(8378, 11160, 296, 279, 9782)` | `[]` | `{'sentence_grounded_transition': 1, 'role_bridge': 1, 'same_subject': 1, 'title_role_grounding': 1}` |
| 30 | 582 | `(580, 7577, 583, 7980, 3013)` | `[]` | `{'role_bridge': 1}` |
| 33 | 102 | `(629, 5254, 619, 9460, 9413)` | `[]` | `{'role_bridge': 2}` |
| 39 | 734 | `(745, 5143, 746, 6935, 1660)` | `[]` | `{'role_bridge': 1, 'same_object': 2, 'title_role_grounding': 1}` |

### transition_valid_reverse_only

| query | gold doc | selected top5 | new endpoints | edge kinds |
| ---: | ---: | --- | --- | --- |
| 6 | 120 | `(133, 119, 128, 122, 1667)` | `[]` | `{'same_object': 2, 'sentence_grounded_transition': 1, 'role_bridge': 1, 'title_role_grounding': 1}` |
| 25 | 503 | `(496, 491, 501, 488, 6046)` | `[]` | `{'role_bridge': 1, 'same_object': 1}` |
| 26 | 273 | `(504, 2033, 267, 9815, 270)` | `[]` | `{'same_object': 3, 'role_bridge': 1}` |
| 36 | 680 | `(1287, 679, 3936, 686, 673)` | `[]` | `{'sentence_grounded_transition': 1}` |
| 70 | 1247 | `(1264, 52, 1251, 1259, 11392)` | `[]` | `{'sentence_grounded_transition': 1}` |

