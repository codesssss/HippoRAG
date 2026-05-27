# OpenIE/STO Trace Gap Audit

| field | value |
| --- | --- |
| dataset | musique |
| rows | 100 |
| base report | `run_logs/evidence_transition_graphragv4_fact_witnessed_sto_musique_limit100_20260510/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json` |
| reference report | `run_logs/evidence_transition_graphragv4_fact_witnessed_sto_gpt4omini_musique_limit100_20260511/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json` |
| base OpenIE | `/mnt/nvme/code/HippoRAG/run_logs/evidence_transition_qwen8b_nv2_limit100_musique_20260509/musique/index/openie_results_ner_qwen3-8b-train.json` |
| reference OpenIE | `/mnt/nvme/code/HippoRAG/run_logs/evidence_transition_graphragv4_fact_witnessed_sto_gpt4omini_musique_limit100_20260511/musique/index/openie_results_ner_gpt-4o-mini.json` |

| substrate | R@5 | all-gold@5 | mean certified docs@5 |
| --- | ---: | ---: | ---: |
| base | 0.709167 | 0.390000 | 2.750000 |
| reference | 0.728333 | 0.420000 | 2.840000 |

## Query Delta

| key | count |
| --- | ---: |
| base_better | 1 |
| reference_better | 6 |
| tie | 93 |

## Query Endpoint Delta

| key | count |
| --- | ---: |
| changed | 16 |
| same | 84 |

## Base-Missing Gold Outcome On Reference

| key | count |
| --- | ---: |
| candidate_missing | 18 |
| graph_tail_not_ordered | 64 |
| raw_graph_order_filtered | 1 |
| reference_selected | 6 |
| source_prior_not_selected | 2 |

## Base Stage -> Reference Outcome

| key | count |
| --- | ---: |
| candidate_missing -> candidate_missing | 14 |
| candidate_missing -> graph_tail_not_ordered | 5 |
| candidate_missing -> reference_selected | 1 |
| graph_tail_not_ordered -> candidate_missing | 4 |
| graph_tail_not_ordered -> graph_tail_not_ordered | 59 |
| graph_tail_not_ordered -> reference_selected | 4 |
| raw_graph_order_filtered -> raw_graph_order_filtered | 1 |
| source_prior_not_selected -> reference_selected | 1 |
| source_prior_not_selected -> source_prior_not_selected | 2 |

## Reference Recovered Gold By Depth

| key | count |
| ---: | ---: |
| 2 | 2 |
| 3 | 3 |
| 4 | 1 |

## Examples

### candidate_missing -> candidate_missing

| query | gold_doc | base_top5 | reference_top5 | base_R@5 | reference_R@5 |
| ---: | ---: | --- | --- | ---: | ---: |
| 1 | 22 | [932, 35, 29, 8419, 32] | [29, 533, 35, 8419, 32] | 0.666667 | 0.666667 |
| 5 | 102 | [110, 1743, 5940, 104, 11588] | [110, 9460, 1743, 5940, 104] | 0.500000 | 0.500000 |
| 7 | 158 | [147, 2628, 1123, 5545, 149] | [147, 2621, 1123, 5545, 149] | 0.250000 | 0.250000 |
| 8 | 168 | [4485, 167, 2470, 1054, 9827] | [4485, 167, 2470, 1054, 9827] | 0.000000 | 0.000000 |
| 12 | 254 | [239, 1272, 244, 5132, 246] | [239, 6556, 1272, 244, 5132] | 0.500000 | 0.500000 |

### candidate_missing -> graph_tail_not_ordered

| query | gold_doc | base_top5 | reference_top5 | base_R@5 | reference_R@5 |
| ---: | ---: | --- | --- | ---: | ---: |
| 7 | 148 | [147, 2628, 1123, 5545, 149] | [147, 2621, 1123, 5545, 149] | 0.250000 | 0.250000 |
| 13 | 273 | [261, 265, 267, 9815, 262] | [259, 532, 267, 9815, 262] | 0.333333 | 0.666667 |
| 22 | 439 | [433, 427, 428, 432, 6156] | [428, 3106, 433, 427, 432] | 0.500000 | 0.500000 |
| 34 | 638 | [636, 641, 637, 645, 11037] | [636, 637, 645, 11037, 642] | 0.500000 | 0.500000 |
| 49 | 146 | [931, 2628, 140, 923, 5120] | [931, 2621, 140, 923, 5120] | 0.250000 | 0.250000 |

### candidate_missing -> reference_selected

| query | gold_doc | base_top5 | reference_top5 | base_R@5 | reference_R@5 |
| ---: | ---: | --- | --- | ---: | ---: |
| 97 | 1709 | [1715, 348, 6344, 8662, 8557] | [1715, 1709, 348, 6344, 8662] | 0.500000 | 1.000000 |

### graph_tail_not_ordered -> candidate_missing

| query | gold_doc | base_top5 | reference_top5 | base_R@5 | reference_R@5 |
| ---: | ---: | --- | --- | ---: | ---: |
| 8 | 178 | [4485, 167, 2470, 1054, 9827] | [4485, 167, 2470, 1054, 9827] | 0.000000 | 0.000000 |
| 40 | 762 | [749, 6829, 761, 760, 2439] | [749, 761, 760, 2439, 751] | 0.666667 | 0.666667 |
| 70 | 1247 | [1264, 52, 1251, 1259, 11392] | [1264, 3093, 1251, 1259, 11392] | 0.500000 | 0.500000 |
| 91 | 1601 | [1594, 1287, 1605, 7749, 1804] | [1594, 1287, 1605, 7749, 1804] | 0.250000 | 0.250000 |

### graph_tail_not_ordered -> graph_tail_not_ordered

| query | gold_doc | base_top5 | reference_top5 | base_R@5 | reference_R@5 |
| ---: | ---: | --- | --- | ---: | ---: |
| 5 | 112 | [110, 1743, 5940, 104, 11588] | [110, 9460, 1743, 5940, 104] | 0.500000 | 0.500000 |
| 6 | 120 | [133, 119, 128, 122, 1667] | [133, 119, 128, 122, 1667] | 0.500000 | 0.500000 |
| 6 | 124 | [133, 119, 128, 122, 1667] | [133, 119, 128, 122, 1667] | 0.500000 | 0.500000 |
| 7 | 146 | [147, 2628, 1123, 5545, 149] | [147, 2621, 1123, 5545, 149] | 0.250000 | 0.250000 |
| 11 | 225 | [220, 217, 4330, 223, 4331] | [220, 4250, 217, 4330, 223] | 0.500000 | 0.500000 |

### graph_tail_not_ordered -> reference_selected

| query | gold_doc | base_top5 | reference_top5 | base_R@5 | reference_R@5 |
| ---: | ---: | --- | --- | ---: | ---: |
| 13 | 259 | [261, 265, 267, 9815, 262] | [259, 532, 267, 9815, 262] | 0.333333 | 0.666667 |
| 30 | 584 | [580, 7577, 583, 7980, 3013] | [580, 584, 583, 7980, 3013] | 0.500000 | 0.750000 |
| 68 | 613 | [1241, 609, 1238, 3770, 2185] | [1241, 613, 1238, 3770, 2185] | 0.666667 | 1.000000 |
| 71 | 1272 | [1273, 5695, 1106, 2005, 10975] | [1273, 1272, 5695, 1106, 2005] | 0.500000 | 1.000000 |

### raw_graph_order_filtered -> raw_graph_order_filtered

| query | gold_doc | base_top5 | reference_top5 | base_R@5 | reference_R@5 |
| ---: | ---: | --- | --- | ---: | ---: |
| 8 | 161 | [4485, 167, 2470, 1054, 9827] | [4485, 167, 2470, 1054, 9827] | 0.000000 | 0.000000 |

### source_prior_not_selected -> reference_selected

| query | gold_doc | base_top5 | reference_top5 | base_R@5 | reference_R@5 |
| ---: | ---: | --- | --- | ---: | ---: |
| 36 | 680 | [1287, 679, 3936, 686, 673] | [679, 3936, 686, 673, 680] | 0.666667 | 1.000000 |

### source_prior_not_selected -> source_prior_not_selected

| query | gold_doc | base_top5 | reference_top5 | base_R@5 | reference_R@5 |
| ---: | ---: | --- | --- | ---: | ---: |
| 79 | 1402 | [1395, 1145, 1401, 1397, 11255] | [1395, 6606, 1401, 1397, 11255] | 0.666667 | 0.666667 |
| 91 | 1595 | [1594, 1287, 1605, 7749, 1804] | [1594, 1287, 1605, 7749, 1804] | 0.250000 | 0.250000 |
