# AG-STO Clean-RRF Audit

## Config

| parameter | value |
| --- | --- |
| datasets | 2wikimultihopqa,hotpotqa,musique |
| limit | 100 |
| pool_k | 100 |
| candidate_limit | 120 |
| proposal_candidate_depth | 10 |
| support_proposal_depth | 6 |
| beam_size | 12 |
| stable_anchor_k | 2 |
| max_endpoint_degree | 30 |
| support_weights | 1.0,1.25,1.5,1.75,2.0,2.25 |
| openie_template | outputs_step0_general_nvembed_{dataset}/openie_results_ner_qwen3-8b.json |
| hipporag_pool_template | run_logs/hipporag_pool_exports_full1000_20260503/{dataset}_hipporag_pool100.json |

## 2wikimultihopqa

### Path Usage
| metric | value |
| --- | --- |
| selection_policy_counts | {'sto_proposal_consensus': 100} |
| consensus_nonempty_rate | 1.0 |
| selection_by_consensus_rate | 1.0 |
| completion_applied_rate | 0.09 |
| final_changed_by_completion_rate | 0.09 |

### Retrieval Metrics

| variant | R@5 | R@20 | R@100 |
| --- | --- | --- | --- |
| current_agsto | 0.8325 | 0.895 | 0.91 |
| hipporag | 0.815 | 0.9 | 0.9425 |
| current_pre_completion | 0.8275 | 0.8275 | 0.8275 |
| weighted_rrf_main | 0.8275 | 0.9425 | 0.9425 |
| unweighted_rrf | 0.875 | 0.9425 | 0.9425 |

### Channel Health

| channel | nonempty@20 | nonempty@100 | mean_len | R@5 | R@20 | R@100 |
| --- | --- | --- | --- | --- | --- | --- |
| specificity | 1.0 | 1.0 | 100.0 | 0.88 | 0.885 | 0.9 |
| hybrid_residual | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neighborhood | 1.0 | 1.0 | 100.0 | 0.855 | 0.9375 | 0.965 |
| support_set | 1.0 | 1.0 | 6.47 | 0.605 | 0.63 | 0.63 |

### Channel Overlap

| pair | Jaccard@20 | Jaccard@100 |
| --- | --- | --- |
| specificity__hybrid_residual | 0.0 | 0.0 |
| specificity__neighborhood | 0.295504 | 0.590125 |
| specificity__support_set | 0.126708 | 0.037015 |
| hybrid_residual__neighborhood | 0.0 | 0.0 |
| hybrid_residual__support_set | 0.0 | 0.0 |
| neighborhood__support_set | 0.283436 | 0.0647 |

### Unweighted Leave-One-Out

| variant | R@5 | R@20 | R@100 | Delta R@5 |
| --- | --- | --- | --- | --- |
| minus_specificity | 0.8 | 0.9075 | 0.9075 | -0.075 |
| minus_hybrid_residual | 0.875 | 0.9425 | 0.9425 | 0.0 |
| minus_neighborhood | 0.8475 | 0.915 | 0.915 | -0.0275 |
| minus_support_set | 0.89 | 0.9425 | 0.9425 | 0.015 |

### Weighted Leave-One-Out

| variant | R@5 | R@20 | R@100 | Delta R@5 |
| --- | --- | --- | --- | --- |
| minus_specificity | 0.67 | 0.9075 | 0.9075 | -0.1575 |
| minus_hybrid_residual | 0.8275 | 0.9425 | 0.9425 | 0.0 |
| minus_neighborhood | 0.7275 | 0.915 | 0.915 | -0.1 |
| minus_support_set | 0.89 | 0.9425 | 0.9425 | 0.0625 |

### Support Weight Sweep

| support_weight | R@5 | R@20 | R@100 |
| --- | --- | --- | --- |
| 1.0 | 0.875 | 0.9425 | 0.9425 |
| 1.25 | 0.85 | 0.9425 | 0.9425 |
| 1.5 | 0.8375 | 0.9425 | 0.9425 |
| 1.75 | 0.8275 | 0.9425 | 0.9425 |
| 2.0 | 0.815 | 0.9425 | 0.9425 |
| 2.25 | 0.8075 | 0.9425 | 0.9425 |

### Feasibility

| metric | value |
| --- | --- |
| contains_anchor_at5 | 1.0 |
| connected_at5 | 0.3 |
| hub_violation_at5 | 0.43 |

### AG-STO vs HippoRAG

| metric | value |
| --- | --- |
| hippo_hit_agsto_miss | 14 |
| both_hit | 43 |
| both_miss | 25 |
| agsto_hit_hippo_miss | 18 |
| mean_current_r5_minus_hippo_r5 | 0.0175 |
| mean_current_r20_minus_hippo_r20 | -0.005 |
| mean_current_r100_minus_hippo_r100 | -0.0325 |

## hotpotqa

### Path Usage
| metric | value |
| --- | --- |
| selection_policy_counts | {'sto_proposal_consensus': 100} |
| consensus_nonempty_rate | 1.0 |
| selection_by_consensus_rate | 1.0 |
| completion_applied_rate | 0.19 |
| final_changed_by_completion_rate | 0.19 |

### Retrieval Metrics

| variant | R@5 | R@20 | R@100 |
| --- | --- | --- | --- |
| current_agsto | 0.87 | 0.965 | 0.985 |
| hipporag | 0.925 | 0.995 | 1.0 |
| current_pre_completion | 0.855 | 0.855 | 0.855 |
| weighted_rrf_main | 0.855 | 0.95 | 0.95 |
| unweighted_rrf | 0.85 | 0.95 | 0.95 |

### Channel Health

| channel | nonempty@20 | nonempty@100 | mean_len | R@5 | R@20 | R@100 |
| --- | --- | --- | --- | --- | --- | --- |
| specificity | 1.0 | 1.0 | 100.0 | 0.84 | 0.955 | 0.98 |
| hybrid_residual | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neighborhood | 1.0 | 1.0 | 100.0 | 0.79 | 0.925 | 0.99 |
| support_set | 1.0 | 1.0 | 6.85 | 0.665 | 0.7 | 0.7 |

### Channel Overlap

| pair | Jaccard@20 | Jaccard@100 |
| --- | --- | --- |
| specificity__hybrid_residual | 0.0 | 0.0 |
| specificity__neighborhood | 0.287006 | 0.595485 |
| specificity__support_set | 0.2237 | 0.055043 |
| hybrid_residual__neighborhood | 0.0 | 0.0 |
| hybrid_residual__support_set | 0.0 | 0.0 |
| neighborhood__support_set | 0.302725 | 0.0685 |

### Unweighted Leave-One-Out

| variant | R@5 | R@20 | R@100 | Delta R@5 |
| --- | --- | --- | --- | --- |
| minus_specificity | 0.81 | 0.9 | 0.9 | -0.04 |
| minus_hybrid_residual | 0.85 | 0.95 | 0.95 | 0.0 |
| minus_neighborhood | 0.805 | 0.94 | 0.94 | -0.045 |
| minus_support_set | 0.835 | 0.94 | 0.94 | -0.015 |

### Weighted Leave-One-Out

| variant | R@5 | R@20 | R@100 | Delta R@5 |
| --- | --- | --- | --- | --- |
| minus_specificity | 0.72 | 0.9 | 0.9 | -0.135 |
| minus_hybrid_residual | 0.855 | 0.95 | 0.95 | 0.0 |
| minus_neighborhood | 0.73 | 0.94 | 0.94 | -0.125 |
| minus_support_set | 0.835 | 0.94 | 0.94 | -0.02 |

### Support Weight Sweep

| support_weight | R@5 | R@20 | R@100 |
| --- | --- | --- | --- |
| 1.0 | 0.85 | 0.95 | 0.95 |
| 1.25 | 0.85 | 0.95 | 0.95 |
| 1.5 | 0.855 | 0.95 | 0.95 |
| 1.75 | 0.855 | 0.95 | 0.95 |
| 2.0 | 0.83 | 0.95 | 0.95 |
| 2.25 | 0.825 | 0.95 | 0.95 |

### Feasibility

| metric | value |
| --- | --- |
| contains_anchor_at5 | 1.0 |
| connected_at5 | 0.56 |
| hub_violation_at5 | 0.46 |

### AG-STO vs HippoRAG

| metric | value |
| --- | --- |
| both_miss | 9 |
| hippo_hit_agsto_miss | 15 |
| both_hit | 70 |
| agsto_hit_hippo_miss | 6 |
| mean_current_r5_minus_hippo_r5 | -0.055 |
| mean_current_r20_minus_hippo_r20 | -0.03 |
| mean_current_r100_minus_hippo_r100 | -0.015 |

## musique

### Path Usage
| metric | value |
| --- | --- |
| selection_policy_counts | {'sto_proposal_consensus': 100} |
| consensus_nonempty_rate | 1.0 |
| selection_by_consensus_rate | 1.0 |
| completion_applied_rate | 0.25 |
| final_changed_by_completion_rate | 0.25 |

### Retrieval Metrics

| variant | R@5 | R@20 | R@100 |
| --- | --- | --- | --- |
| current_agsto | 0.516667 | 0.695 | 0.808333 |
| hipporag | 0.67 | 0.845833 | 0.955833 |
| current_pre_completion | 0.505 | 0.505 | 0.505 |
| weighted_rrf_main | 0.505 | 0.663333 | 0.663333 |
| unweighted_rrf | 0.4975 | 0.663333 | 0.663333 |

### Channel Health

| channel | nonempty@20 | nonempty@100 | mean_len | R@5 | R@20 | R@100 |
| --- | --- | --- | --- | --- | --- | --- |
| specificity | 1.0 | 1.0 | 100.0 | 0.506667 | 0.68 | 0.789167 |
| hybrid_residual | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neighborhood | 1.0 | 1.0 | 100.0 | 0.485833 | 0.595833 | 0.831667 |
| support_set | 1.0 | 1.0 | 6.81 | 0.456667 | 0.470833 | 0.470833 |

### Channel Overlap

| pair | Jaccard@20 | Jaccard@100 |
| --- | --- | --- |
| specificity__hybrid_residual | 0.0 | 0.0 |
| specificity__neighborhood | 0.281009 | 0.553743 |
| specificity__support_set | 0.17093 | 0.0469 |
| hybrid_residual__neighborhood | 0.0 | 0.0 |
| hybrid_residual__support_set | 0.0 | 0.0 |
| neighborhood__support_set | 0.304136 | 0.0681 |

### Unweighted Leave-One-Out

| variant | R@5 | R@20 | R@100 | Delta R@5 |
| --- | --- | --- | --- | --- |
| minus_specificity | 0.49 | 0.57 | 0.57 | -0.0075 |
| minus_hybrid_residual | 0.4975 | 0.663333 | 0.663333 | 0.0 |
| minus_neighborhood | 0.5025 | 0.638333 | 0.638333 | 0.005 |
| minus_support_set | 0.4925 | 0.649167 | 0.649167 | -0.005 |

### Weighted Leave-One-Out

| variant | R@5 | R@20 | R@100 | Delta R@5 |
| --- | --- | --- | --- | --- |
| minus_specificity | 0.461667 | 0.57 | 0.57 | -0.043333 |
| minus_hybrid_residual | 0.505 | 0.663333 | 0.663333 | 0.0 |
| minus_neighborhood | 0.4825 | 0.638333 | 0.638333 | -0.0225 |
| minus_support_set | 0.4925 | 0.649167 | 0.649167 | -0.0125 |

### Support Weight Sweep

| support_weight | R@5 | R@20 | R@100 |
| --- | --- | --- | --- |
| 1.0 | 0.4975 | 0.663333 | 0.663333 |
| 1.25 | 0.498333 | 0.663333 | 0.663333 |
| 1.5 | 0.501667 | 0.663333 | 0.663333 |
| 1.75 | 0.505 | 0.663333 | 0.663333 |
| 2.0 | 0.490833 | 0.663333 | 0.663333 |
| 2.25 | 0.4825 | 0.663333 | 0.663333 |

### Feasibility

| metric | value |
| --- | --- |
| contains_anchor_at5 | 1.0 |
| connected_at5 | 0.44 |
| hub_violation_at5 | 0.49 |

### AG-STO vs HippoRAG

| metric | value |
| --- | --- |
| both_hit | 18 |
| both_miss | 55 |
| agsto_hit_hippo_miss | 7 |
| hippo_hit_agsto_miss | 20 |
| mean_current_r5_minus_hippo_r5 | -0.153333 |
| mean_current_r20_minus_hippo_r20 | -0.150833 |
| mean_current_r100_minus_hippo_r100 | -0.1475 |
