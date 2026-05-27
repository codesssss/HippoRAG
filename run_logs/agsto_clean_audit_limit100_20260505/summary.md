# AG-STO Clean-RRF Audit

## Config

| parameter | value |
| --- | --- |
| datasets | 2wikimultihopqa,hotpotqa,musique |
| limit | 100 |
| pool_k | 100 |
| candidate_limit | 180 |
| proposal_candidate_depth | 24 |
| support_proposal_depth | 12 |
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
| completion_applied_rate | 0.16 |
| final_changed_by_completion_rate | 0.16 |

### Retrieval Metrics

| variant | R@5 | R@20 | R@100 |
| --- | --- | --- | --- |
| current_agsto | 0.8375 | 0.8975 | 0.9125 |
| hipporag | 0.815 | 0.9 | 0.9425 |
| current_pre_completion | 0.825 | 0.825 | 0.825 |
| weighted_rrf_main | 0.825 | 0.955 | 0.96 |
| unweighted_rrf | 0.8725 | 0.955 | 0.96 |

### Channel Health

| channel | nonempty@20 | nonempty@100 | mean_len | R@5 | R@20 | R@100 |
| --- | --- | --- | --- | --- | --- | --- |
| specificity | 1.0 | 1.0 | 100.0 | 0.88 | 0.885 | 0.9 |
| hybrid_residual | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neighborhood | 1.0 | 1.0 | 100.0 | 0.855 | 0.9375 | 0.965 |
| support_set | 1.0 | 1.0 | 11.16 | 0.605 | 0.6825 | 0.6825 |

### Channel Overlap

| pair | Jaccard@20 | Jaccard@100 |
| --- | --- | --- |
| specificity__hybrid_residual | 0.0 | 0.0 |
| specificity__neighborhood | 0.295504 | 0.590125 |
| specificity__support_set | 0.115426 | 0.041161 |
| hybrid_residual__neighborhood | 0.0 | 0.0 |
| hybrid_residual__support_set | 0.0 | 0.0 |
| neighborhood__support_set | 0.423972 | 0.111376 |

### Unweighted Leave-One-Out

| variant | R@5 | R@20 | R@100 | Delta R@5 |
| --- | --- | --- | --- | --- |
| minus_specificity | 0.805 | 0.9425 | 0.9425 | -0.0675 |
| minus_hybrid_residual | 0.8725 | 0.955 | 0.96 | 0.0 |
| minus_neighborhood | 0.84 | 0.945 | 0.95 | -0.0325 |
| minus_support_set | 0.89 | 0.9425 | 0.955 | 0.0175 |

### Weighted Leave-One-Out

| variant | R@5 | R@20 | R@100 | Delta R@5 |
| --- | --- | --- | --- | --- |
| minus_specificity | 0.655 | 0.9425 | 0.9425 | -0.17 |
| minus_hybrid_residual | 0.825 | 0.955 | 0.96 | 0.0 |
| minus_neighborhood | 0.725 | 0.945 | 0.95 | -0.1 |
| minus_support_set | 0.89 | 0.9425 | 0.955 | 0.065 |

### Support Weight Sweep

| support_weight | R@5 | R@20 | R@100 |
| --- | --- | --- | --- |
| 1.0 | 0.8725 | 0.955 | 0.96 |
| 1.25 | 0.8625 | 0.955 | 0.96 |
| 1.5 | 0.845 | 0.955 | 0.96 |
| 1.75 | 0.825 | 0.955 | 0.96 |
| 2.0 | 0.8075 | 0.955 | 0.96 |
| 2.25 | 0.7925 | 0.955 | 0.96 |

### Feasibility

| metric | value |
| --- | --- |
| contains_anchor_at5 | 1.0 |
| connected_at5 | 0.3 |
| hub_violation_at5 | 0.42 |

### AG-STO vs HippoRAG

| metric | value |
| --- | --- |
| hippo_hit_agsto_miss | 14 |
| both_hit | 43 |
| agsto_hit_hippo_miss | 20 |
| both_miss | 23 |
| mean_current_r5_minus_hippo_r5 | 0.0225 |
| mean_current_r20_minus_hippo_r20 | -0.0025 |
| mean_current_r100_minus_hippo_r100 | -0.03 |

## hotpotqa

### Path Usage
| metric | value |
| --- | --- |
| selection_policy_counts | {'sto_proposal_consensus': 100} |
| consensus_nonempty_rate | 1.0 |
| selection_by_consensus_rate | 1.0 |
| completion_applied_rate | 0.22 |
| final_changed_by_completion_rate | 0.22 |

### Retrieval Metrics

| variant | R@5 | R@20 | R@100 |
| --- | --- | --- | --- |
| current_agsto | 0.87 | 0.965 | 0.985 |
| hipporag | 0.925 | 0.995 | 1.0 |
| current_pre_completion | 0.85 | 0.85 | 0.85 |
| weighted_rrf_main | 0.85 | 0.965 | 0.98 |
| unweighted_rrf | 0.85 | 0.965 | 0.98 |

### Channel Health

| channel | nonempty@20 | nonempty@100 | mean_len | R@5 | R@20 | R@100 |
| --- | --- | --- | --- | --- | --- | --- |
| specificity | 1.0 | 1.0 | 100.0 | 0.84 | 0.955 | 0.98 |
| hybrid_residual | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neighborhood | 1.0 | 1.0 | 100.0 | 0.79 | 0.925 | 0.99 |
| support_set | 1.0 | 1.0 | 12.22 | 0.665 | 0.735 | 0.735 |

### Channel Overlap

| pair | Jaccard@20 | Jaccard@100 |
| --- | --- | --- |
| specificity__hybrid_residual | 0.0 | 0.0 |
| specificity__neighborhood | 0.287006 | 0.595485 |
| specificity__support_set | 0.226895 | 0.073299 |
| hybrid_residual__neighborhood | 0.0 | 0.0 |
| hybrid_residual__support_set | 0.0 | 0.0 |
| neighborhood__support_set | 0.480875 | 0.1222 |

### Unweighted Leave-One-Out

| variant | R@5 | R@20 | R@100 | Delta R@5 |
| --- | --- | --- | --- | --- |
| minus_specificity | 0.815 | 0.94 | 0.945 | -0.035 |
| minus_hybrid_residual | 0.85 | 0.965 | 0.98 | 0.0 |
| minus_neighborhood | 0.805 | 0.97 | 0.98 | -0.045 |
| minus_support_set | 0.835 | 0.96 | 0.98 | -0.015 |

### Weighted Leave-One-Out

| variant | R@5 | R@20 | R@100 | Delta R@5 |
| --- | --- | --- | --- | --- |
| minus_specificity | 0.68 | 0.94 | 0.945 | -0.17 |
| minus_hybrid_residual | 0.85 | 0.965 | 0.98 | 0.0 |
| minus_neighborhood | 0.73 | 0.97 | 0.98 | -0.12 |
| minus_support_set | 0.835 | 0.96 | 0.98 | -0.015 |

### Support Weight Sweep

| support_weight | R@5 | R@20 | R@100 |
| --- | --- | --- | --- |
| 1.0 | 0.85 | 0.965 | 0.98 |
| 1.25 | 0.85 | 0.965 | 0.98 |
| 1.5 | 0.85 | 0.965 | 0.98 |
| 1.75 | 0.85 | 0.965 | 0.98 |
| 2.0 | 0.82 | 0.965 | 0.98 |
| 2.25 | 0.82 | 0.965 | 0.98 |

### Feasibility

| metric | value |
| --- | --- |
| contains_anchor_at5 | 1.0 |
| connected_at5 | 0.55 |
| hub_violation_at5 | 0.45 |

### AG-STO vs HippoRAG

| metric | value |
| --- | --- |
| both_miss | 8 |
| hippo_hit_agsto_miss | 16 |
| both_hit | 69 |
| agsto_hit_hippo_miss | 7 |
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
| completion_applied_rate | 0.27 |
| final_changed_by_completion_rate | 0.27 |

### Retrieval Metrics

| variant | R@5 | R@20 | R@100 |
| --- | --- | --- | --- |
| current_agsto | 0.504167 | 0.69 | 0.803333 |
| hipporag | 0.67 | 0.845833 | 0.955833 |
| current_pre_completion | 0.499167 | 0.499167 | 0.499167 |
| weighted_rrf_main | 0.499167 | 0.680833 | 0.774167 |
| unweighted_rrf | 0.500833 | 0.680833 | 0.774167 |

### Channel Health

| channel | nonempty@20 | nonempty@100 | mean_len | R@5 | R@20 | R@100 |
| --- | --- | --- | --- | --- | --- | --- |
| specificity | 1.0 | 1.0 | 100.0 | 0.506667 | 0.68 | 0.789167 |
| hybrid_residual | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neighborhood | 1.0 | 1.0 | 100.0 | 0.485833 | 0.595833 | 0.831667 |
| support_set | 1.0 | 1.0 | 12.12 | 0.456667 | 0.53 | 0.53 |

### Channel Overlap

| pair | Jaccard@20 | Jaccard@100 |
| --- | --- | --- |
| specificity__hybrid_residual | 0.0 | 0.0 |
| specificity__neighborhood | 0.281009 | 0.553743 |
| specificity__support_set | 0.196798 | 0.06489 |
| hybrid_residual__neighborhood | 0.0 | 0.0 |
| hybrid_residual__support_set | 0.0 | 0.0 |
| neighborhood__support_set | 0.481904 | 0.120976 |

### Unweighted Leave-One-Out

| variant | R@5 | R@20 | R@100 | Delta R@5 |
| --- | --- | --- | --- | --- |
| minus_specificity | 0.4875 | 0.614167 | 0.635 | -0.013333 |
| minus_hybrid_residual | 0.500833 | 0.680833 | 0.774167 | 0.0 |
| minus_neighborhood | 0.5075 | 0.6775 | 0.7425 | 0.006667 |
| minus_support_set | 0.5 | 0.675833 | 0.774167 | -0.000833 |

### Weighted Leave-One-Out

| variant | R@5 | R@20 | R@100 | Delta R@5 |
| --- | --- | --- | --- | --- |
| minus_specificity | 0.456667 | 0.614167 | 0.635 | -0.0425 |
| minus_hybrid_residual | 0.499167 | 0.680833 | 0.774167 | 0.0 |
| minus_neighborhood | 0.485 | 0.675 | 0.7425 | -0.014167 |
| minus_support_set | 0.5 | 0.675833 | 0.774167 | 0.000833 |

### Support Weight Sweep

| support_weight | R@5 | R@20 | R@100 |
| --- | --- | --- | --- |
| 1.0 | 0.500833 | 0.680833 | 0.774167 |
| 1.25 | 0.500833 | 0.680833 | 0.774167 |
| 1.5 | 0.499167 | 0.680833 | 0.774167 |
| 1.75 | 0.499167 | 0.680833 | 0.774167 |
| 2.0 | 0.493333 | 0.680833 | 0.774167 |
| 2.25 | 0.4825 | 0.680833 | 0.774167 |

### Feasibility

| metric | value |
| --- | --- |
| contains_anchor_at5 | 1.0 |
| connected_at5 | 0.45 |
| hub_violation_at5 | 0.49 |

### AG-STO vs HippoRAG

| metric | value |
| --- | --- |
| both_hit | 17 |
| both_miss | 55 |
| agsto_hit_hippo_miss | 7 |
| hippo_hit_agsto_miss | 21 |
| mean_current_r5_minus_hippo_r5 | -0.165833 |
| mean_current_r20_minus_hippo_r20 | -0.155833 |
| mean_current_r100_minus_hippo_r100 | -0.1525 |
