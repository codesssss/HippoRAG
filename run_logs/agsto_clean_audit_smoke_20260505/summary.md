# AG-STO Clean-RRF Audit

## Config

| parameter | value |
| --- | --- |
| datasets | 2wikimultihopqa |
| limit | 3 |
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
| selection_policy_counts | {'sto_proposal_consensus': 3} |
| consensus_nonempty_rate | 1.0 |
| selection_by_consensus_rate | 1.0 |
| completion_applied_rate | 0.666667 |
| final_changed_by_completion_rate | 0.666667 |

### Retrieval Metrics

| variant | R@5 | R@20 | R@100 |
| --- | --- | --- | --- |
| current_agsto | 0.833333 | 0.833333 | 1.0 |
| hipporag | 0.833333 | 1.0 | 1.0 |
| current_pre_completion | 0.666667 | 0.666667 | 0.666667 |
| weighted_rrf_main | 0.666667 | 0.833333 | 1.0 |
| unweighted_rrf | 0.666667 | 0.833333 | 1.0 |

### Channel Health

| channel | nonempty@20 | nonempty@100 | mean_len | R@5 | R@20 | R@100 |
| --- | --- | --- | --- | --- | --- | --- |
| specificity | 1.0 | 1.0 | 100.0 | 0.666667 | 0.666667 | 0.833333 |
| hybrid_residual | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neighborhood | 1.0 | 1.0 | 100.0 | 0.833333 | 0.833333 | 1.0 |
| support_set | 1.0 | 1.0 | 10.0 | 0.333333 | 0.5 | 0.5 |

### Channel Overlap

| pair | Jaccard@20 | Jaccard@100 |
| --- | --- | --- |
| specificity__hybrid_residual | 0.0 | 0.0 |
| specificity__neighborhood | 0.375661 | 0.591427 |
| specificity__support_set | 0.114286 | 0.047346 |
| hybrid_residual__neighborhood | 0.0 | 0.0 |
| hybrid_residual__support_set | 0.0 | 0.0 |
| neighborhood__support_set | 0.373077 | 0.1 |

### Unweighted Leave-One-Out

| variant | R@5 | R@20 | R@100 | Delta R@5 |
| --- | --- | --- | --- | --- |
| minus_specificity | 0.666667 | 0.833333 | 0.833333 | 0.0 |
| minus_hybrid_residual | 0.666667 | 0.833333 | 1.0 | 0.0 |
| minus_neighborhood | 0.666667 | 0.833333 | 1.0 | 0.0 |
| minus_support_set | 0.833333 | 0.833333 | 1.0 | 0.166666 |

### Weighted Leave-One-Out

| variant | R@5 | R@20 | R@100 | Delta R@5 |
| --- | --- | --- | --- | --- |
| minus_specificity | 0.5 | 0.833333 | 0.833333 | -0.166667 |
| minus_hybrid_residual | 0.666667 | 0.833333 | 1.0 | 0.0 |
| minus_neighborhood | 0.5 | 0.833333 | 1.0 | -0.166667 |
| minus_support_set | 0.833333 | 0.833333 | 1.0 | 0.166666 |

### Support Weight Sweep

| support_weight | R@5 | R@20 | R@100 |
| --- | --- | --- | --- |
| 1.0 | 0.666667 | 0.833333 | 1.0 |
| 1.25 | 0.666667 | 0.833333 | 1.0 |
| 1.5 | 0.666667 | 0.833333 | 1.0 |
| 1.75 | 0.666667 | 0.833333 | 1.0 |
| 2.0 | 0.666667 | 0.833333 | 1.0 |
| 2.25 | 0.666667 | 0.833333 | 1.0 |

### Feasibility

| metric | value |
| --- | --- |
| contains_anchor_at5 | 1.0 |
| connected_at5 | 0.0 |
| hub_violation_at5 | 0.0 |

### AG-STO vs HippoRAG

| metric | value |
| --- | --- |
| hippo_hit_agsto_miss | 1 |
| both_hit | 1 |
| agsto_hit_hippo_miss | 1 |
| mean_current_r5_minus_hippo_r5 | 0.0 |
| mean_current_r20_minus_hippo_r20 | -0.166667 |
| mean_current_r100_minus_hippo_r100 | 0.0 |
