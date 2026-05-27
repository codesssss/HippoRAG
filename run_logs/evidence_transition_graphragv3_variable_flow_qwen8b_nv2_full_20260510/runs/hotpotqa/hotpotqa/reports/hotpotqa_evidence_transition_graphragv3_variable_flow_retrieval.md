# Evidence Transition GraphRAG v3 Variable-Flow Retrieval

| field | value |
| --- | --- |
| dataset | hotpotqa |
| method | evidence_transition_graphragv3_variable_flow |
| rows | 1000 |
| fresh OpenIE | /mnt/nvme/code/HippoRAG/run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_full_20260510/runs/hotpotqa/hotpotqa/index/openie_results_ner_qwen3-8b-train.json |
| fresh embeddings | /mnt/nvme/code/HippoRAG/run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_full_20260510/runs/hotpotqa/hotpotqa/index/qwen3-8b-train_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
| candidate source | fresh_dense_seeded_sto_query_local_sto_graph |

| metric | value |
| --- | ---: |
| R@5 | 0.9505 |
| all-gold@5 | 0.9050 |
| mean certified docs@5 | 3.25 |
