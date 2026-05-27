# Evidence Transition GraphRAG v3 Variable-Flow Retrieval

| field | value |
| --- | --- |
| dataset | 2wikimultihopqa |
| method | evidence_transition_graphragv3_variable_flow |
| rows | 1000 |
| fresh OpenIE | /mnt/nvme/code/HippoRAG/run_logs/evidence_transition_graphragv3_variable_flow_qwen32b_nv2_full1000_20260511/2wikimultihopqa/index/openie_results_ner_qwen3-32b-judge.json |
| fresh embeddings | /mnt/nvme/code/HippoRAG/run_logs/evidence_transition_graphragv3_variable_flow_qwen32b_nv2_full1000_20260511/2wikimultihopqa/index/qwen3-32b-judge_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
| candidate source | fresh_dense_seeded_sto_query_local_sto_graph |

| metric | value |
| --- | ---: |
| R@5 | 0.9055 |
| all-gold@5 | 0.7140 |
| mean certified docs@5 | 2.15 |
