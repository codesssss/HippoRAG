# Evidence Transition GraphRAG v4 Fact-Witnessed STO Retrieval

| field | value |
| --- | --- |
| dataset | 2wikimultihopqa |
| method | evidence_transition_graphragv4_fact_witnessed_sto |
| rows | 100 |
| fresh OpenIE | /mnt/nvme/zly/HippoRAG/run_logs/fresh_query_grounded_sto_gpt4omini_limit100_20260508/2wikimultihopqa/index/openie_results_ner_gpt-4o-mini.json |
| fresh embeddings | run_logs/evidence_transition_graphragv4_branch_balanced_gpt4omini_3x100_20260511/2wikimultihopqa/index/gpt-4o-mini_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
| candidate source | fresh_dense_seeded_sto_query_local_sto_graph |

| metric | value |
| --- | ---: |
| R@5 | 0.9400 |
| all-gold@5 | 0.8500 |
| mean certified docs@5 | 2.41 |
