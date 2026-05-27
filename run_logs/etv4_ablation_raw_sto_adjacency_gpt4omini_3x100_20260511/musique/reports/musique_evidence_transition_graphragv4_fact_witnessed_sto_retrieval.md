# Evidence Transition GraphRAG v4 Fact-Witnessed STO Retrieval

| field | value |
| --- | --- |
| dataset | musique |
| method | evidence_transition_graphragv4_fact_witnessed_sto |
| rows | 100 |
| fresh OpenIE | /mnt/nvme/code/HippoRAG/run_logs/etv4_ablation_raw_sto_adjacency_gpt4omini_3x100_20260511/musique/index/openie_results_ner_gpt-4o-mini.json |
| fresh embeddings | run_logs/etv4_ablation_raw_sto_adjacency_gpt4omini_3x100_20260511/musique/index/gpt-4o-mini_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
| candidate source | fresh_dense_seeded_sto_query_local_sto_graph |

| metric | value |
| --- | ---: |
| R@5 | 0.7358 |
| all-gold@5 | 0.4300 |
| mean certified docs@5 | 2.73 |
