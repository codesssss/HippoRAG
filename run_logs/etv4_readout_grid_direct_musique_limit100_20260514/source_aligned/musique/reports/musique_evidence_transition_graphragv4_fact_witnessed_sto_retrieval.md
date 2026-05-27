# Evidence Transition GraphRAG v4 Fact-Witnessed STO Retrieval

| field | value |
| --- | --- |
| dataset | musique |
| method | evidence_transition_graphragv4_fact_witnessed_sto |
| rows | 100 |
| fresh OpenIE | /mnt/nvme/code/HippoRAG/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/index/openie_results_ner_qwen3-32b-judge.json |
| fresh embeddings | /mnt/nvme/code/HippoRAG/run_logs/etv4_readout_grid_direct_musique_limit100_20260514/source_aligned/musique/index/qwen3-32b-judge_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
| candidate source | fresh_dense_seeded_sto_query_local_sto_graph |

| metric | value |
| --- | ---: |
| R@5 | 0.7142 |
| all-gold@5 | 0.3900 |
| mean certified docs@5 | 2.94 |
