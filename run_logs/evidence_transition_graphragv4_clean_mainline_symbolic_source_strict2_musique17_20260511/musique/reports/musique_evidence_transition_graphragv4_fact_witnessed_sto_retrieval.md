# Evidence Transition GraphRAG v4 Fact-Witnessed STO Retrieval

| field | value |
| --- | --- |
| dataset | musique |
| method | evidence_transition_graphragv4_fact_witnessed_sto |
| rows | 17 |
| fresh OpenIE | /mnt/nvme/code/HippoRAG/run_logs/evidence_transition_graphragv4_clean_mainline_symbolic_source_strict2_musique17_20260511/musique/index/openie_results_ner_gpt-4o-mini.json |
| fresh embeddings | run_logs/evidence_transition_graphragv4_clean_mainline_symbolic_source_strict2_musique17_20260511/musique/index/gpt-4o-mini_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
| candidate source | fresh_dense_seeded_sto_query_local_sto_graph |

| metric | value |
| --- | ---: |
| R@5 | 0.6520 |
| all-gold@5 | 0.3529 |
| mean certified docs@5 | 2.29 |
