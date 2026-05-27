# Evidence Transition GraphRAG v3 Variable-Flow Retrieval

| field | value |
| --- | --- |
| dataset | musique |
| method | evidence_transition_graphragv3_variable_flow |
| rows | 100 |
| fresh OpenIE | /mnt/nvme/code/HippoRAG/run_logs/evidence_transition_qwen8b_nv2_limit100_musique_20260509/musique/index/openie_results_ner_qwen3-8b-train.json |
| fresh embeddings | run_logs/evidence_transition_graphragv4_layered_transition_musique_limit100_20260509/musique/index/qwen3-8b-train_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
| candidate source | fresh_dense_seeded_sto_query_local_sto_graph |

| metric | value |
| --- | ---: |
| R@5 | 0.5967 |
| all-gold@5 | 0.3000 |
| mean certified docs@5 | 3.73 |
