# Fresh Query-Grounded STO GraphRAG Retrieval

| field | value |
| --- | --- |
| dataset | musique |
| method | evidence_transition_v2_graph_native_retrieval |
| runner | evidence_transition_v2 |
| rows | 100 |
| fresh OpenIE | /mnt/nvme/code/HippoRAG/run_logs/evidence_transition_qwen8b_nv2_limit100_musique_20260509/musique/index/openie_results_ner_qwen3-8b-train.json |
| fresh chunk embeddings | /mnt/nvme/code/HippoRAG/run_logs/evidence_transition_qwen8b_nv2_limit100_musique_20260509/musique/index/qwen3-8b-train_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
| candidate source | fresh_dense_seeded_sto_query_grounded_sto |

| metric | value |
| --- | ---: |
| R@5 | 0.4825 |
| all-gold@5 | 0.1900 |
| mean certified docs@5 | 4.11 |
