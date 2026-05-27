# Evidence Transition GraphRAG v4 Fact-Witnessed STO Retrieval

| field | value |
| --- | --- |
| dataset | musique |
| method | evidence_transition_graphragv4_fact_witnessed_sto |
| rows | 5 |
| fresh OpenIE | /mnt/nvme/code/HippoRAG/run_logs/evidence_transition_qwen8b_nv2_limit100_musique_20260509/musique/index/openie_results_ner_qwen3-8b-train.json |
| fresh embeddings | run_logs/evidence_transition_graphragv4_fact_witnessed_sto_smoke_musique_limit5_20260510/musique/index/qwen3-8b-train_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
| candidate source | fresh_dense_seeded_sto_query_local_sto_graph |

| metric | value |
| --- | ---: |
| R@5 | 0.9333 |
| all-gold@5 | 0.8000 |
| mean certified docs@5 | 2.20 |
