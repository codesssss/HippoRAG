# Fresh Query-Grounded STO GraphRAG Retrieval

| field | value |
| --- | --- |
| dataset | 2wikimultihopqa |
| method | evidence_transition_graph_native_retrieval |
| runner | query_grounded_sto_graph_native |
| rows | 5 |
| fresh OpenIE | /mnt/nvme/code/HippoRAG/run_logs/evidence_transition_qwen8b_nv2_smoke_20260509/2wikimultihopqa/index/openie_results_ner_qwen3-8b-train.json |
| fresh chunk embeddings | /mnt/nvme/code/HippoRAG/run_logs/evidence_transition_qwen8b_nv2_smoke_20260509/2wikimultihopqa/index/qwen3-8b-train_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
| candidate source | fresh_dense_seeded_sto_query_grounded_sto |

| metric | value |
| --- | ---: |
| R@5 | 1.0000 |
| all-gold@5 | 1.0000 |
| mean certified docs@5 | 1.80 |
