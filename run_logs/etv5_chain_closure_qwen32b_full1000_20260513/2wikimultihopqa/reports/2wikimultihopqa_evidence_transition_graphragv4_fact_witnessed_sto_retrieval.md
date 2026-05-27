# Evidence Transition GraphRAG v4 Fact-Witnessed STO Retrieval

| field | value |
| --- | --- |
| dataset | 2wikimultihopqa |
| method | evidence_transition_graphragv4_fact_witnessed_sto |
| rows | 1000 |
| fresh OpenIE | /mnt/nvme/code/HippoRAG/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/2wikimultihopqa/index/openie_results_ner_qwen3-32b-judge.json |
| fresh embeddings | /mnt/nvme/code/HippoRAG/run_logs/etv5_chain_closure_qwen32b_full1000_20260513/2wikimultihopqa/index/qwen3-32b-judge_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
| candidate source | fresh_dense_seeded_sto_query_local_sto_graph |

| metric | value |
| --- | ---: |
| R@5 | 0.9350 |
| all-gold@5 | 0.8260 |
| mean certified docs@5 | 2.36 |
