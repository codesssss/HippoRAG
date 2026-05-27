# Evidence Transition GraphRAG v4 Fact-Witnessed STO Retrieval

| field | value |
| --- | --- |
| dataset | popqa |
| method | evidence_transition_graphragv4_fact_witnessed_sto |
| rows | 100 |
| fresh OpenIE | /mnt/nvme/code/HippoRAG/run_logs/evidenceflow_nq_popqa_limit100_32b_no_think_gpt4omini_20260518/etv4_clean_mainline/popqa/index/openie_results_ner_qwen3-32b-judge.json |
| fresh embeddings | /mnt/nvme/code/HippoRAG/run_logs/evidenceflow_nq_popqa_limit100_32b_no_think_gpt4omini_20260518/etv4_clean_mainline/popqa/index/qwen3-32b-judge_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
| candidate source | fresh_dense_seeded_sto_query_local_sto_graph |

| metric | value |
| --- | ---: |
| R@5 | 0.6050 |
| all-gold@5 | 0.2100 |
| mean certified docs@5 | 3.40 |
