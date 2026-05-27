# Evidence Transition GraphRAG v4 Fact-Witnessed STO Retrieval

| field | value |
| --- | --- |
| dataset | nq_rear |
| method | evidence_transition_graphragv4_fact_witnessed_sto |
| rows | 1000 |
| fresh OpenIE | /mnt/nvme/code/HippoRAG/run_logs/nq_popqa_non_neocor_resume_32b_no_think_gpt4omini_full1000_20260517/etv4_clean_mainline/nq_rear/index/openie_results_ner_qwen3-32b-judge.json |
| fresh embeddings | run_logs/nq_popqa_non_neocor_resume_32b_no_think_gpt4omini_full1000_20260517/etv4_clean_mainline/nq_rear/index/qwen3-32b-judge_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
| candidate source | fresh_dense_seeded_sto_query_local_sto_graph |

| metric | value |
| --- | ---: |
| R@5 | 0.7233 |
| all-gold@5 | 0.3970 |
| mean certified docs@5 | 3.70 |
