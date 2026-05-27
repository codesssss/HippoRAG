# Evidence Transition GraphRAG v4 Fact-Witnessed STO Retrieval

| field | value |
| --- | --- |
| dataset | hotpotqa |
| method | evidence_transition_graphragv4_fact_witnessed_sto |
| rows | 1000 |
| fresh OpenIE | /mnt/nvme/code/HippoRAG/run_logs/evidence_transition_graphragv4_clean_mainline_multi_anchor_strict_musique100_20260511/hotpotqa/index/openie_results_ner_gpt-4o-mini.json |
| fresh embeddings | /mnt/nvme/code/HippoRAG/run_logs/etv4_no_multi_anchor_precision_full1000_reuse_gpt4omini_20260511/hotpotqa/index/gpt-4o-mini_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
| candidate source | fresh_dense_seeded_sto_query_local_sto_graph |

| metric | value |
| --- | ---: |
| R@5 | 0.9505 |
| all-gold@5 | 0.9060 |
| mean certified docs@5 | 3.13 |
