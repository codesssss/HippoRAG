# Evidence Transition GraphRAG v4 Fact-Witnessed STO Summary

| dataset | rows | R@5 | EM | F1 | candidate source | fresh OpenIE | fresh embeddings |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2wikimultihopqa | 2 | 1.0000 |  |  | fresh_dense_seeded_sto_query_local_sto_graph | /mnt/nvme/code/HippoRAG/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/2wikimultihopqa/index/openie_results_ner_qwen3-32b-judge.json | run_logs/etv4_qwen32b_guard_repair_smoke_20260513/2wikimultihopqa/index/qwen3-32b-judge_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
