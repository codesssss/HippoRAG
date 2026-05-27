# Evidence Transition GraphRAG v4 Fact-Witnessed STO Summary

| dataset | rows | R@5 | EM | F1 | candidate source | fresh OpenIE | fresh embeddings |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| musique | 100 | 0.7108 |  |  | fresh_dense_seeded_sto_query_local_sto_graph | /mnt/nvme/code/HippoRAG/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/index/openie_results_ner_qwen3-32b-judge.json | /mnt/nvme/code/HippoRAG/run_logs/etv4_readout_grid_direct_musique_limit100_20260514/transition_valid_closure/musique/index/qwen3-32b-judge_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet |
