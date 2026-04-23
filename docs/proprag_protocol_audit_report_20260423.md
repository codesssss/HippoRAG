# PropRAG / DtC Protocol Audit Report

Date: 2026-04-23

## Bottom Line

- HippoRAG baseline vs DtC inside each DtC JSON is a valid same-pool comparison.
- PropRAG clean no-think 100 vs DtC is aligned on broad runtime conditions, but not on fixed candidate pool.
- PropRAG should be labeled as a cross-system same-reader/same-embedding baseline unless it is adapted to consume HippoRAG's top-100 pool.

## Query Alignment

| Dataset | Dataset File First-100 | Saved Examples Compared | Saved Examples Match | Notes |
| --- | ---: | ---: | ---: | --- |
| 2Wiki | True | 10 | True | PropRAG output serializes only first 10 examples |
| HotpotQA | True | 10 | True | PropRAG output serializes only first 10 examples |
| MuSiQue | True | 10 | True | PropRAG output serializes only first 10 examples |

## Protocol Checks

| Status | Check | Evidence |
| --- | --- | --- |
| PASS | 2Wiki: pilot limit | both reports use limit=100 |
| PASS | 2Wiki: reader model | qwen3-8b-train |
| PASS | 2Wiki: embedding model | VLLM/nvidia/NV-Embed-v2 |
| PASS | 2Wiki: embedding endpoint | http://localhost:8019/v1/embeddings |
| PASS | 2Wiki: top-100 budget | retrieval_top_k/setwise_pool_k=100 |
| PASS | 2Wiki: QA context width | PropRAG docs and DtC trace top titles are width 5 |
| PASS | 2Wiki: PropRAG max_new_tokens | 2048 |
| WARN | 2Wiki: DtC QA max_new_tokens | DtC report does not serialize qa max_new_tokens; build_config currently sets BaseConfig.max_new_tokens=None. |
| PASS | HotpotQA: pilot limit | both reports use limit=100 |
| PASS | HotpotQA: reader model | qwen3-8b-train |
| PASS | HotpotQA: embedding model | VLLM/nvidia/NV-Embed-v2 |
| PASS | HotpotQA: embedding endpoint | http://localhost:8019/v1/embeddings |
| PASS | HotpotQA: top-100 budget | retrieval_top_k/setwise_pool_k=100 |
| PASS | HotpotQA: QA context width | PropRAG docs and DtC trace top titles are width 5 |
| PASS | HotpotQA: PropRAG max_new_tokens | 2048 |
| WARN | HotpotQA: DtC QA max_new_tokens | DtC report does not serialize qa max_new_tokens; build_config currently sets BaseConfig.max_new_tokens=None. |
| PASS | MuSiQue: pilot limit | both reports use limit=100 |
| PASS | MuSiQue: reader model | qwen3-8b-train |
| PASS | MuSiQue: embedding model | VLLM/nvidia/NV-Embed-v2 |
| PASS | MuSiQue: embedding endpoint | http://localhost:8019/v1/embeddings |
| PASS | MuSiQue: top-100 budget | retrieval_top_k/setwise_pool_k=100 |
| PASS | MuSiQue: QA context width | PropRAG docs and DtC trace top titles are width 5 |
| PASS | MuSiQue: PropRAG max_new_tokens | 2048 |
| WARN | MuSiQue: DtC QA max_new_tokens | DtC report does not serialize qa max_new_tokens; build_config currently sets BaseConfig.max_new_tokens=None. |
| WARN | fixed-pool comparability | PropRAG retrieves from its own proposition graph; DtC selects from HippoRAG top-100. This is not same fixed-pool composition. |

## Gold-Leakage / No-Think Checks

| Status | Check | Evidence |
| --- | --- | --- |
| PASS | PropRAG QA prompt source | qa() body uses query_solution.docs[:qa_top_k], not gold fields |
| PASS | PropRAG gold_docs retrieval use | retrieved docs are built before recall evaluation: top_k_docs line 353, recall line 358 |
| PASS | 2Wiki: PropRAG examples gold leakage fields | only gold_answers is serialized; no gold_docs in examples |
| PASS | HotpotQA: PropRAG examples gold leakage fields | only gold_answers is serialized; no gold_docs in examples |
| PASS | MuSiQue: PropRAG examples gold leakage fields | only gold_answers is serialized; no gold_docs in examples |
| PASS | PropRAG QA no-think | /mnt/nvme/code/PropRAG/src/proprag/PropRAG.py |
| PASS | EnhancedOpenIE no-think | /mnt/nvme/code/PropRAG/src/proprag/information_extraction/enhanced_openie.py |
| PASS | Proposition extraction no-think | /mnt/nvme/code/PropRAG/src/proprag/information_extraction/proposition_extraction.py |
| PASS | OpenAI wrapper strips think envelope | /mnt/nvme/code/PropRAG/src/proprag/llm/openai_gpt.py |
| PASS | PropRAG clean text artifacts contain think tags | no <think> / </think> tags found in JSON/log/OpenIE text artifacts |

## Key Caveats

- PropRAG JSON currently saves only 10 examples, so saved-example query alignment is partial. Dataset-file first-100 alignment is the stronger check.
- PropRAG uses its own proposition graph retrieval. This is intentionally stronger/different than a same-pool selector baseline.
- DtC reports do not serialize QA `max_new_tokens`; the current HippoRAG `build_config` sets `BaseConfig.max_new_tokens=None`, while PropRAG explicitly uses 2048.
