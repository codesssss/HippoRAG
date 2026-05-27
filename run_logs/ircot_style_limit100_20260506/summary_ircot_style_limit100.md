# IRCoT-Style Limit100 Baseline

Status: completed

This is a local IRCoT-style iterative retrieval baseline, not an official IRCoT reproduction. It uses the current HippoRAG/Qwen/NV-Embed stack instead of the official IRCoT Elasticsearch/retriever-server pipeline.

Protocol:

- Datasets: 2WikiMultihopQA, HotpotQA, MuSiQue
- Limit: 100 queries per dataset
- Iterations: 3
- Top-k per iteration: 5
- Final reader budget: top 5 documents
- Final document order: round_robin
- LLM: qwen3-8b-train
- Embedding: VLLM/nvidia/NV-Embed-v2
- Retriever substrate: HippoRAG retrieval with the local LLM fact reranker

## Raw Results

| Dataset | EM | F1 | Support R@5 | LLM Calls / Query | Latency / Query |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 0.570 | 0.615 | 0.850 | 3.0 | 11.46s |
| HotpotQA | 0.550 | 0.652 | 0.900 | 3.0 | 19.01s |
| MuSiQue | 0.370 | 0.461 | 0.642 | 3.0 | 17.08s |

Notes:

- `Support R@5` is the script's trace-level `supporting_paragraph_recall`, computed over the final 5 reader documents.
- The `retrieval` object in the HotpotQA JSON reports much lower `Recall@5` than trace-level support recall. The trace-level value is the comparable value for this run because it uses the same final top-5 document strings used by the reader.

## Comparison Against Current PropRAG Limit100 Rows

| Dataset | Method | EM | F1 | R@5 |
|---|---|---:|---:|---:|
| 2Wiki | Top5 | 0.580 | 0.632 | 0.935 |
| 2Wiki | DAEC | 0.580 | 0.643 | 0.953 |
| 2Wiki | IRCoT-style | 0.570 | 0.615 | 0.850 |
| HotpotQA | Top5 | 0.570 | 0.691 | 0.930 |
| HotpotQA | DAEC | 0.570 | 0.691 | 0.950 |
| HotpotQA | IRCoT-style | 0.550 | 0.652 | 0.900 |
| MuSiQue | Top5 | 0.380 | 0.437 | 0.698 |
| MuSiQue | DAEC | 0.390 | 0.466 | 0.738 |
| MuSiQue | IRCoT-style | 0.370 | 0.461 | 0.642 |

## Delta vs DAEC

| Dataset | Delta EM | Delta F1 | Delta R@5 |
|---|---:|---:|---:|
| 2Wiki | -0.010 | -0.028 | -0.103 |
| HotpotQA | -0.020 | -0.039 | -0.050 |
| MuSiQue | -0.020 | -0.005 | -0.096 |

## Takeaway

Under this local protocol, IRCoT-style iterative retrieval does not beat DAEC on the three limit100 datasets. It adds iterative LLM query generation and three retrieval rounds, but the final top-5 support recall is lower than DAEC in all three cases. This is a useful reviewer baseline: direct iterative retrieval is not an automatic substitute for demand-aware evidence composition under the same reader budget.

