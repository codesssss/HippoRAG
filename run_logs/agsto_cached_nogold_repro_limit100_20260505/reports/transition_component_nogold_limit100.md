# Transition Component / Completion Retriever

This evaluates generic transition retrieval controls over the Source/Title/OpenIE substrate. It does not consume HippoRAGv2 traces, PPR scores, SFB gates, or typed query-family routing.

## Config

| parameter | value |
| --- | ---: |
| `seed_unit_k` | 200 |
| `max_queries` | 100 |
| `max_endpoint_degree` | 30 |
| `max_candidate_units` | 600 |
| `max_docs_before_penalty` | 5 |
| `retrieval_top_k` | 20 |
| `max_components` | 40 |
| `bm25_anchor_k` | 10 |
| `max_completion_docs` | 200 |
| `support_candidate_docs` | 400 |
| `support_set_size` | 2 |
| `support_beam_size` | 12 |
| `context_anchor_k` | 10 |
| `hybrid_bm25_anchor_k` | 0 |
| `semantic_residual_weight` | 12.0 |
| `native_dense_anchor` | True |
| `native_dense_anchor_top_k` | 20 |
| `chunk_embedding_path` | None |
| `embedding_base_url` | http://localhost:8018/v1/embeddings |
| `embedding_model` | /mnt/nvme/Qwen3-Embedding-8B |
| `embedding_batch_size` | 8 |
| `embedding_timeout` | 120.0 |
| `dense_query_instruction_mode` | raw |
| `dense_query_view_mode` | raw |
| `dense_query_instruction` | Given a question, retrieve relevant documents that best answer the question. |

## Retrieval Metrics

| dataset | units | endpoints | BM25 R@5 | BM25 R@10 | native dense R@5 | native dense R@10 | transition R@5 | transition R@10 | pair-emission R@5 | pair-emission R@10 | residual-pair R@5 | residual-pair R@10 | semantic-pair R@5 | semantic-pair R@10 | specificity-pair R@5 | specificity-pair R@10 | endpoint-transition R@5 | endpoint-transition R@10 | hybrid-residual R@5 | hybrid-residual R@10 | neighborhood R@5 | neighborhood R@10 | support-set R@5 | support-set R@10 | anchor-evidence R@5 | anchor-evidence R@10 | candidate recall | TCR-v1 R@5 | TCR-v1 R@10 | context R@5 | context R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | 74709 | 51586 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hotpotqa | 143702 | 99158 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| musique | 151271 | 104411 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation Guard

- This v1 uses IDF-weighted query seeding, strong endpoint component expansion, and transition-aware doc ranking.
- BM25 is a document-level lexical control over the same STO token inventory, not a graph method.
- BM25+transition keeps BM25 as the query entry and uses strong STO endpoint neighborhoods only for completion.
- Pair-emission emits anchor-completion support pairs to test whether graph completion can enter the reader-facing top5.
- Specificity-pair keeps aggregate semantic pair scoring but regularizes it with endpoint specificity and hub pressure.
- Endpoint-transition readout scores individual `(anchor, endpoint, completion)` transitions so endpoint specificity can compete with residual semantic support.
- Query-conditioned neighborhood first builds a bounded local STO graph from dense/lexical anchors, then selects connected support pages by residual demand. This is the upstream replacement target for PPR-style global diffusion.
- Support-set search ranks small connected passage sets, then emits their member passages. It is the first retrieval head whose ranked object is a multi-passage support set rather than a passage or pair.
- Anchor-guided evidence selection uses stable native anchors as entry points, scores evidence sets by coverage/connectivity/specificity/support agreement, and treats the reader-facing top5 as the selected object.
- A low score does not invalidate the transition substrate oracle; it means component activation/scoring is still weak.
- The primary retrieval target is reader-facing top5 support quality: R@5 and all-gold@5.
- R@10 is retained only as a residual diagnostic for whether useful evidence sits just below the reader-facing window. It is not an optimization objective.
- R@200 is intentionally omitted.
