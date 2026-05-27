# IRCoT Protocol Audit, 2026-05-19

## Verdict

The completed IRCoT run is useful as an IRCoT-style iterative query-rewriting baseline under the shared reader protocol, but it is not a paper-faithful reproduction of original IRCoT. Each iteration calls `HippoRAG.retrieve()`, which uses the HippoRAG graph/PPR retriever over the Qwen3-32B OpenIE index, not a plain dense/BM25 retriever.

The completed run was also not strict `enable_thinking=false`. The prompts included `/no_think`, but the direct Qwen HTTP call did not pass `chat_template_kwargs.enable_thinking=false`. A scan found 3 generated follow-up queries with visible `<think>...</think>` remnants out of 6,000 generated follow-up queries.

## Current Result Table

All rows use GPT-4o-mini reader, top-5 reader context, 1,000 queries.

| Method | Dataset | R@5 | EM | F1 |
| --- | --- | ---: | ---: | ---: |
| Dense | HotpotQA | 93.35 | 59.50 | 73.01 |
| HippoRAG 2 | HotpotQA | 92.60 | 59.30 | 72.51 |
| IRCoT-style | HotpotQA | 94.60 | 60.70 | 74.29 |
| Dense | 2WikiMultiHopQA | 75.85 | 55.00 | 60.49 |
| HippoRAG 2 | 2WikiMultiHopQA | 82.75 | 56.70 | 63.43 |
| IRCoT-style | 2WikiMultiHopQA | 92.08 | 62.80 | 71.27 |
| Dense | MuSiQue | 68.25 | 35.20 | 46.19 |
| HippoRAG 2 | MuSiQue | 70.96 | 34.90 | 46.54 |
| IRCoT-style | MuSiQue | 73.48 | 39.60 | 50.28 |

## Iteration Effect

R@5 computed from the saved `examples[*].docs` and title-level gold support titles.

| Dataset | Step-1 R@5 | Final R@5 | Delta | Queries with final top-5 changed |
| --- | ---: | ---: | ---: | ---: |
| HotpotQA | 92.60 | 94.60 | +2.00 | 865 / 1000 |
| 2WikiMultiHopQA | 82.73 | 92.08 | +9.35 | 962 / 1000 |
| MuSiQue | 70.96 | 73.48 | +2.53 | 939 / 1000 |

## No-Think Audit

Endpoint test on `http://localhost:8046/v1/chat/completions`:

| Request body | Visible thinking output |
| --- | --- |
| `/no_think` only | Produced an empty `<think>...</think>` envelope before JSON |
| `/no_think` plus `chat_template_kwargs: {enable_thinking: false}` | Produced JSON directly |

Code patch applied after the audit:

- `scripts/bsgs_run_ircot_baseline.py` now sends top-level `chat_template_kwargs.enable_thinking=false` in the direct HTTP call and strips `<think>...</think>` blocks before parsing follow-up queries.
- `src/hipporag/llm/openai_gpt.py` now passes `extra_body.chat_template_kwargs.enable_thinking=false` for Qwen requests when `HIPPORAG_RERANK_FORCE_NO_THINK=1` is set or the prompt contains `/no_think`.

These patches affect future reruns only. The completed full1000 IRCoT result above was generated before the strict no-think patch.

## Metric Notes

HotpotQA originally had a broken exact full-document matching metric (`R@5=10.65`) because dataset support strings and corpus strings differed by normalization. The corrected current result uses title/corpus alignment (`R@5=94.60`, `All@5=90.10`).

MuSiQue has a stale retrieval field in the saved IRCoT reader-input JSON (`R@5=70.54`). Recomputing from `examples[*].gold_titles` and `examples[*].docs`, which is what the reader report does, gives `R@5=73.48`. The paper should use the reader report value or the stale input metric should be synchronized before automated table generation.

## Protocol Recommendation

Do not label this row as a direct IRCoT reproduction. Recommended label:

`IRCoT-style + HippoRAG retriever`

Recommended footnote:

`IRCoT is implemented as iterative follow-up query generation over our shared HippoRAG/NV-Embed-v2 retrieval backend and evaluated with the same GPT-4o-mini top-5 reader; it is a controlled IRCoT-style adaptation rather than a paper-faithful reproduction.`
