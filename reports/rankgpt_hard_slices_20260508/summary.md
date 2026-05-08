# RankGPT-Style Sliding Hard-Slice Analysis

Date: 2026-05-08

Purpose: close the reviewer-facing hard-slice gap by comparing DAEC-selective against the corrected RankGPT-style sliding-window local adaptation on support-depth slices.

This report reuses existing full1000 reader outputs; no new LLM calls are made. Bootstrap is query-paired percentile bootstrap with 10,000 resamples. Delta is left method minus right method.

The primary hard slice is 2Wiki `gold_doc_count>=3`, which is exactly the 4-document subset in this aligned full1000 split (`N=235`).

## Primary 2Wiki 4-Doc Hard Slice

| Dataset | Slice | N | Comparison | Metric | Left | Right | Delta | 95% CI | Excludes 0 |
|---|---|---:|---|---|---:|---:|---:|---:|---|
| 2Wiki | gold_doc_count>=3 | 235 | DAEC-selective - RankGPT-style sliding | EM | 0.9149 | 0.8553 | +0.0596 | [+0.0128, +0.1064] | yes |
| 2Wiki | gold_doc_count>=3 | 235 | DAEC-selective - RankGPT-style sliding | F1 | 0.9149 | 0.8567 | +0.0582 | [+0.0142, +0.1035] | yes |
| 2Wiki | gold_doc_count>=3 | 235 | DAEC-selective - RankGPT-style sliding | R5_TITLE | 0.9298 | 0.9053 | +0.0245 | [+0.0021, +0.0468] | yes |
| 2Wiki | gold_doc_count>=3 | 235 | SetR-faithful - RankGPT-style sliding | EM | 0.7660 | 0.8553 | -0.0894 | [-0.1489, -0.0298] | yes |
| 2Wiki | gold_doc_count>=3 | 235 | SetR-faithful - RankGPT-style sliding | F1 | 0.7712 | 0.8567 | -0.0856 | [-0.1452, -0.0260] | yes |
| 2Wiki | gold_doc_count>=3 | 235 | SetR-faithful - RankGPT-style sliding | R5_TITLE | 0.7830 | 0.9053 | -0.1223 | [-0.1500, -0.0947] | yes |
| 2Wiki | gold_doc_count>=3 | 235 | DAEC-selective - SetR-faithful | EM | 0.9149 | 0.7660 | +0.1489 | [+0.0936, +0.2085] | yes |
| 2Wiki | gold_doc_count>=3 | 235 | DAEC-selective - SetR-faithful | F1 | 0.9149 | 0.7712 | +0.1437 | [+0.0879, +0.1991] | yes |
| 2Wiki | gold_doc_count>=3 | 235 | DAEC-selective - SetR-faithful | R5_TITLE | 0.9298 | 0.7830 | +0.1468 | [+0.1170, +0.1755] | yes |

## DAEC vs RankGPT-Style Sliding by Support Depth

| Dataset | Slice | N | Comparison | Metric | Left | Right | Delta | 95% CI | Excludes 0 |
|---|---|---:|---|---|---:|---:|---:|---:|---|
| 2Wiki | all | 1000 | DAEC-selective - RankGPT-style sliding | F1 | 0.7118 | 0.6659 | +0.0460 | [+0.0240, +0.0672] | yes |
| 2Wiki | gold_doc_count=2 | 765 | DAEC-selective - RankGPT-style sliding | F1 | 0.6494 | 0.6072 | +0.0422 | [+0.0176, +0.0671] | yes |
| 2Wiki | gold_doc_count>=3 | 235 | DAEC-selective - RankGPT-style sliding | F1 | 0.9149 | 0.8567 | +0.0582 | [+0.0142, +0.1035] | yes |
| 2Wiki | gold_doc_count>=4 | 235 | DAEC-selective - RankGPT-style sliding | F1 | 0.9149 | 0.8567 | +0.0582 | [+0.0156, +0.1050] | yes |
| MuSiQue | all | 1000 | DAEC-selective - RankGPT-style sliding | F1 | 0.4548 | 0.4093 | +0.0455 | [+0.0181, +0.0723] | yes |
| MuSiQue | gold_doc_count=2 | 518 | DAEC-selective - RankGPT-style sliding | F1 | 0.5525 | 0.5082 | +0.0443 | [+0.0064, +0.0830] | yes |
| MuSiQue | gold_doc_count>=3 | 482 | DAEC-selective - RankGPT-style sliding | F1 | 0.3499 | 0.3031 | +0.0468 | [+0.0086, +0.0863] | yes |
| MuSiQue | gold_doc_count>=4 | 166 | DAEC-selective - RankGPT-style sliding | F1 | 0.2755 | 0.2299 | +0.0457 | [-0.0165, +0.1076] | no |

## Paper-Facing Takeaway

- On the 2Wiki 4-document hard slice, DAEC-selective beats RankGPT-style sliding by `+0.0582` F1 with a 95% CI `[+0.0142, +0.1035]`, so the hard-slice advantage holds against the strongest RankGPT-style local adaptation (CI excluding zero).
- The earlier large hard-slice win over SetR-faithful remains strong: DAEC-selective beats SetR-faithful by `+0.1437` F1 with a 95% CI `[+0.0879, +0.1991]`.
- RankGPT-style sliding is substantially stronger than SetR-faithful on this hard slice (`SetR-faithful - RankGPT-style sliding = -0.0856` F1, 95% CI `[-0.1452, -0.0260]`), so this is a meaningful stronger-baseline check rather than a weak-baseline artifact.
- The safe main-paper claim is therefore: DAEC strongly repairs SetR-style under-selection on 2Wiki 4-doc questions and still significantly outperforms RankGPT-style sliding on the same hard slice under the controlled Qwen3-8B `/no_think` substrate.

## Claim Boundary

Allowed:

```text
On 2Wiki 4-document queries, DAEC-selective dramatically outperforms SetR-faithful
and significantly outperforms RankGPT-style sliding-window reranking under the
same Qwen3-8B /no_think substrate.
```

Not allowed:

```text
DAEC outperforms original GPT-3.5/4 RankGPT on hard multi-hop questions.
```

## Files

- CSV: `reports/rankgpt_hard_slices_20260508/hard_slice_ci.csv`
- JSON: `reports/rankgpt_hard_slices_20260508/hard_slice_ci.json`
