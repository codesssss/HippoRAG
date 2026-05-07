# Reviewer Baseline Hard-Slice Analysis - 2026-05-06

Offline gold-support-count slice over the same aligned full1000 rows as the main reviewer-baseline paired CI report.
Bootstrap is query-paired percentile bootstrap with `10000` resamples. Delta is left method minus right method.

Primary pre-specified hard slice: `gold_doc_count>=3`. Exact-count and `>=4` rows are included only to expose sample size and trend; they are not new tuned decision boundaries.

Support R@5 is the same unified title-multiset `R5_TITLE` metric used in `paired_ci.md`.

## DAEC-selective vs SetR-style k20, Answer F1

| Dataset | Slice | N | Left | Right | Delta | 95% CI | Excludes 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | all | 1000 | 0.7118 | 0.6936 | 0.0182 | [-0.0017, 0.0381] | False |
| 2Wiki | gold_doc_count=2 | 765 | 0.6494 | 0.6396 | 0.0098 | [-0.0119, 0.0325] | False |
| 2Wiki | gold_doc_count>=3 | 235 | 0.9149 | 0.8695 | 0.0454 | [0.0014, 0.0894] | True |
| 2Wiki | gold_doc_count=3 | 0 | -- | -- | -- | -- | -- |
| 2Wiki | gold_doc_count>=4 | 235 | 0.9149 | 0.8695 | 0.0454 | [0.0000, 0.0894] | True |
| HotpotQA | all | 1000 | 0.7473 | 0.7552 | -0.0079 | [-0.0222, 0.0064] | False |
| HotpotQA | gold_doc_count=2 | 1000 | 0.7473 | 0.7552 | -0.0079 | [-0.0223, 0.0062] | False |
| HotpotQA | gold_doc_count>=3 | 0 | -- | -- | -- | -- | -- |
| HotpotQA | gold_doc_count=3 | 0 | -- | -- | -- | -- | -- |
| HotpotQA | gold_doc_count>=4 | 0 | -- | -- | -- | -- | -- |
| MuSiQue | all | 1000 | 0.4548 | 0.4761 | -0.0212 | [-0.0438, 0.0015] | False |
| MuSiQue | gold_doc_count=2 | 518 | 0.5525 | 0.5858 | -0.0333 | [-0.0639, -0.0031] | True |
| MuSiQue | gold_doc_count>=3 | 482 | 0.3499 | 0.3582 | -0.0083 | [-0.0431, 0.0260] | False |
| MuSiQue | gold_doc_count=3 | 316 | 0.3889 | 0.3937 | -0.0048 | [-0.0474, 0.0361] | False |
| MuSiQue | gold_doc_count>=4 | 166 | 0.2755 | 0.2906 | -0.0150 | [-0.0739, 0.0451] | False |

## DAEC-selective vs SetR-style k20, Unified Support R@5

| Dataset | Slice | N | Left | Right | Delta | 95% CI | Excludes 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | all | 1000 | 0.9410 | 0.9425 | -0.0015 | [-0.0120, 0.0088] | False |
| 2Wiki | gold_doc_count=2 | 765 | 0.9444 | 0.9497 | -0.0052 | [-0.0170, 0.0065] | False |
| 2Wiki | gold_doc_count>=3 | 235 | 0.9298 | 0.9191 | 0.0106 | [-0.0106, 0.0319] | False |
| 2Wiki | gold_doc_count=3 | 0 | -- | -- | -- | -- | -- |
| 2Wiki | gold_doc_count>=4 | 235 | 0.9298 | 0.9191 | 0.0106 | [-0.0106, 0.0319] | False |
| HotpotQA | all | 1000 | 0.9625 | 0.9735 | -0.0110 | [-0.0200, -0.0020] | True |
| HotpotQA | gold_doc_count=2 | 1000 | 0.9625 | 0.9735 | -0.0110 | [-0.0200, -0.0020] | True |
| HotpotQA | gold_doc_count>=3 | 0 | -- | -- | -- | -- | -- |
| HotpotQA | gold_doc_count=3 | 0 | -- | -- | -- | -- | -- |
| HotpotQA | gold_doc_count>=4 | 0 | -- | -- | -- | -- | -- |
| MuSiQue | all | 1000 | 0.7745 | 0.7837 | -0.0092 | [-0.0230, 0.0047] | False |
| MuSiQue | gold_doc_count=2 | 518 | 0.8774 | 0.8948 | -0.0174 | [-0.0367, 0.0019] | False |
| MuSiQue | gold_doc_count>=3 | 482 | 0.6639 | 0.6642 | -0.0003 | [-0.0201, 0.0197] | False |
| MuSiQue | gold_doc_count=3 | 316 | 0.7468 | 0.7363 | 0.0105 | [-0.0137, 0.0348] | False |
| MuSiQue | gold_doc_count>=4 | 166 | 0.5060 | 0.5271 | -0.0211 | [-0.0542, 0.0120] | False |

## Primary Hard Slice F1 Against Reviewer Baselines

| Dataset | Slice | N | Left | Right | Delta | 95% CI | Excludes 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | gold_doc_count>=3 | 235 | 0.9149 | 0.8128 | 0.1021 | [0.0511, 0.1532] | True |
| HotpotQA | gold_doc_count>=3 | 0 | -- | -- | -- | -- | -- |
| MuSiQue | gold_doc_count>=3 | 482 | 0.3499 | 0.3133 | 0.0366 | [0.0065, 0.0671] | True |

## Primary Hard Slice F1: DAEC-selective - IRCoT-style local

| Dataset | Slice | N | Left | Right | Delta | 95% CI | Excludes 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | gold_doc_count>=3 | 235 | 0.9149 | 0.7204 | 0.1945 | [0.1350, 0.2584] | True |
| HotpotQA | gold_doc_count>=3 | 0 | -- | -- | -- | -- | -- |
| MuSiQue | gold_doc_count>=3 | 482 | 0.3499 | 0.3407 | 0.0091 | [-0.0277, 0.0451] | False |

## Primary Hard Slice F1: DAEC-selective - LLM-direct title

| Dataset | Slice | N | Left | Right | Delta | 95% CI | Excludes 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | gold_doc_count>=3 | 235 | 0.9149 | 0.6920 | 0.2229 | [0.1633, 0.2811] | True |
| HotpotQA | gold_doc_count>=3 | 0 | -- | -- | -- | -- | -- |
| MuSiQue | gold_doc_count>=3 | 482 | 0.3499 | 0.2945 | 0.0554 | [0.0166, 0.0950] | True |

## Primary Hard Slice F1: DAEC-selective - LLM-direct snippet128

| Dataset | Slice | N | Left | Right | Delta | 95% CI | Excludes 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | gold_doc_count>=3 | 235 | 0.9149 | 0.8328 | 0.0821 | [0.0335, 0.1319] | True |
| HotpotQA | gold_doc_count>=3 | 0 | -- | -- | -- | -- | -- |
| MuSiQue | gold_doc_count>=3 | 482 | 0.3499 | 0.3024 | 0.0474 | [0.0073, 0.0872] | True |

## Interpretation

- HotpotQA has no `gold_doc_count>=3` rows in this aligned subset, so it cannot support a deep-composition slice defense; it should be treated as a shallow/saturated contrast dataset.
- On 2Wiki, the primary hard slice is exactly the 4-document subset (`N=235`), where DAEC-selective beats SetR-style on answer F1 with a 95% CI excluding zero; unified support R@5 remains tied.
- On MuSiQue, SetR-style remains stronger on 2-document questions, while the `gold_doc_count>=3` slice is statistically tied on answer F1 and support R@5. Selective binding narrows the base DAEC gap but does not overturn SetR-style.
- DAEC-selective remains clearly stronger than Top5 and both LLM-direct controls on the pre-specified hard slices where those slices exist; the hard MuSiQue comparison to IRCoT-style local is positive but not significant on answer F1.
