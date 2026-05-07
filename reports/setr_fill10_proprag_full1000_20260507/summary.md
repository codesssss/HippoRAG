# SetR-Fill@10 PropRAG Full1000

Date: 2026-05-07

Purpose: add a bounded `SetR-Fill@10` ablation for the official SetR repository's conversion-style behavior. The official `convert_rankify.py` uses `k=10`: parse ranks from `Final Selection`, then fill missing slots from the original top-20 in rank order until 10 contexts are written. Because the SetR README evaluation section is `TBD`, this is reported as an official conversion-style ablation, not as proof of the paper-faithful reader setting.

All rows reuse the existing PropRAG full1000 SetR selection JSONL from `run_logs/setr_full1000_20260503`; no selector calls were rerun. Fill@10 changes only the conversion into the reader pool and the reader budget (`qa_top_k=10`).

## Fill@10 Selection Sanity

| Dataset | Records | Parse success | Parse failure | Empty fallback | Avg selected | Avg rank-fill | Avg reader passages |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 1000 | 1000 | 0 | 0 | 2.577 | 7.431 | 10.0 |
| HotpotQA | 1000 | 1000 | 0 | 0 | 2.740 | 7.277 | 10.0 |
| MuSiQue | 1000 | 1000 | 0 | 0 | 3.622 | 6.412 | 10.0 |

## Mean Metrics

EM/F1 are answer metrics. R@5/R@10 are unified title-multiset support recall computed from each method's final reader titles. DAEC-selective has a strict 5-document reader set, so its R@10 equals R@5.

| Dataset | Method | EM | F1 | R@5 title | R@10 title |
|---|---|---:|---:|---:|---:|
| 2Wiki | DAEC-selective | 0.6420 | 0.7118 | 0.9410 | 0.9410 |
| 2Wiki | SetR-faithful | 0.6030 | 0.6746 | 0.8840 | 0.8840 |
| 2Wiki | SetR-Fill@5 | 0.6240 | 0.6936 | 0.9425 | 0.9425 |
| 2Wiki | SetR-Fill@10 | 0.6420 | 0.7114 | 0.9425 | 0.9547 |
| 2Wiki | Top-10 retriever | 0.6240 | 0.6962 | 0.9028 | 0.9393 |
| HotpotQA | DAEC-selective | 0.6200 | 0.7473 | 0.9625 | 0.9625 |
| HotpotQA | SetR-faithful | 0.6250 | 0.7435 | 0.9245 | 0.9245 |
| HotpotQA | SetR-Fill@5 | 0.6290 | 0.7552 | 0.9735 | 0.9735 |
| HotpotQA | SetR-Fill@10 | 0.6200 | 0.7400 | 0.9735 | 0.9890 |
| HotpotQA | Top-10 retriever | 0.6180 | 0.7449 | 0.9520 | 0.9855 |
| MuSiQue | DAEC-selective | 0.3530 | 0.4548 | 0.7745 | 0.7745 |
| MuSiQue | SetR-faithful | 0.3440 | 0.4467 | 0.6909 | 0.6909 |
| MuSiQue | SetR-Fill@5 | 0.3770 | 0.4761 | 0.7837 | 0.7837 |
| MuSiQue | SetR-Fill@10 | 0.3990 | 0.5035 | 0.7837 | 0.8533 |
| MuSiQue | Top-10 retriever | 0.3780 | 0.4828 | 0.7378 | 0.8288 |

## DAEC-Selective vs SetR-Fill@10

Query-paired percentile bootstrap with 10,000 resamples. Delta is DAEC-selective minus SetR-Fill@10.

| Dataset | Metric | DAEC-selective | SetR-Fill@10 | Delta | 95% CI | P(delta > 0) | Excludes 0 |
|---|---|---:|---:|---:|---:|---:|---|
| 2Wiki | EM | 0.6420 | 0.6420 | +0.0000 | [-0.0210, 0.0210] | 0.491 | no |
| 2Wiki | F1 | 0.7118 | 0.7114 | +0.0004 | [-0.0187, 0.0198] | 0.521 | no |
| 2Wiki | R5_TITLE | 0.9410 | 0.9425 | -0.0015 | [-0.0120, 0.0090] | 0.378 | no |
| 2Wiki | R10_TITLE | 0.9410 | 0.9547 | -0.0138 | [-0.0240, -0.0040] | 0.003 | yes |
| HotpotQA | EM | 0.6200 | 0.6200 | +0.0000 | [-0.0180, 0.0180] | 0.483 | no |
| HotpotQA | F1 | 0.7473 | 0.7400 | +0.0073 | [-0.0091, 0.0233] | 0.810 | no |
| HotpotQA | R5_TITLE | 0.9625 | 0.9735 | -0.0110 | [-0.0195, -0.0020] | 0.008 | yes |
| HotpotQA | R10_TITLE | 0.9625 | 0.9890 | -0.0265 | [-0.0350, -0.0185] | 0.000 | yes |
| MuSiQue | EM | 0.3530 | 0.3990 | -0.0460 | [-0.0720, -0.0210] | 0.000 | yes |
| MuSiQue | F1 | 0.4548 | 0.5035 | -0.0487 | [-0.0729, -0.0245] | 0.000 | yes |
| MuSiQue | R5_TITLE | 0.7745 | 0.7837 | -0.0092 | [-0.0231, 0.0047] | 0.098 | no |
| MuSiQue | R10_TITLE | 0.7745 | 0.8533 | -0.0788 | [-0.0925, -0.0653] | 0.000 | yes |

## Top-10 Retriever Budget Probe

`Top-10 retriever` gives the reader the original PropRAG rank top-10 with no LLM selection. This isolates the effect of larger reader budget plus retriever recall from SetR's selected-plus-fallback conversion.

| Dataset | DAEC F1 | Top-10 F1 | SetR-Fill@10 F1 | dF1 DAEC-Top10 | 95% CI | dF1 Fill@10-Top10 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.7118 | 0.6962 | 0.7114 | +0.0156 | [-0.0039, 0.0349] | +0.0151 | [0.0036, 0.0273] |
| HotpotQA | 0.7473 | 0.7449 | 0.7400 | +0.0024 | [-0.0129, 0.0180] | -0.0049 | [-0.0164, 0.0066] |
| MuSiQue | 0.4548 | 0.4828 | 0.5035 | -0.0280 | [-0.0510, -0.0051] | +0.0207 | [0.0010, 0.0406] |

## SetR Variant Deltas

| Dataset | Comparison | Metric | Left | Right | Delta | 95% CI | Excludes 0 |
|---|---|---|---:|---:|---:|---:|---|
| 2Wiki | SetR-faithful - SetR-Fill@5 | F1 | 0.6746 | 0.6936 | -0.0190 | [-0.0372, -0.0015] | yes |
| 2Wiki | SetR-faithful - SetR-Fill@10 | F1 | 0.6746 | 0.7114 | -0.0368 | [-0.0568, -0.0174] | yes |
| 2Wiki | SetR-Fill@5 - SetR-Fill@10 | F1 | 0.6936 | 0.7114 | -0.0178 | [-0.0323, -0.0035] | yes |
| 2Wiki | SetR-Fill@10 - Top-10 retriever | F1 | 0.7114 | 0.6962 | +0.0151 | [0.0036, 0.0273] | yes |
| HotpotQA | SetR-faithful - SetR-Fill@5 | F1 | 0.7435 | 0.7552 | -0.0117 | [-0.0267, 0.0033] | no |
| HotpotQA | SetR-faithful - SetR-Fill@10 | F1 | 0.7435 | 0.7400 | +0.0035 | [-0.0138, 0.0210] | no |
| HotpotQA | SetR-Fill@5 - SetR-Fill@10 | F1 | 0.7552 | 0.7400 | +0.0152 | [0.0009, 0.0299] | yes |
| HotpotQA | SetR-Fill@10 - Top-10 retriever | F1 | 0.7400 | 0.7449 | -0.0049 | [-0.0164, 0.0066] | no |
| MuSiQue | SetR-faithful - SetR-Fill@5 | F1 | 0.4467 | 0.4761 | -0.0294 | [-0.0474, -0.0115] | yes |
| MuSiQue | SetR-faithful - SetR-Fill@10 | F1 | 0.4467 | 0.5035 | -0.0569 | [-0.0791, -0.0351] | yes |
| MuSiQue | SetR-Fill@5 - SetR-Fill@10 | F1 | 0.4761 | 0.5035 | -0.0275 | [-0.0475, -0.0076] | yes |
| MuSiQue | SetR-Fill@10 - Top-10 retriever | F1 | 0.5035 | 0.4828 | +0.0207 | [0.0010, 0.0406] | yes |

## 2Wiki 4-Doc Hard Slice

The hard slice is the same pre-specified `gold_doc_count>=3` slice used in the SetR-faithful report. In the aligned 2Wiki full1000 split this is the 4-document subset (`N=235`).

| Comparison | Metric | Left | Right | Delta | 95% CI | Excludes 0 |
|---|---|---:|---:|---:|---:|---|
| DAEC-selective - SetR-faithful | EM | 0.9149 | 0.7660 | +0.1489 | [0.0936, 0.2043] | yes |
| DAEC-selective - SetR-faithful | F1 | 0.9149 | 0.7712 | +0.1437 | [0.0894, 0.2005] | yes |
| DAEC-selective - SetR-faithful | R5_TITLE | 0.9298 | 0.7830 | +0.1468 | [0.1170, 0.1755] | yes |
| DAEC-selective - SetR-faithful | R10_TITLE | 0.9298 | 0.7830 | +0.1468 | [0.1181, 0.1755] | yes |
| DAEC-selective - SetR-Fill@5 | EM | 0.9149 | 0.8681 | +0.0468 | [0.0043, 0.0936] | yes |
| DAEC-selective - SetR-Fill@5 | F1 | 0.9149 | 0.8695 | +0.0454 | [0.0000, 0.0894] | yes |
| DAEC-selective - SetR-Fill@5 | R5_TITLE | 0.9298 | 0.9191 | +0.0106 | [-0.0117, 0.0319] | no |
| DAEC-selective - SetR-Fill@5 | R10_TITLE | 0.9298 | 0.9191 | +0.0106 | [-0.0117, 0.0319] | no |
| DAEC-selective - SetR-Fill@10 | EM | 0.9149 | 0.8979 | +0.0170 | [-0.0255, 0.0596] | no |
| DAEC-selective - SetR-Fill@10 | F1 | 0.9149 | 0.8991 | +0.0158 | [-0.0255, 0.0584] | no |
| DAEC-selective - SetR-Fill@10 | R5_TITLE | 0.9298 | 0.9191 | +0.0106 | [-0.0117, 0.0319] | no |
| DAEC-selective - SetR-Fill@10 | R10_TITLE | 0.9298 | 0.9415 | -0.0117 | [-0.0309, 0.0074] | no |
| SetR-Fill@5 - SetR-Fill@10 | EM | 0.8681 | 0.8979 | -0.0298 | [-0.0638, 0.0000] | no |
| SetR-Fill@5 - SetR-Fill@10 | F1 | 0.8695 | 0.8991 | -0.0296 | [-0.0624, 0.0016] | no |
| SetR-Fill@5 - SetR-Fill@10 | R5_TITLE | 0.9191 | 0.9191 | +0.0000 | [0.0000, 0.0000] | no |
| SetR-Fill@5 - SetR-Fill@10 | R10_TITLE | 0.9191 | 0.9415 | -0.0223 | [-0.0330, -0.0128] | yes |

## Interpretation

- `SetR-Fill@10` closes the full-set 2Wiki answer gap almost completely: DAEC-selective F1 0.7118 vs Fill@10 F1 0.7114, paired CI crosses zero. This should be framed as DAEC top-5 matching the larger official-conversion-style SetR reader budget, not as a significant DAEC win.
- On HotpotQA, DAEC-selective is higher than Fill@10 on F1 by +0.0073, but the paired CI crosses zero. Fill@10 is not monotonically stronger than Fill@5 here.
- On MuSiQue, pure Top-10 retriever F1 is 0.4828, between DAEC-selective (0.4548) and SetR-Fill@10 (0.5035). Thus the Fill@10 advantage is partly a larger-reader-budget/retriever-recall effect, but not fully explained by raw Top-10 rank order.
- SetR-Fill@10 beats Top-10 retriever by +0.0207 F1 on MuSiQue, showing that selected-plus-fallback ordering adds value under a 10-document budget. This does not change the fairness interpretation: the relevant DAEC-matched comparison remains Fill@5.
- The 2Wiki 4-doc hard slice remains the strongest DAEC evidence against the faithful adaptive SetR setting and the budget-matched Fill@5 setting. Against Fill@10, the answer gap is no longer significant, but Fill@10 uses twice DAEC's reader budget.
- Paper framing: primary SetR baseline is `SetR-faithful`; `SetR-Fill@5` is DAEC-budget-matched; `SetR-Fill@10` is official conversion-style and should be reported to close the repo-code attack point.
