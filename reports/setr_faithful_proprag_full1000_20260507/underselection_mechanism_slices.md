# SetR-faithful Under-Selection Mechanism Slices

Date: 2026-05-07

This analysis uses no new LLM calls. It aligns existing DAEC-selective, SetR-faithful, and SetR+Fill@5 full1000 rows by question occurrence, then compares SetR-faithful's effective reader passage count against each query's gold support count.

`under-select` means:

```text
SetR-faithful selected passage count < gold_doc_count
```

This is a count-based diagnostic only; it does not claim the selected passages are the correct supports.

## Under-Selection Rates

| Dataset | Slice | N | Mean gold docs | Mean SetR-faithful passages | Under-selected | Under-select rate |
|---|---|---:|---:|---:|---:|---:|
| 2Wiki | all | 1000 | 2.470 | 2.577 | 157 | 15.7% |
| 2Wiki | gold_doc_count=2 | 765 | 2.000 | 2.325 | 35 | 4.6% |
| 2Wiki | 4-doc hard subset | 235 | 4.000 | 3.396 | 122 | 51.9% |
| HotpotQA | all / gold_doc_count=2 | 1000 | 2.000 | 2.740 | 32 | 3.2% |
| MuSiQue | all | 1000 | 2.648 | 3.622 | 130 | 13.0% |
| MuSiQue | gold_doc_count=2 | 518 | 2.000 | 3.197 | 24 | 4.6% |
| MuSiQue | gold_doc_count>=3 | 482 | 3.344 | 4.079 | 106 | 22.0% |

## Conditional DAEC-selective vs SetR-faithful

Query-paired percentile bootstrap with 10,000 resamples. Delta is DAEC-selective minus SetR-faithful.

| Dataset | Slice | N | dF1 | 95% CI | dR5_TITLE | 95% CI | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| 2Wiki | 4-doc hard subset | 235 | +0.1437 | [0.0889, 0.1995] | +0.1468 | [0.1181, 0.1755] | DAEC strongly wins |
| 2Wiki | 4-doc, SetR selected >= gold_count | 113 | -0.0088 | [-0.0619, 0.0531] | -0.0155 | [-0.0487, 0.0155] | tied; SetR point higher |
| 2Wiki | 4-doc, SetR selected < gold_count | 122 | +0.2851 | [0.1976, 0.3698] | +0.2971 | [0.2705, 0.3238] | DAEC gap is concentrated here |
| HotpotQA | all, SetR selected >= gold_count | 968 | +0.0003 | [-0.0188, 0.0186] | +0.0253 | [0.0129, 0.0377] | answer tied; DAEC support higher |
| HotpotQA | all, SetR selected < gold_count | 32 | +0.1091 | [-0.0208, 0.2513] | +0.4219 | [0.3438, 0.5000] | small under-selected slice |
| MuSiQue | gold_doc_count>=3 | 482 | +0.0190 | [-0.0205, 0.0580] | +0.0883 | [0.0647, 0.1119] | answer tied; DAEC support higher |
| MuSiQue | gold_doc_count>=3, SetR selected >= gold_count | 376 | +0.0013 | [-0.0437, 0.0464] | +0.0654 | [0.0388, 0.0920] | answer tied; DAEC support higher |
| MuSiQue | gold_doc_count>=3, SetR selected < gold_count | 106 | +0.0819 | [0.0025, 0.1643] | +0.1698 | [0.1234, 0.2162] | DAEC significantly wins |

## SetR Selected-Enough Subsets

These rows answer whether DAEC still wins when SetR-faithful selects at least four passages.

| Dataset | Slice | N | DAEC-selective F1 | SetR-faithful F1 | dF1 | 95% CI | Conclusion |
|---|---|---:|---:|---:|---:|---:|---|
| 2Wiki | SetR selected >=4 | 177 | 0.7850 | 0.7945 | -0.0095 | [-0.0622, 0.0426] | tied |
| 2Wiki | 4-doc hard subset and SetR selected >=4 | 113 | 0.9115 | 0.9204 | -0.0088 | [-0.0619, 0.0531] | tied |
| HotpotQA | SetR selected >=4 | 172 | 0.6869 | 0.6890 | -0.0021 | [-0.0503, 0.0465] | tied |
| MuSiQue | SetR selected >=4 | 414 | 0.4087 | 0.3803 | +0.0284 | [-0.0131, 0.0699] | tied |

## Interpretation

- The under-selection story is strongest on the 2Wiki 4-doc hard subset: SetR-faithful selects fewer than 4 passages for 122/235 queries (51.9%), and DAEC-selective wins those under-selected cases by +0.285 F1.
- When SetR-faithful selects enough passages on the same 2Wiki hard subset, DAEC-selective is statistically tied and slightly lower by point estimate. This means the large 2Wiki hard-slice gain is primarily a coverage-under-selection effect, not a claim that DAEC dominates SetR whenever SetR selects enough evidence.
- MuSiQue shows the same direction but weaker magnitude. On gold_doc_count>=3, SetR-faithful under-selects 106/482 queries (22.0%); DAEC-selective significantly wins that under-selected subset by +0.082 F1, while the selected-enough subset is answer-tied.
- HotpotQA is a shallow/saturated contrast: all aligned examples have two gold docs and SetR-faithful under-selects only 32/1000 queries (3.2%). This explains why answer F1 is tied even though DAEC has higher support R@5.
- Paper wording should therefore say that DAEC's advantage is concentrated in deep compositional cases where adaptive prompt-only selection under-selects required supports. It should not claim that DAEC beats SetR-faithful after conditioning on SetR selecting enough passages.

## Paper-Facing Sentence

On deep compositional queries, the gap is explained by systematic under-selection in adaptive prompt-only selection: SetR-faithful chooses fewer than four passages for 51.9% of 2Wiki 4-doc queries, and DAEC-selective improves F1 by +0.285 on exactly those cases. When SetR-faithful selects enough passages, answer F1 is statistically tied, indicating that DAEC's primary advantage is enforcing sufficient structured evidence coverage rather than universally outperforming prompt-only selection on already-covered queries.
