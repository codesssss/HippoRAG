# 2Wiki 4-Doc Hard Slice: DAEC-selective vs SetR-faithful

Date: 2026-05-07

This report reuses the existing hard-slice protocol from `reports/reviewer_baseline_ci_20260506/hard_slice_gold_doc_count.md`. The primary hard slice is `gold_doc_count>=3`; in the aligned 2Wiki full1000 split this is exactly the 4-document subset (`N=235`), since the remaining 765 examples have 2 gold documents and there are no 3-document cases.

Bootstrap is query-paired percentile bootstrap with 10,000 resamples. Delta is left method minus right method.

## Main Hard-Slice Result

| Slice | N | Metric | DAEC-selective | SetR-faithful | Delta | 95% CI | Excludes 0 |
|---|---:|---|---:|---:|---:|---:|---|
| 2Wiki 4-doc | 235 | EM | 0.9149 | 0.7660 | +0.1489 | [0.0936, 0.2043] | yes |
| 2Wiki 4-doc | 235 | F1 | 0.9149 | 0.7712 | +0.1437 | [0.0879, 0.2010] | yes |
| 2Wiki 4-doc | 235 | R5_TITLE | 0.9298 | 0.7830 | +0.1468 | [0.1181, 0.1755] | yes |

## Context Against SetR + Fill@5

| Slice | N | Metric | DAEC-selective | SetR + Fill@5 | Delta | 95% CI | Excludes 0 |
|---|---:|---|---:|---:|---:|---:|---|
| 2Wiki 4-doc | 235 | EM | 0.9149 | 0.8681 | +0.0468 | [0.0043, 0.0894] | yes |
| 2Wiki 4-doc | 235 | F1 | 0.9149 | 0.8695 | +0.0454 | [0.0028, 0.0894] | yes |
| 2Wiki 4-doc | 235 | R5_TITLE | 0.9298 | 0.9191 | +0.0106 | [-0.0117, 0.0319] | no |

| Slice | N | Metric | SetR-faithful | SetR + Fill@5 | Delta | 95% CI | Excludes 0 |
|---|---:|---|---:|---:|---:|---:|---|
| 2Wiki 4-doc | 235 | EM | 0.7660 | 0.8681 | -0.1021 | [-0.1532, -0.0511] | yes |
| 2Wiki 4-doc | 235 | F1 | 0.7712 | 0.8695 | -0.0983 | [-0.1494, -0.0478] | yes |
| 2Wiki 4-doc | 235 | R5_TITLE | 0.7830 | 0.9191 | -0.1362 | [-0.1596, -0.1138] | yes |

## Selection-Size Sanity

| Slice | N | Avg SetR-faithful reader passages | Selected-count distribution |
|---|---:|---:|---|
| all 2Wiki | 1000 | 2.577 | {1: 35, 2: 624, 3: 164, 4: 136, 5: 23, 6: 10, 8: 2, 9: 1, 10: 2, 12: 2, 14: 1} |
| gold_doc_count=2 | 765 | 2.325 | {1: 35, 2: 572, 3: 94, 4: 38, 5: 14, 6: 6, 8: 2, 10: 2, 12: 2} |
| 2Wiki 4-doc | 235 | 3.396 | {2: 52, 3: 70, 4: 98, 5: 9, 6: 4, 9: 1, 14: 1} |

On the 4-doc subset, SetR-faithful selects fewer than 4 passages for 122/235 queries (51.9%). This is consistent with the large support gap against DAEC-selective under the faithful adaptive-context setting. Rank-order fill-to-5 substantially repairs SetR on this slice, but DAEC-selective still remains significantly higher on answer EM/F1.

## Paper-Facing Takeaway

DAEC-selective's advantage over SetR-faithful is concentrated where the task actually requires deep composition. On the 2Wiki 4-doc subset, DAEC-selective improves answer F1 by +0.1437 with a 95% CI [0.0879, 0.2010]. Even under the stronger budget-matched `SetR + Fill@5` ablation, DAEC-selective remains significantly higher on answer F1 by +0.0454.
