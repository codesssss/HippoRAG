# PCEC Residual Budget Sweep Retrieval-Only

Date: 2026-05-19

This sweep reuses the existing ETv4 fact-witnessed STO top-100 pools,
frozen requirement reports, and frozen binding caches from the Table 1
EvidenceFlow run. It changes only the PCEC preservation budget:

- `K=5`, `pool_k=100`
- `r = K - m`, where `m=prefix_budget_m`
- no GPT-4o-mini reader calls
- no ETv4 pool regeneration
- binding model string: `qwen3-32b-judge`
- binding cache misses: `0` for all runs

New retrieval-only outputs:

- `run_logs/pcec_residual_budget_sweep_retrieval_only_20260519/evals/*prefix3_residual2*`
- `run_logs/pcec_residual_budget_sweep_retrieval_only_20260519/evals/*prefix0_residual5*`

Existing reused outputs:

- `r=0`: `run_logs/etv4_ablation_relation_coverage_all32_gpt4omini_full1000_20260516/no_coverage/evals/*prefix5_residual0*`
- `r=1`: `run_logs/all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514/pcec/etv4/evals/*prefix4_residual1*`

## Aggregate Retrieval Metrics

| Variant | Hotpot R@5 | Hotpot All@5 | 2Wiki R@5 | 2Wiki All@5 | MuSiQue R@5 | MuSiQue All@5 | Avg R@5 | Avg All@5 | Changed avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r=0` (`m=5`, no admission) | 95.05 | 90.50 | 93.50 | 82.60 | 74.17 | 46.60 | 87.57 | 73.23 | 0.0 |
| `r=1` (`m=4`, current) | 96.55 | 93.40 | 96.30 | 89.70 | 76.42 | 51.00 | 89.76 | 78.03 | 342.0 |
| `r=2` (`m=3`) | 96.35 | 93.10 | 95.43 | 86.60 | 76.50 | 50.90 | 89.43 | 76.87 | 356.3 |
| `r=5` (`m=0`, no prefix) | 94.50 | 89.40 | 91.80 | 79.00 | 72.92 | 45.90 | 86.41 | 71.43 | 356.7 |

## Change Accounting

| Variant | Dataset | Changed | All@5 Improve | All@5 Worsen | Gold-admitted queries | Gold-displaced queries |
|---|---|---:|---:|---:|---:|---:|
| `r=1` (`m=4`, current) | 2Wiki | 334 | 78 | 9 | 83 | 17 |
| `r=1` (`m=4`, current) | HotpotQA | 149 | 28 | 0 | 30 | 1 |
| `r=1` (`m=4`, current) | MuSiQue | 543 | 60 | 22 | 106 | 52 |
| `r=2` (`m=3`) | 2Wiki | 346 | 94 | 56 | 101 | 66 |
| `r=2` (`m=3`) | HotpotQA | 154 | 30 | 4 | 32 | 6 |
| `r=2` (`m=3`) | MuSiQue | 569 | 70 | 29 | 136 | 77 |
| `r=5` (`m=0`, no prefix) | 2Wiki | 346 | 80 | 121 | 101 | 157 |
| `r=5` (`m=0`, no prefix) | HotpotQA | 154 | 30 | 41 | 32 | 43 |
| `r=5` (`m=0`, no prefix) | MuSiQue | 570 | 73 | 84 | 143 | 172 |

## Interpretation

The current setting, `r=1` (`m=4`), is the best retrieval setting in this
sweep by both average R@5 and average All@5. Opening a second residual
slot (`r=2`) slightly reduces average R@5 and All@5. Removing prefix
preservation entirely (`r=5`, `m=0`) is clearly worse than both the
current setting and the no-admission baseline.

The change accounting explains why. The no-prefix setting admits more
gold passages in some cases, but it also displaces many more gold
passages that were already in the ETv4 top-5. On 2Wiki, no-prefix causes
121 All@5 regressions versus only 9 for the current one-slot setting; on
MuSiQue, no-prefix causes 84 regressions versus 22 for the current
setting.

This supports the paper-facing interpretation that PCEC should be a
conservative residual repair step rather than an unrestricted reranker.
The prefix constraint is not a scoring trick; it prevents the evidence
utility from overwriting high-precision ETv4 prefix evidence.
