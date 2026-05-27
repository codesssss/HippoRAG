# DAEC-L1-Safe-v2 Aligned Limit100 Summary

Date: 2026-05-01

## Protocol

- Candidate pool: `run_logs/proprag_pool_exports_full1000_20260424/{dataset}_pool100.json`
- Selector: `daec_noisyor_safe`
- Reader: Qwen3-8B endpoint per dataset, no-think forced by `HIPPORAG_RERANK_FORCE_NO_THINK=1`
- `qa_top_k=5`
- `qa_doc_max_chars=2048`
- `setwise_pool_k=100`
- `dtc_decomposition_mode=llm`
- `dtc_binding_max_candidates=5`

Safe-v2 parameters:

| Parameter | Value |
|---|---:|
| `daec_safe_min_objective_gain` | 0.02 |
| `daec_safe_min_swap_gain` | 0.01 |
| `daec_safe_max_swaps` | 1 |
| `daec_safe_preserve_top_m` | 1 |
| `daec_safe_retriever_margin_threshold` | 0.80 |
| `daec_safe_retriever_rank_penalty` | 0.20 |

Code change: Safe-v2 adds a retriever rank-loss penalty to each candidate swap:

```text
adjusted_gain = daec_objective_gain - rank_penalty * max(0, rank_score[out] - rank_score[in])
```

The parameter defaults to `0.0`, so prior DAEC/Safe behavior is preserved unless explicitly enabled.

## Results

| Dataset | R@5 | R@20 | EM | F1 | EM Delta vs Prop | F1 Delta vs Prop | Changed | Decisions | Query gains/reg/same |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 2wikimultihopqa | 0.9375 | 0.9650 | 0.5900 | 0.6420 | +0.0100 | +0.0102 | 12 | fallback_low_rebuild_gain:33, fallback_retriever_margin:55, minimal_edit_applied:12 | 1/0/99 |
| hotpotqa | 0.9350 | 0.9950 | 0.5700 | 0.6912 | +0.0000 | +0.0000 | 12 | fallback_low_rebuild_gain:55, fallback_no_eligible_swap:1, fallback_retriever_margin:32, minimal_edit_applied:12 | 0/0/100 |
| musique | 0.6700 | 0.8900 | 0.3400 | 0.4030 | -0.0400 | -0.0344 | 42 | fallback_low_rebuild_gain:24, fallback_retriever_margin:33, minimal_edit_applied:43 | 2/6/92 |

## Interpretation

Safe-v2 fixes the 2Wiki failure mode:
- Safe-v1 `s1_p1_g002`: 2Wiki EM/F1 = 0.5500 / 0.5942
- Safe-v2: 2Wiki EM/F1 = 0.5900 / 0.6420
- Prop baseline: 2Wiki EM/F1 = 0.5800 / 0.6318
- Original DAEC: 2Wiki EM/F1 = 0.5800 / 0.6435

HotpotQA becomes neutral:
- Safe-v2 exactly matches Prop EM/F1.
- It no longer shows the Safe-v1 regression, but it also loses DAEC-L1's positive HotpotQA gain.

MuSiQue fails:
- Safe-v2 EM/F1 = 0.3400 / 0.4030, below Prop and below all useful DAEC variants.
- The unified retriever-margin/rank-penalty risk model is too blunt for MuSiQue long-chain questions.

## Judgment

Safe-v2 is a useful diagnostic, not a full method yet.

It shows that Claude's point about missing retriever-aware risk control was valid: adding retriever fallback and rank-loss penalty repairs 2Wiki and prevents Hotpot regressions. However, the same global admission policy damages MuSiQue. This means the next version cannot be another global threshold tweak. It needs dataset/query-structure-aware admission, likely keyed by hop count, binding confidence, or sentence/relation-level evidence support.

Do not run Safe-v2 full1000 from this configuration. The correct paper-level use is an ablation/diagnostic showing that risk-aware projection matters, but that a single global risk gate is insufficient for long-chain MuSiQue.
