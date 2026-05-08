# DAEC-Selective vs SetR Cross-Pool CI Deposition - 2026-05-08

## Purpose

This note consolidates the cross-pool paired-CI comparison between DAEC-selective and the existing SetR-style `k20_doc768` full1000 runs.

It is a reviewer-defense result deposition. The goal is to preserve:

- the Dense/HippoRAG/PropRAG x 2Wiki/HotpotQA/MuSiQue paired-CI result;
- the exact claim boundary against SetR-style selection;
- the evidence that answer F1 is tie-range rather than DAEC dominance;
- canonical report paths for paper writing.

## Executive Takeaway

DAEC-selective and SetR-style k20 are in answer-F1 tie range across all 9 pool-dataset cells.

This is not a DAEC dominance result:

- DAEC-selective is mean-higher than SetR-style in `4/9` cells.
- Significant positive F1 wins over SetR-style occur in `0/9` cells.
- Significant negative F1 losses against SetR-style occur in `0/9` cells.
- DAEC-selective has higher title-multiset support R@5 than SetR-style in `2/9` cells, with significant positive R@5 differences in `2/9` cells.

The paper-safe message is:

```text
Across three retrieval pools, DAEC-selective and SetR-style k20 are in
answer-F1 tie range in all 9 pool-dataset cells. DAEC-selective is
mean-higher on 2Wiki, while SetR-style is mean-higher on most HotpotQA
and MuSiQue cells.
```

## Canonical Artifacts

Primary report:

- `reports/daec_setr_cross_pool_ci_20260508/summary.md`
- `reports/daec_setr_cross_pool_ci_20260508/method_summary.csv`
- `reports/daec_setr_cross_pool_ci_20260508/paired_ci.csv`
- `reports/daec_setr_cross_pool_ci_20260508/summary.json`

Implementation:

- `scripts/analyze_daec_setr_cross_pool_ci.py`
- `tests/test_daec_setr_cross_pool_ci.py`

Inputs:

- DAEC-selective cross-pool runs:
  - `run_logs/daec_selective_titleuniq_dense_hipporag_full1000_20260506/`
  - `run_logs/daec_selective_titleuniq_proprag_full1000_20260506/`
- Ungated DAEC cross-pool runs:
  - `run_logs/daec_llm_wiki_title_dense_hipporag_full1000_20260503/`
  - `run_logs/daec_llm_wiki_title_proprag_full1000_20260503/`
- SetR-style cross-pool eval reports:
  - `reports/setr_full1000_20260503/*_setr_k20_doc768.eval.json`

Protocol:

- Existing outputs only; no new LLM calls.
- Fixed Dense/HippoRAG/PropRAG pool100.
- Qwen3-8B `/no_think` reader substrate.
- `qa_top_k=5`.
- `qa_doc_max_chars=2048`.
- SetR-style row: `k20_doc768`.
- Query-paired percentile bootstrap with `10,000` resamples.
- `R5_TITLE` is recomputed uniformly from final reader top-5 titles using title-multiset support recall.

## Main F1 Summary

| Pool | Dataset | Top-5 F1 | SetR-style F1 | DAEC-selective F1 | DAEC-selective - SetR F1 | 95% CI | SetR - Top-5 F1 | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 2Wiki | 0.4984 | 0.5655 | 0.5798 | +0.0144 | [-0.0105, +0.0387] | +0.0671 | [+0.0484, +0.0864] |
| Dense | HotpotQA | 0.7106 | 0.7470 | 0.7483 | +0.0013 | [-0.0139, +0.0164] | +0.0365 | [+0.0212, +0.0512] |
| Dense | MuSiQue | 0.3896 | 0.4108 | 0.4002 | -0.0106 | [-0.0332, +0.0125] | +0.0212 | [-0.0006, +0.0431] |
| HippoRAG | 2Wiki | 0.5850 | 0.6409 | 0.6431 | +0.0023 | [-0.0206, +0.0251] | +0.0559 | [+0.0384, +0.0739] |
| HippoRAG | HotpotQA | 0.7010 | 0.7491 | 0.7403 | -0.0088 | [-0.0234, +0.0056] | +0.0481 | [+0.0327, +0.0638] |
| HippoRAG | MuSiQue | 0.3947 | 0.4391 | 0.4206 | -0.0185 | [-0.0411, +0.0039] | +0.0443 | [+0.0220, +0.0671] |
| PropRAG | 2Wiki | 0.6457 | 0.6936 | 0.7118 | +0.0182 | [-0.0013, +0.0384] | +0.0479 | [+0.0318, +0.0641] |
| PropRAG | HotpotQA | 0.7227 | 0.7552 | 0.7473 | -0.0079 | [-0.0226, +0.0067] | +0.0325 | [+0.0195, +0.0461] |
| PropRAG | MuSiQue | 0.4266 | 0.4761 | 0.4548 | -0.0212 | [-0.0442, +0.0015] | +0.0494 | [+0.0276, +0.0716] |

## Interpretation

This result narrows the cross-pool claim:

- DAEC-selective remains a strong Top-5 repair method across pools.
- SetR-style is also a strong Top-5 repair method across pools.
- Against SetR-style specifically, DAEC-selective should be framed as competitive/tie-range on answer F1 rather than consistently superior.
- The clearest DAEC-specific advantage remains the 2Wiki deep/hard-slice result, especially against RankGPT-style sliding and SetR-faithful.
- HotpotQA and MuSiQue should be written as mixed/competitive settings, with MuSiQue explicitly discussed as a budget-conditioned limitation.

## Claim Boundary

Allowed:

```text
DAEC-selective and SetR-style k20 are statistically tied on answer F1 across
all three retrieval pools and all three datasets under the controlled local
Qwen3-8B substrate.
```

Allowed:

```text
Both DAEC-selective and SetR-style substantially improve over same-pool Top-5
retriever baselines; their relative answer-F1 differences are dataset-dependent.
```

Not allowed:

```text
DAEC-selective uniformly outperforms SetR-style across pools and datasets.
```

Not allowed:

```text
DAEC-selective consistently improves support R@5 over SetR-style.
```
