# DAEC-Selective vs SetR-Style Cross-Pool Paired CI

Date: 2026-05-08

Purpose: close the reviewer-facing cross-pool baseline-CI gap by comparing DAEC-selective against the existing SetR-style `k20_doc768` full1000 runs on Dense, HippoRAG, and PropRAG pools.

This report reuses existing outputs only; no new LLM calls are made. EM/F1 are answer metrics. `R5_TITLE` is recomputed uniformly as title-multiset support recall from final reader top-5 titles. Delta is left method minus right method; CIs use query-paired percentile bootstrap with 10,000 resamples.

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

## Detailed F1 CI

| Pool | Dataset | Comparison | Left | Right | Delta | 95% CI | P(delta > 0) | Excludes 0 |
|---|---|---|---:|---:|---:|---:|---:|---|
| Dense | 2Wiki | DAEC-selective - SetR-style k20 | 0.5798 | 0.5655 | +0.0144 | [-0.0105, +0.0387] | 0.8731 | no |
| Dense | 2Wiki | DAEC - SetR-style k20 | 0.5798 | 0.5655 | +0.0144 | [-0.0102, +0.0395] | 0.8764 | no |
| Dense | 2Wiki | SetR-style k20 - Top-5 | 0.5655 | 0.4984 | +0.0671 | [+0.0484, +0.0864] | 1.0000 | yes |
| Dense | 2Wiki | DAEC-selective - Top-5 | 0.5798 | 0.4984 | +0.0814 | [+0.0560, +0.1075] | 1.0000 | yes |
| Dense | HotpotQA | DAEC-selective - SetR-style k20 | 0.7483 | 0.7470 | +0.0013 | [-0.0139, +0.0164] | 0.5677 | no |
| Dense | HotpotQA | DAEC - SetR-style k20 | 0.7483 | 0.7470 | +0.0013 | [-0.0139, +0.0165] | 0.5622 | no |
| Dense | HotpotQA | SetR-style k20 - Top-5 | 0.7470 | 0.7106 | +0.0365 | [+0.0212, +0.0512] | 1.0000 | yes |
| Dense | HotpotQA | DAEC-selective - Top-5 | 0.7483 | 0.7106 | +0.0378 | [+0.0236, +0.0523] | 1.0000 | yes |
| Dense | MuSiQue | DAEC-selective - SetR-style k20 | 0.4002 | 0.4108 | -0.0106 | [-0.0332, +0.0125] | 0.1880 | no |
| Dense | MuSiQue | DAEC - SetR-style k20 | 0.3918 | 0.4108 | -0.0190 | [-0.0432, +0.0046] | 0.0558 | no |
| Dense | MuSiQue | SetR-style k20 - Top-5 | 0.4108 | 0.3896 | +0.0212 | [-0.0006, +0.0431] | 0.9710 | no |
| Dense | MuSiQue | DAEC-selective - Top-5 | 0.4002 | 0.3896 | +0.0106 | [-0.0095, +0.0305] | 0.8505 | no |
| HippoRAG | 2Wiki | DAEC-selective - SetR-style k20 | 0.6431 | 0.6409 | +0.0023 | [-0.0206, +0.0251] | 0.5903 | no |
| HippoRAG | 2Wiki | DAEC - SetR-style k20 | 0.6431 | 0.6409 | +0.0023 | [-0.0203, +0.0255] | 0.5812 | no |
| HippoRAG | 2Wiki | SetR-style k20 - Top-5 | 0.6409 | 0.5850 | +0.0559 | [+0.0384, +0.0739] | 1.0000 | yes |
| HippoRAG | 2Wiki | DAEC-selective - Top-5 | 0.6431 | 0.5850 | +0.0581 | [+0.0331, +0.0836] | 1.0000 | yes |
| HippoRAG | HotpotQA | DAEC-selective - SetR-style k20 | 0.7403 | 0.7491 | -0.0088 | [-0.0234, +0.0056] | 0.1188 | no |
| HippoRAG | HotpotQA | DAEC - SetR-style k20 | 0.7403 | 0.7491 | -0.0088 | [-0.0227, +0.0054] | 0.1092 | no |
| HippoRAG | HotpotQA | SetR-style k20 - Top-5 | 0.7491 | 0.7010 | +0.0481 | [+0.0327, +0.0638] | 1.0000 | yes |
| HippoRAG | HotpotQA | DAEC-selective - Top-5 | 0.7403 | 0.7010 | +0.0393 | [+0.0240, +0.0551] | 1.0000 | yes |
| HippoRAG | MuSiQue | DAEC-selective - SetR-style k20 | 0.4206 | 0.4391 | -0.0185 | [-0.0411, +0.0039] | 0.0539 | no |
| HippoRAG | MuSiQue | DAEC - SetR-style k20 | 0.4142 | 0.4391 | -0.0249 | [-0.0474, -0.0020] | 0.0161 | yes |
| HippoRAG | MuSiQue | SetR-style k20 - Top-5 | 0.4391 | 0.3947 | +0.0443 | [+0.0220, +0.0671] | 0.9999 | yes |
| HippoRAG | MuSiQue | DAEC-selective - Top-5 | 0.4206 | 0.3947 | +0.0259 | [+0.0054, +0.0463] | 0.9923 | yes |
| PropRAG | 2Wiki | DAEC-selective - SetR-style k20 | 0.7118 | 0.6936 | +0.0182 | [-0.0013, +0.0384] | 0.9666 | no |
| PropRAG | 2Wiki | DAEC - SetR-style k20 | 0.7118 | 0.6936 | +0.0182 | [-0.0017, +0.0383] | 0.9641 | no |
| PropRAG | 2Wiki | SetR-style k20 - Top-5 | 0.6936 | 0.6457 | +0.0479 | [+0.0318, +0.0641] | 1.0000 | yes |
| PropRAG | 2Wiki | DAEC-selective - Top-5 | 0.7118 | 0.6457 | +0.0661 | [+0.0439, +0.0883] | 1.0000 | yes |
| PropRAG | HotpotQA | DAEC-selective - SetR-style k20 | 0.7473 | 0.7552 | -0.0079 | [-0.0226, +0.0067] | 0.1355 | no |
| PropRAG | HotpotQA | DAEC - SetR-style k20 | 0.7473 | 0.7552 | -0.0079 | [-0.0224, +0.0065] | 0.1395 | no |
| PropRAG | HotpotQA | SetR-style k20 - Top-5 | 0.7552 | 0.7227 | +0.0325 | [+0.0195, +0.0461] | 1.0000 | yes |
| PropRAG | HotpotQA | DAEC-selective - Top-5 | 0.7473 | 0.7227 | +0.0246 | [+0.0109, +0.0386] | 0.9999 | yes |
| PropRAG | MuSiQue | DAEC-selective - SetR-style k20 | 0.4548 | 0.4761 | -0.0212 | [-0.0442, +0.0015] | 0.0352 | no |
| PropRAG | MuSiQue | DAEC - SetR-style k20 | 0.4359 | 0.4761 | -0.0402 | [-0.0628, -0.0175] | 0.0001 | yes |
| PropRAG | MuSiQue | SetR-style k20 - Top-5 | 0.4761 | 0.4266 | +0.0494 | [+0.0276, +0.0716] | 1.0000 | yes |
| PropRAG | MuSiQue | DAEC-selective - Top-5 | 0.4548 | 0.4266 | +0.0282 | [+0.0074, +0.0487] | 0.9967 | yes |

## Paper-Facing Takeaway

- DAEC-selective is higher than SetR-style k20 by mean F1 in `4/9` pool-dataset cells.
- Significant positive F1 wins over SetR-style occur in `0/9` cells; significant negative F1 losses occur in `0/9` cells.
- DAEC-selective has higher title-multiset support R@5 than SetR-style in `2/9` cells, with significant positive R@5 differences in `2/9` cells.
- The correct cross-pool message is competitive/tie-range answer F1 against SetR-style, plus stronger same-pool Top-5 improvements for both methods. It is not a dominance result.

## Claim Boundary

Allowed:

```text
Across three retrieval pools, DAEC-selective and SetR-style k20 are in
answer-F1 tie range in all 9 pool-dataset cells; DAEC-selective is
mean-higher on 2Wiki, while SetR-style is mean-higher on most HotpotQA
and MuSiQue cells.
```

Not allowed:

```text
DAEC-selective uniformly outperforms SetR-style across all pools and datasets.
```

Also not allowed:

```text
DAEC-selective consistently improves support R@5 over SetR-style.
```

## Files

- Summary CSV: `reports/daec_setr_cross_pool_ci_20260508/method_summary.csv`
- Paired CI CSV: `reports/daec_setr_cross_pool_ci_20260508/paired_ci.csv`
- JSON: `reports/daec_setr_cross_pool_ci_20260508/summary.json`
