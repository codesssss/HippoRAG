# RankGPT-Style Sliding Reader Paired CI

Query-paired percentile bootstrap over `10000` resamples. Delta is left method minus right method.

| Dataset | Comparison | Metric | Left | Right | Delta | 95% CI | P(delta > 0) | Excludes 0 |
|---|---|---|---:|---:|---:|---:|---:|---|
| 2Wiki | DAEC-selective - RankGPT-style sliding | EM | 0.6420 | 0.6020 | +0.0400 | [+0.0170, +0.0630] | 0.9995 | yes |
| 2Wiki | DAEC-selective - RankGPT-style sliding | F1 | 0.7118 | 0.6659 | +0.0460 | [+0.0244, +0.0674] | 1.0000 | yes |
| 2Wiki | SetR-faithful - RankGPT-style sliding | EM | 0.6030 | 0.6020 | +0.0010 | [-0.0220, +0.0240] | 0.5145 | no |
| 2Wiki | SetR-faithful - RankGPT-style sliding | F1 | 0.6746 | 0.6659 | +0.0087 | [-0.0139, +0.0316] | 0.7732 | no |
| 2Wiki | DAEC-selective - SetR-faithful | EM | 0.6420 | 0.6030 | +0.0390 | [+0.0160, +0.0630] | 0.9996 | yes |
| 2Wiki | DAEC-selective - SetR-faithful | F1 | 0.7118 | 0.6746 | +0.0372 | [+0.0149, +0.0598] | 0.9996 | yes |
| HotpotQA | DAEC-selective - RankGPT-style sliding | EM | 0.6200 | 0.5660 | +0.0540 | [+0.0330, +0.0750] | 1.0000 | yes |
| HotpotQA | DAEC-selective - RankGPT-style sliding | F1 | 0.7473 | 0.6845 | +0.0628 | [+0.0418, +0.0838] | 1.0000 | yes |
| HotpotQA | SetR-faithful - RankGPT-style sliding | EM | 0.6250 | 0.5660 | +0.0590 | [+0.0370, +0.0810] | 1.0000 | yes |
| HotpotQA | SetR-faithful - RankGPT-style sliding | F1 | 0.7435 | 0.6845 | +0.0590 | [+0.0380, +0.0802] | 1.0000 | yes |
| HotpotQA | DAEC-selective - SetR-faithful | EM | 0.6200 | 0.6250 | -0.0050 | [-0.0250, +0.0150] | 0.2920 | no |
| HotpotQA | DAEC-selective - SetR-faithful | F1 | 0.7473 | 0.7435 | +0.0038 | [-0.0149, +0.0221] | 0.6522 | no |
| MuSiQue | DAEC-selective - RankGPT-style sliding | EM | 0.3530 | 0.3190 | +0.0340 | [+0.0050, +0.0620] | 0.9895 | yes |
| MuSiQue | DAEC-selective - RankGPT-style sliding | F1 | 0.4548 | 0.4093 | +0.0455 | [+0.0185, +0.0733] | 0.9994 | yes |
| MuSiQue | SetR-faithful - RankGPT-style sliding | EM | 0.3440 | 0.3190 | +0.0250 | [+0.0000, +0.0510] | 0.9699 | no |
| MuSiQue | SetR-faithful - RankGPT-style sliding | F1 | 0.4467 | 0.4093 | +0.0373 | [+0.0117, +0.0625] | 0.9983 | yes |
| MuSiQue | DAEC-selective - SetR-faithful | EM | 0.3530 | 0.3440 | +0.0090 | [-0.0180, +0.0360] | 0.7286 | no |
| MuSiQue | DAEC-selective - SetR-faithful | F1 | 0.4548 | 0.4467 | +0.0082 | [-0.0180, +0.0346] | 0.7253 | no |
