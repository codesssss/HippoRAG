# Full-Scale Width-Matched Control: HotpotQA + 2Wiki top-5

| Dataset | Run | EM | F1 | R@5 | R@20 | R@100 | num_queries | ΔEM vs baseline top-5 | ΔEM vs baseline top-10+CE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa | baseline_top5 | 0.5640 | 0.6795 | 0.8840 | 0.9735 | 0.9930 | 1000 | — | — |
| hotpotqa | baseline_top10_plus_ce | 0.5940 | 0.7185 | 0.9390 | 0.9735 | 0.9930 | — | +0.0300 | +0.0000 |
| hotpotqa | random3_deep_plus_ce | 0.5880 | 0.7136 | 0.9375 | 0.9740 | 0.9930 | — | +0.0240 | -0.0060 |
| hotpotqa | bridge_append_plus_ce | 0.5940 | 0.7186 | 0.9405 | 0.9740 | 0.9930 | — | +0.0300 | +0.0000 |
| 2wikimultihopqa | baseline_top5 | 0.4300 | 0.4843 | 0.7863 | 0.8540 | 0.9060 | 1000 | — | — |
| 2wikimultihopqa | baseline_top10_plus_ce | 0.4240 | 0.4796 | 0.7755 | 0.8540 | 0.9060 | — | -0.0060 | +0.0000 |
| 2wikimultihopqa | random3_deep_plus_ce | 0.4170 | 0.4761 | 0.7685 | 0.8545 | 0.9060 | — | -0.0130 | -0.0070 |
| 2wikimultihopqa | bridge_append_plus_ce | 0.4300 | 0.4901 | 0.7860 | 0.8622 | 0.9060 | — | +0.0000 | +0.0060 |
