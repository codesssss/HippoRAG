# DAEC Selective Binding Phase-1 Fresh Run

Protocol: PropRAG pool100, Qwen3-8B reader/decomposition, NV-Embed-v2, `wiki_title` LLM binding, query-level router frozen at `bind_conf_title_unique >= 0.88`.

## Main Results

| Dataset | Method | EM | F1 | R@5 |
|---|---|---:|---:|---:|
| 2Wiki | DAEC | 0.642 | 0.7118 | 0.9410 |
| 2Wiki | Nobind | 0.548 | 0.6120 | 0.8610 |
| 2Wiki | DAEC-selective | 0.642 | 0.7118 | 0.9410 |
| 2Wiki | SetR-style k20 | 0.624 | 0.6936 | 0.9423 |
| HotpotQA | DAEC | 0.620 | 0.7473 | 0.9605 |
| HotpotQA | Nobind | 0.616 | 0.7387 | 0.9545 |
| HotpotQA | DAEC-selective | 0.620 | 0.7473 | 0.9605 |
| HotpotQA | SetR-style k20 | 0.629 | 0.7552 | 0.9730 |
| MuSiQue | DAEC | 0.337 | 0.4359 | 0.7269 |
| MuSiQue | Nobind | 0.343 | 0.4458 | 0.7297 |
| MuSiQue | DAEC-selective | 0.353 | 0.4548 | 0.7469 |
| MuSiQue | SetR-style k20 | 0.377 | 0.4761 | 0.7547 |

## Deltas

| Dataset | dF1 vs DAEC | dF1 vs Nobind | dF1 vs SetR-style | dR@5 vs DAEC | dR@5 vs SetR-style |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 0.0000 | 0.0998 | 0.0182 | 0.0000 | -0.0013 |
| HotpotQA | 0.0000 | 0.0086 | -0.0079 | 0.0000 | -0.0125 |
| MuSiQue | 0.0189 | 0.0090 | -0.0213 | 0.0200 | -0.0078 |

## Fresh-vs-Phase0 Consistency

| Dataset | Phase0 F1 | Fresh F1 | Fresh-Phase0 F1 | Phase0 Null | Fresh Null | Expected Selection Agreement |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.7090 | 0.7118 | 0.0028 | 0.3120 | 0.3020 | 1000/1000 |
| HotpotQA | 0.7493 | 0.7473 | -0.0020 | 0.3720 | 0.3430 | 1000/1000 |
| MuSiQue | 0.4553 | 0.4548 | -0.0005 | 0.6380 | 0.6290 | 1000/1000 |

## Router Behavior

| Dataset | Bind | Abstain | Abstain Same As DAEC | Abstain Changed From DAEC |
|---|---:|---:|---:|---:|
| 2Wiki | 698 | 302 | 302 | 0 |
| HotpotQA | 657 | 343 | 343 | 0 |
| MuSiQue | 371 | 629 | 367 | 262 |

## Paired Bootstrap CI

Query-paired bootstrap over answer EM/F1, `10000` resamples. Delta is left method minus right method.

| Dataset | Comparison | Metric | Delta | 95% CI | P(delta > 0) | Excludes 0 |
|---|---|---:|---:|---:|---:|---:|
| 2Wiki | DAEC-selective - DAEC | EM | 0.0000 | [0.0000, 0.0000] | 0.000 | False |
| 2Wiki | DAEC-selective - DAEC | F1 | 0.0000 | [0.0000, 0.0000] | 0.000 | False |
| 2Wiki | DAEC-selective - Nobind | EM | 0.0940 | [0.0720, 0.1160] | 1.000 | True |
| 2Wiki | DAEC-selective - Nobind | F1 | 0.0998 | [0.0791, 0.1216] | 1.000 | True |
| 2Wiki | DAEC-selective - SetR-style k20 | EM | 0.0180 | [-0.0040, 0.0400] | 0.940 | False |
| 2Wiki | DAEC-selective - SetR-style k20 | F1 | 0.0182 | [-0.0019, 0.0379] | 0.964 | False |
| 2Wiki | DAEC - Nobind | EM | 0.0940 | [0.0720, 0.1160] | 1.000 | True |
| 2Wiki | DAEC - Nobind | F1 | 0.0998 | [0.0781, 0.1216] | 1.000 | True |
| HotpotQA | DAEC-selective - DAEC | EM | 0.0000 | [0.0000, 0.0000] | 0.000 | False |
| HotpotQA | DAEC-selective - DAEC | F1 | 0.0000 | [0.0000, 0.0000] | 0.000 | False |
| HotpotQA | DAEC-selective - Nobind | EM | 0.0040 | [-0.0100, 0.0180] | 0.688 | False |
| HotpotQA | DAEC-selective - Nobind | F1 | 0.0086 | [-0.0039, 0.0211] | 0.910 | False |
| HotpotQA | DAEC-selective - SetR-style k20 | EM | -0.0090 | [-0.0250, 0.0070] | 0.122 | False |
| HotpotQA | DAEC-selective - SetR-style k20 | F1 | -0.0079 | [-0.0225, 0.0066] | 0.142 | False |
| HotpotQA | DAEC - Nobind | EM | 0.0040 | [-0.0100, 0.0180] | 0.688 | False |
| HotpotQA | DAEC - Nobind | F1 | 0.0086 | [-0.0042, 0.0212] | 0.906 | False |
| MuSiQue | DAEC-selective - DAEC | EM | 0.0160 | [0.0060, 0.0270] | 0.999 | True |
| MuSiQue | DAEC-selective - DAEC | F1 | 0.0189 | [0.0076, 0.0309] | 1.000 | True |
| MuSiQue | DAEC-selective - Nobind | EM | 0.0100 | [-0.0060, 0.0260] | 0.877 | False |
| MuSiQue | DAEC-selective - Nobind | F1 | 0.0090 | [-0.0053, 0.0238] | 0.885 | False |
| MuSiQue | DAEC-selective - SetR-style k20 | EM | -0.0240 | [-0.0470, -0.0010] | 0.020 | True |
| MuSiQue | DAEC-selective - SetR-style k20 | F1 | -0.0212 | [-0.0443, 0.0018] | 0.035 | False |
| MuSiQue | DAEC - Nobind | EM | -0.0060 | [-0.0240, 0.0130] | 0.244 | False |
| MuSiQue | DAEC - Nobind | F1 | -0.0099 | [-0.0285, 0.0081] | 0.139 | False |

## Interpretation

- Fresh-vs-Phase0 F1 deviations are within the pre-set ±0.005 consistency gate for all datasets.
- On 2Wiki and HotpotQA, abstention almost always lands on the same evidence set as DAEC, so selective binding preserves the original DAEC score.
- On MuSiQue, 262 abstentions change the DAEC evidence set and recover most of the Phase-0 predicted improvement: F1 improves from 0.4359 to 0.4548 while support R@5 rises from 0.7269 to 0.7469.
- SetR-style remains stronger on HotpotQA and MuSiQue answer F1, but DAEC-selective keeps the 2Wiki win and improves MuSiQue support coverage over SetR-style.
- Bootstrap CIs are answer-metric only; support-recall CIs are not mixed with SetR-style because SetR's stored aggregate R@5 and transformed-pool title traces use different audit fields.
