# DAEC-LLM Guarded Binding Audit Limit100

Run directory: `run_logs/daec_llm_binding_guarded_limit100_20260503`

Code commit:
- `c2c8a04 Add guarded DAEC LLM title matching`

Protocol:
- selector: `daec_noisyor_llm`
- match mode: `substring_guarded`
- datasets: `2wikimultihopqa`, `hotpotqa`, `musique`
- pools: dense, HippoRAG, PropRAG
- limit: 100
- pool_k: 100
- qa_top_k: 5
- qa_doc_max_chars: 2048
- reader / binding model: Qwen3-8B via 8041/8042/8043
- no-think: `HIPPORAG_RERANK_FORCE_NO_THINK=1`

Launcher:
- `run_logs/launch_daec_llm_binding_guarded_limit100_20260503.sh`

All 9 guarded runs completed with `rc=0`.

## Guard Definition

`substring_guarded` keeps exact title matches unchanged. For substring title matches, it adds:

- numeric-only entity rejection;
- token-boundary containment through normalized token subsequences;
- single-token blocklist for country names and one observed high-risk common name (`muhammad`);
- reverse containment only when the matched title phrase has at least two tokens.

This keeps disambiguated aliases such as `Ian Barry -> Ian Barry (director)` while rejecting observed risky matches such as `France -> Rudolph of France`, `Russia -> Grand Duke Vladimir Alexandrovich of Russia`, `Iran -> Gohar, Iran`, `1926 -> Camille (1926 feature film)`, and `Muhammad -> Abdul-Aziz bin Muhammad`.

## Three-Way Metrics

| Pool | Dataset | Match | Base EM | Sel EM | Delta EM | Base F1 | Sel F1 | Delta F1 | Base R@5 | Sel R@5 | Delta R@5 | EM G/R/S | Changed | exact/sub/unmatched |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2Wiki | exact | 0.410 | 0.460 | +0.050 | 0.4362 | 0.5135 | +0.0773 | 0.695 | 0.782 | +0.088 | 10/5/85 | 92 | 81/0/230 |
| dense | 2Wiki | guarded | 0.410 | 0.470 | +0.060 | 0.4362 | 0.5235 | +0.0873 | 0.695 | 0.777 | +0.083 | 10/4/86 | 89 | 82/19/215 |
| dense | 2Wiki | substring | 0.410 | 0.470 | +0.060 | 0.4362 | 0.5235 | +0.0873 | 0.695 | 0.777 | +0.083 | 10/4/86 | 88 | 82/23/209 |
| dense | HotpotQA | exact | 0.580 | 0.570 | -0.010 | 0.6881 | 0.6798 | -0.0083 | 0.915 | 0.915 | +0.000 | 1/2/97 | 66 | 110/0/608 |
| dense | HotpotQA | guarded | 0.580 | 0.580 | +0.000 | 0.6881 | 0.6948 | +0.0067 | 0.915 | 0.935 | +0.020 | 2/2/96 | 68 | 110/64/525 |
| dense | HotpotQA | substring | 0.580 | 0.580 | +0.000 | 0.6881 | 0.6948 | +0.0067 | 0.915 | 0.935 | +0.020 | 2/2/96 | 69 | 111/79/508 |
| dense | MuSiQue | exact | 0.340 | 0.300 | -0.040 | 0.4028 | 0.3720 | -0.0308 | 0.630 | 0.670 | +0.040 | 6/10/84 | 88 | 120/0/954 |
| dense | MuSiQue | guarded | 0.340 | 0.340 | +0.000 | 0.4028 | 0.4183 | +0.0155 | 0.630 | 0.694 | +0.064 | 9/9/82 | 88 | 120/110/740 |
| dense | MuSiQue | substring | 0.340 | 0.340 | +0.000 | 0.4028 | 0.4200 | +0.0172 | 0.630 | 0.681 | +0.051 | 9/9/82 | 88 | 120/134/693 |
| hipporag | 2Wiki | exact | 0.510 | 0.530 | +0.020 | 0.5470 | 0.5890 | +0.0420 | 0.815 | 0.897 | +0.083 | 10/8/82 | 99 | 104/0/207 |
| hipporag | 2Wiki | guarded | 0.510 | 0.540 | +0.030 | 0.5470 | 0.6019 | +0.0549 | 0.815 | 0.905 | +0.090 | 10/7/83 | 97 | 104/23/183 |
| hipporag | 2Wiki | substring | 0.510 | 0.540 | +0.030 | 0.5470 | 0.6019 | +0.0549 | 0.815 | 0.905 | +0.090 | 10/7/83 | 96 | 104/27/177 |
| hipporag | HotpotQA | exact | 0.560 | 0.590 | +0.030 | 0.6712 | 0.6972 | +0.0260 | 0.925 | 0.940 | +0.015 | 6/3/91 | 81 | 111/0/606 |
| hipporag | HotpotQA | guarded | 0.560 | 0.580 | +0.020 | 0.6712 | 0.7015 | +0.0303 | 0.925 | 0.950 | +0.025 | 6/4/90 | 81 | 111/65/523 |
| hipporag | HotpotQA | substring | 0.560 | 0.580 | +0.020 | 0.6712 | 0.7015 | +0.0303 | 0.925 | 0.950 | +0.025 | 6/4/90 | 82 | 111/81/503 |
| hipporag | MuSiQue | exact | 0.300 | 0.320 | +0.020 | 0.3687 | 0.4170 | +0.0483 | 0.642 | 0.693 | +0.052 | 9/7/84 | 93 | 121/0/965 |
| hipporag | MuSiQue | guarded | 0.300 | 0.340 | +0.040 | 0.3687 | 0.4309 | +0.0622 | 0.642 | 0.696 | +0.054 | 10/6/84 | 92 | 121/109/752 |
| hipporag | MuSiQue | substring | 0.300 | 0.350 | +0.050 | 0.3687 | 0.4409 | +0.0722 | 0.642 | 0.692 | +0.050 | 11/6/83 | 92 | 121/144/706 |
| proprag | 2Wiki | exact | 0.580 | 0.580 | +0.000 | 0.6318 | 0.6259 | -0.0059 | 0.935 | 0.932 | -0.003 | 6/6/88 | 92 | 120/0/190 |
| proprag | 2Wiki | guarded | 0.580 | 0.600 | +0.020 | 0.6318 | 0.6537 | +0.0219 | 0.935 | 0.940 | +0.005 | 6/4/90 | 89 | 120/22/168 |
| proprag | 2Wiki | substring | 0.580 | 0.600 | +0.020 | 0.6318 | 0.6537 | +0.0219 | 0.935 | 0.940 | +0.005 | 6/4/90 | 88 | 120/28/156 |
| proprag | HotpotQA | exact | 0.570 | 0.580 | +0.010 | 0.6912 | 0.6987 | +0.0075 | 0.930 | 0.935 | +0.005 | 4/3/93 | 73 | 110/0/617 |
| proprag | HotpotQA | guarded | 0.570 | 0.600 | +0.030 | 0.6912 | 0.7210 | +0.0298 | 0.930 | 0.950 | +0.020 | 6/3/91 | 74 | 110/75/521 |
| proprag | HotpotQA | substring | 0.570 | 0.600 | +0.030 | 0.6912 | 0.7210 | +0.0298 | 0.930 | 0.950 | +0.020 | 6/3/91 | 75 | 110/89/504 |
| proprag | MuSiQue | exact | 0.380 | 0.390 | +0.010 | 0.4374 | 0.4689 | +0.0315 | 0.677 | 0.697 | +0.019 | 7/6/87 | 85 | 127/0/949 |
| proprag | MuSiQue | guarded | 0.380 | 0.380 | +0.000 | 0.4374 | 0.4595 | +0.0221 | 0.677 | 0.713 | +0.035 | 8/8/84 | 85 | 127/104/743 |
| proprag | MuSiQue | substring | 0.380 | 0.370 | -0.010 | 0.4374 | 0.4480 | +0.0106 | 0.677 | 0.710 | +0.032 | 8/9/83 | 85 | 126/143/693 |

## Aggregate Position

| Match | Avg Delta EM | Avg Delta F1 | Exact Matches | Substring Matches | Unmatched | Failures | Empty Entity Responses |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact | +0.0100 | +0.0208 | 1004 | 0 | 5326 | 0 | 3782 |
| substring_guarded | +0.0222 | +0.0367 | 1005 | 591 | 4370 | 0 | 3782 |
| substring | +0.0222 | +0.0368 | 1005 | 748 | 4149 | 0 | 3782 |

## Interpretation

1. `substring_guarded` removes 157 of 748 raw substring matches, while preserving almost all average F1 gain from full substring.
2. Compared with exact-only, guarded is better on 8 of 9 pool/dataset settings by F1; the exception is PropRAG/MuSiQue, where exact F1 is 0.4689 and guarded F1 is 0.4595.
3. Compared with full substring, guarded is identical on 5 of 9 settings, slightly lower on dense/MuSiQue and HippoRAG/MuSiQue, and better on PropRAG/MuSiQue.
4. The guard removes the observed numeric/country/common-name failures. A scan of retained guarded substring matches found no retained `France`, `Russia`, `Iran`, `1926`, or `Muhammad` substring matches.
5. Some residual one-token proper-name ambiguity remains, for example `Lichtenberg -> Johann Reinhard I, Count of Hanau-Lichtenberg` and `John Henry -> John Henry Kreitler`. A stricter v2 guard could require single-token substring entities to match the title prefix, but v1 already gives a strong cleanliness/performance tradeoff.

## Recommendation

Use `substring_guarded` as the default DAEC-LLM binding mode for the next full1000 run. It keeps the empirical behavior of substring while giving a defensible precision filter:

- exact-only is too conservative;
- raw substring is too hard to defend;
- guarded substring preserves nearly all limit100 gains and removes the most obvious false bindings.
