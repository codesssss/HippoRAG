# DAEC-LLM Binding Audit Limit100

Run directory: `run_logs/daec_llm_binding_audit_limit100_20260503`

Protocol:
- selector: `daec_noisyor_llm`
- datasets: `2wikimultihopqa`, `hotpotqa`, `musique`
- pools: dense, HippoRAG, PropRAG
- match modes: `substring`, `exact`
- limit: 100
- pool_k: 100
- qa_top_k: 5
- qa_doc_max_chars: 2048
- reader / binding model: Qwen3-8B via 8041/8042/8043
- no-think: `HIPPORAG_RERANK_FORCE_NO_THINK=1`

## Run Status

All 18 selector runs completed with `rc=0`.

Launcher:
- `run_logs/launch_daec_llm_binding_audit_limit100_20260503.sh`

Outputs:
- 18 result JSON files: `*_daec_llm_limit100.json`
- 18 binding cache JSON files: `*.binding_cache.json`

## Main Metrics

| Pool | Match | Dataset | Base EM | Sel EM | Delta EM | Base F1 | Sel F1 | Delta F1 | Base R@5 | Sel R@5 | Delta R@5 | EM G/R/S | Changed |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | substring | 2Wiki | 0.410 | 0.470 | +0.060 | 0.4362 | 0.5235 | +0.0873 | 0.695 | 0.777 | +0.083 | 10/4/86 | 88 |
| dense | substring | HotpotQA | 0.580 | 0.580 | +0.000 | 0.6881 | 0.6948 | +0.0067 | 0.915 | 0.935 | +0.020 | 2/2/96 | 69 |
| dense | substring | MuSiQue | 0.340 | 0.340 | +0.000 | 0.4028 | 0.4200 | +0.0172 | 0.630 | 0.681 | +0.051 | 9/9/82 | 88 |
| dense | exact | 2Wiki | 0.410 | 0.460 | +0.050 | 0.4362 | 0.5135 | +0.0773 | 0.695 | 0.782 | +0.088 | 10/5/85 | 92 |
| dense | exact | HotpotQA | 0.580 | 0.570 | -0.010 | 0.6881 | 0.6798 | -0.0083 | 0.915 | 0.915 | +0.000 | 1/2/97 | 66 |
| dense | exact | MuSiQue | 0.340 | 0.300 | -0.040 | 0.4028 | 0.3720 | -0.0308 | 0.630 | 0.670 | +0.040 | 6/10/84 | 88 |
| hipporag | substring | 2Wiki | 0.510 | 0.540 | +0.030 | 0.5470 | 0.6019 | +0.0549 | 0.815 | 0.905 | +0.090 | 10/7/83 | 96 |
| hipporag | substring | HotpotQA | 0.560 | 0.580 | +0.020 | 0.6712 | 0.7015 | +0.0303 | 0.925 | 0.950 | +0.025 | 6/4/90 | 82 |
| hipporag | substring | MuSiQue | 0.300 | 0.350 | +0.050 | 0.3687 | 0.4409 | +0.0722 | 0.642 | 0.692 | +0.050 | 11/6/83 | 92 |
| hipporag | exact | 2Wiki | 0.510 | 0.530 | +0.020 | 0.5470 | 0.5890 | +0.0420 | 0.815 | 0.897 | +0.083 | 10/8/82 | 99 |
| hipporag | exact | HotpotQA | 0.560 | 0.590 | +0.030 | 0.6712 | 0.6972 | +0.0260 | 0.925 | 0.940 | +0.015 | 6/3/91 | 81 |
| hipporag | exact | MuSiQue | 0.300 | 0.320 | +0.020 | 0.3687 | 0.4170 | +0.0483 | 0.642 | 0.693 | +0.052 | 9/7/84 | 93 |
| proprag | substring | 2Wiki | 0.580 | 0.600 | +0.020 | 0.6318 | 0.6537 | +0.0219 | 0.935 | 0.940 | +0.005 | 6/4/90 | 88 |
| proprag | substring | HotpotQA | 0.570 | 0.600 | +0.030 | 0.6912 | 0.7210 | +0.0298 | 0.930 | 0.950 | +0.020 | 6/3/91 | 75 |
| proprag | substring | MuSiQue | 0.380 | 0.370 | -0.010 | 0.4374 | 0.4480 | +0.0106 | 0.677 | 0.710 | +0.032 | 8/9/83 | 85 |
| proprag | exact | 2Wiki | 0.580 | 0.580 | +0.000 | 0.6318 | 0.6259 | -0.0059 | 0.935 | 0.932 | -0.003 | 6/6/88 | 92 |
| proprag | exact | HotpotQA | 0.570 | 0.580 | +0.010 | 0.6912 | 0.6987 | +0.0075 | 0.930 | 0.935 | +0.005 | 4/3/93 | 73 |
| proprag | exact | MuSiQue | 0.380 | 0.390 | +0.010 | 0.4374 | 0.4689 | +0.0315 | 0.677 | 0.697 | +0.019 | 7/6/87 | 85 |

## LLM Binding Audit

Aggregate extraction stats are identical between exact and substring modes because the same LLM extraction is run; only title matching differs.

| Match | Attempts | Calls | Cache Hits | Failures | Empty Entity Responses | Empty / Calls | Cache Hit / Attempts |
|---|---:|---:|---:|---:|---:|---:|---:|
| substring | 6615 | 6361 | 254 | 0 | 3782 | 59.46% | 3.84% |
| exact | 6615 | 6361 | 254 | 0 | 3782 | 59.46% | 3.84% |

Title matching totals:

| Match Mode | Raw Entities | Exact Matches | Substring Matches | Unmatched |
|---|---:|---:|---:|---:|
| substring | 7118 | 1005 | 748 | 4149 |
| exact | 7123 | 1004 | 0 | 5326 |

## Substring vs Exact

| Pool | Dataset | Sub - Exact EM | Sub - Exact F1 | Sub - Exact R@5 | Substring Mode Matches exact/sub | Exact Mode Matches |
|---|---|---:|---:|---:|---:|---:|
| dense | 2Wiki | +0.010 | +0.0100 | -0.005 | 82/23 | 81 |
| dense | HotpotQA | +0.010 | +0.0150 | +0.020 | 111/79 | 110 |
| dense | MuSiQue | +0.040 | +0.0480 | +0.011 | 120/134 | 120 |
| hipporag | 2Wiki | +0.010 | +0.0129 | +0.008 | 104/27 | 104 |
| hipporag | HotpotQA | -0.010 | +0.0043 | +0.010 | 111/81 | 111 |
| hipporag | MuSiQue | +0.030 | +0.0239 | -0.002 | 121/144 | 121 |
| proprag | 2Wiki | +0.020 | +0.0278 | +0.007 | 120/28 | 120 |
| proprag | HotpotQA | +0.020 | +0.0223 | +0.015 | 110/89 | 110 |
| proprag | MuSiQue | -0.020 | -0.0209 | +0.013 | 126/143 | 127 |

## Interpretation

1. LLM binding API stability is good: 0 failures across 6361 calls per match mode. Empty entity responses are high at about 59%, but this is not an execution failure; many dependency-document pairs simply yield no extracted bridge entity.
2. Exact-only is cleaner but not sufficient across the board. It preserves clear gains on HippoRAG and 2Wiki dense, but it regresses dense HotpotQA and dense MuSiQue, and it is weaker than substring on PropRAG 2Wiki/HotpotQA.
3. Substring matching contributes real signal, especially on HotpotQA and MuSiQue, but it also creates risky matches. Examples include `Russia -> Grand Duke Vladimir Alexandrovich of Russia`, `France -> Rudolph of France`, `1926 -> Camille (1926 feature film)`, and `Muhammad -> Abdul-Aziz bin Muhammad`.
4. The method is not a pure exact-title mechanism. The full DAEC-LLM gain should not be described as coming only from exact entity/title binding. A more accurate statement is that LLM extraction provides bridge entities, exact title matches are the clean core, and substring matching adds recall with visible precision risk.
5. For paper framing, exact-only can be a conservative ablation. The main method should either keep substring with a documented precision filter, or replace raw substring with a safer alias/containment rule.

## Recommended Next Step

Before using substring results as the main claim, add one cheap guard:
- reject substring matches for numeric-only entities;
- reject one-token country/common-name entities unless the title exactly equals the entity or the dependency subquery type expects a location answer;
- require token-boundary containment and minimum normalized entity length for substring matches.

Then rerun this same limit100 audit only for substring-guarded mode against current substring and exact.
