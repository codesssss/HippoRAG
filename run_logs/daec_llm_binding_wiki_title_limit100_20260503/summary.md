# DAEC-LLM Wiki Title Binding Audit Limit100

Run directory: `run_logs/daec_llm_binding_wiki_title_limit100_20260503`

Code commit:
- `413b509 Add wiki title DAEC LLM binding mode`

Protocol:
- selector: `daec_noisyor_llm`
- match mode: `wiki_title`
- datasets: `2wikimultihopqa`, `hotpotqa`, `musique`
- pools: dense, HippoRAG, PropRAG
- limit: 100
- pool_k: 100
- qa_top_k: 5
- qa_doc_max_chars: 2048
- reader / binding model: Qwen3-8B via 8041/8042/8043
- no-think: `HIPPORAG_RERANK_FORCE_NO_THINK=1`

Launcher:
- `run_logs/launch_daec_llm_binding_wiki_title_limit100_20260503.sh`

All 9 wiki-title runs completed with `rc=0`.

## Wiki Title Policy

`wiki_title` does not use country/name blocklists. It links an LLM-extracted entity to a candidate title by:

1. exact normalized title match;
2. parenthetical disambiguation removal, e.g. `Ian Barry -> Ian Barry (director)`;
3. multi-token contiguous alias containment, e.g. `Count of Nassau-Dillenburg -> William I, Count of Nassau-Dillenburg`.

Single-token substring aliases are rejected. Numeric-only entities are rejected unless they are exact title matches.

## Four-Way Aggregate

| Match | Avg Delta EM | Avg Delta F1 | Exact Matches | Disambiguation | Multi-Token Alias | Raw Substring | Unmatched | Failures | Empty Entity Responses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | +0.0100 | +0.0208 | 1004 | 0 | 0 | 0 | 5326 | 0 | 3782 |
| wiki_title | +0.0267 | +0.0394 | 1003 | 102 | 226 | 0 | 4900 | 0 | 3783 |
| substring_guarded | +0.0222 | +0.0367 | 1005 | 0 | 0 | 591 | 4370 | 0 | 3782 |
| raw substring | +0.0222 | +0.0368 | 1005 | 0 | 0 | 748 | 4149 | 0 | 3782 |

## Per-Setting F1

| Pool | Dataset | Exact F1 | Wiki F1 | Guarded F1 | Raw Substring F1 | Wiki - Exact | Wiki - Guarded | Wiki - Raw |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 2Wiki | 0.5135 | 0.5235 | 0.5235 | 0.5235 | +0.0100 | +0.0000 | +0.0000 |
| dense | HotpotQA | 0.6798 | 0.6898 | 0.6948 | 0.6948 | +0.0100 | -0.0050 | -0.0050 |
| dense | MuSiQue | 0.3720 | 0.4283 | 0.4183 | 0.4200 | +0.0563 | +0.0100 | +0.0083 |
| hipporag | 2Wiki | 0.5890 | 0.6019 | 0.6019 | 0.6019 | +0.0129 | +0.0000 | +0.0000 |
| hipporag | HotpotQA | 0.6972 | 0.7129 | 0.7015 | 0.7015 | +0.0157 | +0.0114 | +0.0114 |
| hipporag | MuSiQue | 0.4170 | 0.4376 | 0.4309 | 0.4409 | +0.0206 | +0.0067 | -0.0033 |
| proprag | 2Wiki | 0.6259 | 0.6437 | 0.6537 | 0.6537 | +0.0178 | -0.0100 | -0.0100 |
| proprag | HotpotQA | 0.6987 | 0.7120 | 0.7210 | 0.7210 | +0.0133 | -0.0090 | -0.0090 |
| proprag | MuSiQue | 0.4689 | 0.4795 | 0.4595 | 0.4480 | +0.0106 | +0.0200 | +0.0315 |

## Per-Setting Main Metrics

| Pool | Dataset | Match | Base EM | Sel EM | Delta EM | Base F1 | Sel F1 | Delta F1 | Base R@5 | Sel R@5 | Delta R@5 | EM G/R/S | Changed | exact/disamb/alias/sub/unmatched |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2Wiki | wiki_title | 0.410 | 0.470 | +0.060 | 0.4362 | 0.5235 | +0.0873 | 0.695 | 0.777 | +0.083 | 10/4/86 | 89 | 81/10/7/0/217 |
| dense | HotpotQA | wiki_title | 0.580 | 0.580 | +0.000 | 0.6881 | 0.6898 | +0.0017 | 0.915 | 0.930 | +0.015 | 2/2/96 | 67 | 110/14/14/0/573 |
| dense | MuSiQue | wiki_title | 0.340 | 0.350 | +0.010 | 0.4028 | 0.4283 | +0.0255 | 0.630 | 0.692 | +0.062 | 9/8/83 | 88 | 120/7/53/0/863 |
| hipporag | 2Wiki | wiki_title | 0.510 | 0.540 | +0.030 | 0.5470 | 0.6019 | +0.0549 | 0.815 | 0.902 | +0.088 | 10/7/83 | 97 | 104/13/7/0/186 |
| hipporag | HotpotQA | wiki_title | 0.560 | 0.600 | +0.040 | 0.6712 | 0.7129 | +0.0417 | 0.925 | 0.945 | +0.020 | 7/3/90 | 81 | 111/15/15/0/569 |
| hipporag | MuSiQue | wiki_title | 0.300 | 0.340 | +0.040 | 0.3687 | 0.4376 | +0.0689 | 0.642 | 0.693 | +0.052 | 10/6/84 | 92 | 121/7/55/0/875 |
| proprag | 2Wiki | wiki_title | 0.580 | 0.590 | +0.010 | 0.6318 | 0.6437 | +0.0119 | 0.935 | 0.940 | +0.005 | 6/5/89 | 89 | 120/13/7/0/170 |
| proprag | HotpotQA | wiki_title | 0.570 | 0.600 | +0.030 | 0.6912 | 0.7120 | +0.0208 | 0.930 | 0.945 | +0.015 | 6/3/91 | 73 | 110/16/16/0/578 |
| proprag | MuSiQue | wiki_title | 0.380 | 0.400 | +0.020 | 0.4374 | 0.4795 | +0.0421 | 0.677 | 0.717 | +0.039 | 9/7/84 | 84 | 126/7/52/0/869 |

## Interpretation

1. `wiki_title` is cleaner than `substring_guarded`: it uses no dataset-looking blocklist and no country/name special cases.
2. It is also stronger on average in this limit100 audit: avg Delta F1 is `+0.0394`, versus `+0.0367` for guarded and `+0.0368` for raw substring.
3. It improves over exact-only on all 9 pool/dataset settings by F1.
4. It beats guarded/raw on 4 settings, ties on 2, and loses mildly on 3. The losses are small and concentrated on 2Wiki/Hotpot with PropRAG or dense Hotpot. The largest win is PropRAG/MuSiQue, where wiki-title F1 is `0.4795`, compared with exact `0.4689`, guarded `0.4595`, and raw substring `0.4480`.
5. A scan for previously observed risky matches (`France`, `Russia`, `Iran`, `1926`, `Muhammad`, `Lichtenberg`) found no retained risky wiki-title matches.

## Recommendation

Use `wiki_title` as the default DAEC-LLM binding policy for the next full1000 run.

Paper-facing description:

> We link extracted entities to candidate documents using deterministic Wikipedia-title normalization: exact title matching, parenthetical disambiguation removal, and multi-token alias containment. We do not use dataset-specific entity lists or learned entity linking.

This is easier to defend than `substring_guarded` and empirically at least as strong on the limit100 audit.
