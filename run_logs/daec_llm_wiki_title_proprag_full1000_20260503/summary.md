# DAEC-LLM Wiki Title PropRAG Full1000

Run directory: `run_logs/daec_llm_wiki_title_proprag_full1000_20260503`

Launcher:
- `run_logs/launch_daec_llm_wiki_title_proprag_full1000_20260503.sh`

Code:
- `413b509 Add wiki title DAEC LLM binding mode`
- `5a933ab Add wiki-title PropRAG full1000 launcher`

Protocol:
- pool: PropRAG pool100
- selector: `daec_noisyor_llm`
- binding mode: `wiki_title`
- limit: 1000
- qa_top_k: 5
- qa_doc_max_chars: 2048
- reader/binding model: Qwen3-8B (`qwen3-8b-train`)
- endpoints: 8041/8042/8043
- no-think: `HIPPORAG_RERANK_FORCE_NO_THINK=1`

All three runs completed with `rc=0`.

## Main Results

| Dataset | Baseline EM/F1/R@5 | Raw Substring EM/F1/R@5 | Wiki Title EM/F1/R@5 | Wiki - Baseline | Wiki - Raw |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 0.575 / 0.6457 / 0.9028 | 0.648 / 0.7171 / 0.9460 | 0.642 / 0.7118 / 0.9410 | +0.067 / +0.0661 / +0.0382 | -0.006 / -0.0053 / -0.0050 |
| HotpotQA | 0.595 / 0.7227 / 0.9500 | 0.621 / 0.7502 / 0.9615 | 0.620 / 0.7473 / 0.9605 | +0.025 / +0.0246 / +0.0105 | -0.001 / -0.0029 / -0.0010 |
| MuSiQue | 0.330 / 0.4266 / 0.7131 | 0.338 / 0.4358 / 0.7310 | 0.337 / 0.4359 / 0.7269 | +0.007 / +0.0093 / +0.0138 | -0.001 / +0.0001 / -0.0041 |

## Paired EM

| Dataset | Wiki Gain / Regression / Same | Changed Queries |
|---|---:|---:|
| 2Wiki | 105 / 38 / 857 | 896 |
| HotpotQA | 40 / 15 / 945 | 720 |
| MuSiQue | 62 / 55 / 883 | 882 |

## Binding Audit

| Dataset | Exact | Disambiguation | Multi-Token Alias | Raw Substring | Unmatched | Failures | Empty Entity Responses |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 1226 | 151 | 128 | 0 | 2486 | 0 | 5167 |
| HotpotQA | 1038 | 195 | 265 | 0 | 5408 | 0 | 3643 |
| MuSiQue | 1159 | 69 | 438 | 0 | 8101 | 0 | 3750 |

## Interpretation

1. `wiki_title` keeps almost all raw substring full1000 performance while avoiding raw substring matches entirely.
2. Compared with PropRAG baseline, wiki_title DAEC-LLM improves all three datasets:
   - 2Wiki: +0.067 EM / +0.0661 F1
   - HotpotQA: +0.025 EM / +0.0246 F1
   - MuSiQue: +0.007 EM / +0.0093 F1
3. Compared with raw substring DAEC-LLM, wiki_title is slightly lower on 2Wiki and HotpotQA but essentially tied on MuSiQue F1.
4. This supports using `wiki_title` as the paper-facing DAEC-LLM binding policy: cleaner than raw substring, no dataset-specific guard list, and full1000 performance close to the stronger noisy linker.

## Claim Boundary

The result supports a conservative claim:

> In title-indexed multi-hop QA with a PropRAG fixed pool, DAEC-LLM with canonical title linking improves over the PropRAG top-5 baseline across 2Wiki, HotpotQA, and MuSiQue.

It does not by itself establish cross-pool full1000 generalization for `wiki_title`; Dense and HippoRAG full1000 would be needed for that exact claim.
