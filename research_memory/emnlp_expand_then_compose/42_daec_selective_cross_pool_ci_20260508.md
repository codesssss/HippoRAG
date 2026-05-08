# DAEC-Selective Cross-Pool CI Deposition - 2026-05-08

## Purpose

This note consolidates the cross-pool paired-CI audit for the DAEC-selective identifiability gate.

It is a reviewer-defense result deposition, not a new method direction. The goal is to preserve:

- the 3-pool x 3-dataset paired-CI results;
- the exact claim boundary for the identifiability gate;
- the router behavior evidence showing when the gate is score-preserving vs score-changing;
- the canonical report paths for paper writing.

## Executive Takeaway

The identifiability gate is a conservative cross-pool safety layer, not a large universal improvement mechanism.

Across 3 retrieval pools and 3 datasets:

- DAEC-selective is non-negative relative to ungated DAEC in `9/9` pool-dataset cells by mean F1.
- Statistically significant positive F1 gain over ungated DAEC occurs in `1/9` cells: PropRAG-MuSiQue.
- DAEC-selective is positive relative to the same-pool Top-5 retriever baseline in `9/9` cells by mean F1.
- Statistically significant positive F1 gain over Top-5 occurs in `8/9` cells; the only non-significant cell is Dense-MuSiQue.

This closes the reviewer-facing cross-pool paired-CI gap, but it also sharpens the framing:

```text
DAEC-selective preserves DAEC's cross-pool gains while providing a concentrated
MuSiQue safety improvement; it should not be presented as a broadly stronger
variant on every dataset.
```

## Canonical Artifacts

Primary report:

- `reports/daec_selective_cross_pool_20260508/summary.md`
- `reports/daec_selective_cross_pool_20260508/method_summary.csv`
- `reports/daec_selective_cross_pool_20260508/paired_ci.csv`
- `reports/daec_selective_cross_pool_20260508/router_behavior.csv`
- `reports/daec_selective_cross_pool_20260508/summary.json`

Implementation:

- `scripts/analyze_daec_selective_cross_pool.py`
- `tests/test_daec_selective_cross_pool.py`

Input runs:

- Dense/HippoRAG selective: `run_logs/daec_selective_titleuniq_dense_hipporag_full1000_20260506/`
- Dense/HippoRAG ungated DAEC: `run_logs/daec_llm_wiki_title_dense_hipporag_full1000_20260503/`
- PropRAG selective: `run_logs/daec_selective_titleuniq_proprag_full1000_20260506/`
- PropRAG ungated DAEC: `run_logs/daec_llm_wiki_title_proprag_full1000_20260503/`

Protocol:

- Fixed pool100 for each retriever.
- Qwen3-8B `/no_think`.
- `qa_top_k=5`.
- `qa_doc_max_chars=2048`.
- Title-uniqueness threshold `0.88`.
- Query-paired percentile bootstrap with `10,000` resamples.
- No new LLM calls.

## Cross-Pool F1 Summary

| Pool | Dataset | Top-5 F1 | DAEC F1 | DAEC-selective F1 | Selective - Top-5 F1 | 95% CI | Selective - DAEC F1 | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 2Wiki | 0.4984 | 0.5798 | 0.5798 | +0.0814 | [+0.0556, +0.1076] | +0.0000 | [+0.0000, +0.0000] |
| Dense | HotpotQA | 0.7106 | 0.7483 | 0.7483 | +0.0378 | [+0.0234, +0.0527] | +0.0000 | [+0.0000, +0.0000] |
| Dense | MuSiQue | 0.3896 | 0.3918 | 0.4002 | +0.0106 | [-0.0097, +0.0305] | +0.0084 | [-0.0031, +0.0200] |
| HippoRAG | 2Wiki | 0.5850 | 0.6431 | 0.6431 | +0.0581 | [+0.0329, +0.0834] | +0.0000 | [+0.0000, +0.0000] |
| HippoRAG | HotpotQA | 0.7010 | 0.7403 | 0.7403 | +0.0393 | [+0.0241, +0.0553] | +0.0000 | [+0.0000, +0.0000] |
| HippoRAG | MuSiQue | 0.3947 | 0.4142 | 0.4206 | +0.0259 | [+0.0052, +0.0462] | +0.0064 | [-0.0050, +0.0181] |
| PropRAG | 2Wiki | 0.6457 | 0.7118 | 0.7118 | +0.0661 | [+0.0444, +0.0879] | +0.0000 | [+0.0000, +0.0000] |
| PropRAG | HotpotQA | 0.7227 | 0.7473 | 0.7473 | +0.0246 | [+0.0112, +0.0385] | +0.0000 | [+0.0000, +0.0000] |
| PropRAG | MuSiQue | 0.4266 | 0.4359 | 0.4548 | +0.0282 | [+0.0076, +0.0492] | +0.0189 | [+0.0081, +0.0302] |

## Router Behavior

| Pool | Dataset | Bind | Abstain | Selective same as DAEC | Abstain same as DAEC | Abstain changed from DAEC |
|---|---|---:|---:|---:|---:|---:|
| Dense | 2Wiki | 598 (59.8%) | 402 (40.2%) | 1000/1000 (100.0%) | 402 | 0 |
| Dense | HotpotQA | 648 (64.8%) | 352 (35.2%) | 1000/1000 (100.0%) | 352 | 0 |
| Dense | MuSiQue | 379 (37.9%) | 621 (62.1%) | 774/1000 (77.4%) | 395 | 226 |
| HippoRAG | 2Wiki | 682 (68.2%) | 318 (31.8%) | 1000/1000 (100.0%) | 318 | 0 |
| HippoRAG | HotpotQA | 649 (64.9%) | 351 (35.1%) | 1000/1000 (100.0%) | 351 | 0 |
| HippoRAG | MuSiQue | 380 (38.0%) | 620 (62.0%) | 767/1000 (76.7%) | 387 | 233 |
| PropRAG | 2Wiki | 698 (69.8%) | 302 (30.2%) | 1000/1000 (100.0%) | 302 | 0 |
| PropRAG | HotpotQA | 657 (65.7%) | 343 (34.3%) | 1000/1000 (100.0%) | 343 | 0 |
| PropRAG | MuSiQue | 371 (37.1%) | 629 (62.9%) | 738/1000 (73.8%) | 367 | 262 |

Interpretation:

- On 2Wiki and HotpotQA, the gate often abstains but still reproduces the ungated DAEC top-5 exactly.
- On MuSiQue, the gate changes a meaningful fraction of DAEC selections: 226 Dense, 233 HippoRAG, and 262 PropRAG abstentions differ from ungated DAEC.
- Only PropRAG-MuSiQue converts that selection change into a statistically significant answer-F1 gain over ungated DAEC.

## Paper Claim Boundary

Allowed:

```text
Across Dense, HippoRAG, and PropRAG pool100 settings, DAEC-selective is
non-negative relative to ungated DAEC in all 9 dataset-pool cells by mean F1,
with a significant gain on PropRAG-MuSiQue.
```

Allowed:

```text
DAEC-selective significantly improves over the same-pool Top-5 retriever
baseline in 8/9 cells and is positive in all 9 cells by mean F1.
```

Not allowed:

```text
The identifiability gate substantially improves DAEC across all datasets.
```

Not allowed:

```text
The gate is the main source of DAEC's 2Wiki/HotpotQA gains.
```

Best paper framing:

```text
The identifiability gate is a conservative safety mechanism: it preserves
DAEC's cross-pool behavior and selectively repairs high-ambiguity MuSiQue
cases, but its contribution is concentrated rather than universal.
```
