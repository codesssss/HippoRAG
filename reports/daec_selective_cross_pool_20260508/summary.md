# DAEC Selective Binding Cross-Pool Paired-CI Audit

Date: 2026-05-08

Purpose: close the reviewer-facing cross-pool stability gap for the identifiability-gated DAEC-selective variant using existing full1000 outputs. No new LLM calls are made.

Protocol: fixed pool100 per retriever, Qwen3-8B `/no_think`, `qa_top_k=5`, `qa_doc_max_chars=2048`, title-uniqueness threshold `0.88`. Delta is left method minus right method; CIs use query-paired percentile bootstrap with 10,000 resamples.

## F1 Cross-Pool Summary

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

## Detailed F1 CI

| Pool | Dataset | Comparison | Left | Right | Delta | 95% CI | P(delta > 0) | Excludes 0 |
|---|---|---|---:|---:|---:|---:|---:|---|
| Dense | 2Wiki | DAEC-selective - Top-5 | 0.5798 | 0.4984 | +0.0814 | [+0.0556, +0.1076] | 1.0000 | yes |
| Dense | 2Wiki | DAEC - Top-5 | 0.5798 | 0.4984 | +0.0814 | [+0.0562, +0.1074] | 1.0000 | yes |
| Dense | 2Wiki | DAEC-selective - DAEC | 0.5798 | 0.5798 | +0.0000 | [+0.0000, +0.0000] | 0.0000 | no |
| Dense | HotpotQA | DAEC-selective - Top-5 | 0.7483 | 0.7106 | +0.0378 | [+0.0234, +0.0527] | 1.0000 | yes |
| Dense | HotpotQA | DAEC - Top-5 | 0.7483 | 0.7106 | +0.0378 | [+0.0232, +0.0530] | 1.0000 | yes |
| Dense | HotpotQA | DAEC-selective - DAEC | 0.7483 | 0.7483 | +0.0000 | [+0.0000, +0.0000] | 0.0000 | no |
| Dense | MuSiQue | DAEC-selective - Top-5 | 0.4002 | 0.3896 | +0.0106 | [-0.0097, +0.0305] | 0.8454 | no |
| Dense | MuSiQue | DAEC - Top-5 | 0.3918 | 0.3896 | +0.0022 | [-0.0194, +0.0237] | 0.5800 | no |
| Dense | MuSiQue | DAEC-selective - DAEC | 0.4002 | 0.3918 | +0.0084 | [-0.0031, +0.0200] | 0.9217 | no |
| HippoRAG | 2Wiki | DAEC-selective - Top-5 | 0.6431 | 0.5850 | +0.0581 | [+0.0329, +0.0834] | 1.0000 | yes |
| HippoRAG | 2Wiki | DAEC - Top-5 | 0.6431 | 0.5850 | +0.0581 | [+0.0331, +0.0831] | 1.0000 | yes |
| HippoRAG | 2Wiki | DAEC-selective - DAEC | 0.6431 | 0.6431 | +0.0000 | [+0.0000, +0.0000] | 0.0000 | no |
| HippoRAG | HotpotQA | DAEC-selective - Top-5 | 0.7403 | 0.7010 | +0.0393 | [+0.0241, +0.0553] | 1.0000 | yes |
| HippoRAG | HotpotQA | DAEC - Top-5 | 0.7403 | 0.7010 | +0.0393 | [+0.0240, +0.0549] | 1.0000 | yes |
| HippoRAG | HotpotQA | DAEC-selective - DAEC | 0.7403 | 0.7403 | +0.0000 | [+0.0000, +0.0000] | 0.0000 | no |
| HippoRAG | MuSiQue | DAEC-selective - Top-5 | 0.4206 | 0.3947 | +0.0259 | [+0.0052, +0.0462] | 0.9933 | yes |
| HippoRAG | MuSiQue | DAEC - Top-5 | 0.4142 | 0.3947 | +0.0195 | [-0.0012, +0.0408] | 0.9655 | no |
| HippoRAG | MuSiQue | DAEC-selective - DAEC | 0.4206 | 0.4142 | +0.0064 | [-0.0050, +0.0181] | 0.8586 | no |
| PropRAG | 2Wiki | DAEC-selective - Top-5 | 0.7118 | 0.6457 | +0.0661 | [+0.0444, +0.0879] | 1.0000 | yes |
| PropRAG | 2Wiki | DAEC - Top-5 | 0.7118 | 0.6457 | +0.0661 | [+0.0445, +0.0883] | 1.0000 | yes |
| PropRAG | 2Wiki | DAEC-selective - DAEC | 0.7118 | 0.7118 | +0.0000 | [+0.0000, +0.0000] | 0.0000 | no |
| PropRAG | HotpotQA | DAEC-selective - Top-5 | 0.7473 | 0.7227 | +0.0246 | [+0.0112, +0.0385] | 0.9998 | yes |
| PropRAG | HotpotQA | DAEC - Top-5 | 0.7473 | 0.7227 | +0.0246 | [+0.0112, +0.0381] | 0.9998 | yes |
| PropRAG | HotpotQA | DAEC-selective - DAEC | 0.7473 | 0.7473 | +0.0000 | [+0.0000, +0.0000] | 0.0000 | no |
| PropRAG | MuSiQue | DAEC-selective - Top-5 | 0.4548 | 0.4266 | +0.0282 | [+0.0076, +0.0492] | 0.9969 | yes |
| PropRAG | MuSiQue | DAEC - Top-5 | 0.4359 | 0.4266 | +0.0093 | [-0.0116, +0.0301] | 0.8112 | no |
| PropRAG | MuSiQue | DAEC-selective - DAEC | 0.4548 | 0.4359 | +0.0189 | [+0.0081, +0.0302] | 0.9996 | yes |

## Paper-Facing Takeaway

- DAEC-selective is non-negative relative to ungated DAEC in `9/9` pool-dataset cells by mean F1; statistically significant positive gains occur in `1/9` cells.
- DAEC-selective is positive relative to the same-pool Top-5 retriever baseline in `9/9` cells by mean F1; statistically significant positive gains occur in `8/9` cells.
- Router behavior confirms the gate is often score-preserving rather than score-changing: many abstentions reproduce the ungated DAEC top-5 exactly, especially outside MuSiQue.
- Paper-safe framing: the identifiability gate is a conservative cross-pool safety layer with concentrated positive effect, not a large universal improvement mechanism.

## Files

- Summary CSV: `reports/daec_selective_cross_pool_20260508/method_summary.csv`
- Paired CI CSV: `reports/daec_selective_cross_pool_20260508/paired_ci.csv`
- Router CSV: `reports/daec_selective_cross_pool_20260508/router_behavior.csv`
- JSON: `reports/daec_selective_cross_pool_20260508/summary.json`
