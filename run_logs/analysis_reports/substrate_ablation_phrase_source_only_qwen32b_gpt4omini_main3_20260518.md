# Substrate Ablation: Fact-Source vs Phrase-Source-Only

Protocol: Qwen3-32B no-think upstream, GPT-4o-mini reader, main3 full1000. Retrieval R@5 and AllGold@5 use PCEC eval summaries; EM/F1 use reader reports.

| Dataset | Full R@5 | Phrase R@5 | ΔR@5 | Full All@5 | Phrase All@5 | ΔAll@5 | Full EM | Phrase EM | ΔEM | Full F1 | Phrase F1 | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 96.30 | 94.53 | -1.77 | 89.70 | 84.80 | -4.90 | 65.90 | 64.60 | -1.30 | 74.68 | 72.77 | -1.91 |
| HotpotQA | 96.55 | 96.50 | -0.05 | 93.40 | 93.40 | +0.00 | 62.80 | 61.90 | -0.90 | 75.57 | 75.14 | -0.43 |
| MuSiQue | 76.42 | 74.98 | -1.44 | 51.00 | 47.80 | -3.20 | 37.20 | 36.70 | -0.50 | 48.91 | 47.87 | -1.04 |

## Average Over Main3

| Metric | Full | Phrase-source-only | Δ Phrase-Full |
|---|---:|---:|---:|
| retrieval_R5 | 89.76 | 88.67 | -1.09 |
| AllGold@5 | 78.03 | 75.33 | -2.70 |
| EM | 55.30 | 54.40 | -0.90 |
| F1 | 66.39 | 65.26 | -1.13 |

Interpretation: this is the substrate-isolation ablation requested by reviewers. It is not uniformly worse on QA, but it loses retrieval coverage on 2Wiki and MuSiQue, especially AllGold@5, which supports a cautious claim that fact-source grounding improves connected evidence coverage. HotpotQA is effectively tied on coverage.
