# PCEC Residual-Budget Sweep With GPT-4o-mini Reader

Date: 2026-05-19

This report uses the aligned2 rerun:

- Retrieval outputs: `run_logs/pcec_residual_budget_sweep_aligned2_20260519/evals/`
- Reader outputs: `run_logs/pcec_residual_budget_sweep_aligned2_20260519/reader_qa/reports/`

The earlier reader sweep under `run_logs/pcec_residual_budget_sweep_retrieval_only_20260519` is not used here because its `retrieved_doc_indices_top5` were external pool IDs rather than reader-local document IDs. The aligned2 run preserves `reader_pool_doc_ids` and restores valid reader-side R@5/All@5.

## Launcher Status

All six aligned2 GPT-4o-mini reader jobs finished successfully:

| Variant | Dataset | Status |
|---|---|---|
| `r=2,m=3` | HotpotQA | done |
| `r=2,m=3` | 2WikiMultiHopQA | done |
| `r=2,m=3` | MuSiQue | done |
| `r=5,m=0` | HotpotQA | done |
| `r=5,m=0` | 2WikiMultiHopQA | done |
| `r=5,m=0` | MuSiQue | done |

## Compact Summary

All numbers are percentages. Reader R@5/All@5 come from GPT-4o-mini reader reports. Retrieval-title R@5/All@5 come from the PCEC retrieval summaries.

| Variant | HotpotQA F1 | 2Wiki F1 | MuSiQue F1 | Avg F1 | Avg EM | Avg R@5(reader) | Avg All@5(reader) | Avg R@5(retr-title) | Avg All@5(retr-title) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r=0,m=5` no admission | 74.78 | 72.79 | 48.33 | 65.30 | 54.47 | 86.79 | 71.77 | 87.57 | 73.23 |
| `r=1,m=4` current | 75.57 | 74.68 | 48.91 | 66.39 | 55.30 | 88.74 | 76.03 | 89.76 | 78.03 |
| `r=2,m=3` | 75.94 | 74.99 | 48.90 | 66.61 | 55.30 | 88.38 | 74.90 | 89.43 | 76.87 |
| `r=5,m=0` no prefix | 73.59 | 71.94 | 44.28 | 63.27 | 52.73 | 85.17 | 69.40 | 86.41 | 71.43 |

## Detailed Reader Metrics

| Variant | Dataset | R@5 | All@5 | EM | F1 |
|---|---|---:|---:|---:|---:|
| `r=0,m=5` no admission | HotpotQA | 95.05 | 90.50 | 62.00 | 74.78 |
| `r=0,m=5` no admission | 2WikiMultiHopQA | 93.50 | 82.60 | 64.20 | 72.79 |
| `r=0,m=5` no admission | MuSiQue | 71.83 | 42.20 | 37.20 | 48.33 |
| `r=1,m=4` current | HotpotQA | 96.50 | 93.30 | 62.80 | 75.57 |
| `r=1,m=4` current | 2WikiMultiHopQA | 96.23 | 89.50 | 65.90 | 74.68 |
| `r=1,m=4` current | MuSiQue | 73.51 | 45.30 | 37.20 | 48.91 |
| `r=2,m=3` | HotpotQA | 96.35 | 93.10 | 62.80 | 75.94 |
| `r=2,m=3` | 2WikiMultiHopQA | 95.33 | 86.40 | 66.20 | 74.99 |
| `r=2,m=3` | MuSiQue | 73.47 | 45.20 | 36.90 | 48.90 |
| `r=5,m=0` no prefix | HotpotQA | 94.50 | 89.40 | 60.80 | 73.59 |
| `r=5,m=0` no prefix | 2WikiMultiHopQA | 91.50 | 78.50 | 63.90 | 71.94 |
| `r=5,m=0` no prefix | MuSiQue | 69.53 | 40.30 | 33.50 | 44.28 |

## Interpretation

The sweep supports prefix preservation as a stability constraint rather than a cosmetic trick. Removing prefix preservation entirely (`r=5,m=0`) hurts every dataset and drops average F1 by 3.12 points relative to the current `r=1,m=4` setting.

The `r=2,m=3` setting is close: it improves average F1 by 0.22 points over the current setting, but it lowers reader All@5 by 1.13 points and retrieval-title All@5 by 1.16 points. Since the paper's method claim is evidence-set coverage under a tight reader budget, `r=1,m=4` remains the more conservative and support-preserving default.

