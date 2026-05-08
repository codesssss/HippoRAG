# RankGPT-Style Sliding Reader Full1000

Date: 2026-05-08

Purpose: evaluate a RankGPT-style sliding-window local adaptation under the same controlled Qwen3-8B `/no_think` substrate used by the DBEC/DAEC experiments. This is not a GPT-3.5/4 RankGPT reproduction.

Important correction: `reports/rankgpt_sliding_reader_full1000_20260508/` is superseded. The corrected materialization filters selector rows by dataset before `query_index`; otherwise later-dataset rows can overwrite earlier datasets. All numbers below use `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/`.

## Method Boundary

| Aspect | Original RankGPT | This run |
|---|---|---|
| LLM | ChatGPT/GPT-4 | Qwen3-8B |
| Candidate pool | BM25 top100 | PropRAG pool100 |
| Ranking algorithm | Sliding-window permutation | Back-to-front sliding-window local adaptation, window=20, step=10 |
| Thinking | Unspecified/default | `/no_think` |
| Metric | IR nDCG | Multi-hop support metrics and QA EM/F1 |

## Selector Variant Check

Sliding-window is the most faithful RankGPT-style variant and improves single-pass `rank5_no_think` on overall selector Support R@5 for all three datasets. It remains below DAEC/SetR on selector support metrics.

| Dataset | Variant | Calls/query | Support R@5 | Complete@5 | Parse ok |
|---|---|---:|---:|---:|---:|
| 2Wiki | rank5_no_think | 1 | 89.6% | 78.7% | 100.0% |
| 2Wiki | select5_no_think | 1 | 88.5% | 75.2% | 100.0% |
| 2Wiki | sliding20_step10_no_think | 9 | 91.2% | 80.0% | 90.1% |
| HotpotQA | rank5_no_think | 1 | 86.6% | 76.2% | 100.0% |
| HotpotQA | select5_no_think | 1 | 85.0% | 73.3% | 100.0% |
| HotpotQA | sliding20_step10_no_think | 9 | 88.0% | 78.4% | 95.4% |
| MuSiQue | rank5_no_think | 1 | 65.8% | 37.4% | 100.0% |
| MuSiQue | select5_no_think | 1 | 65.2% | 36.6% | 100.0% |
| MuSiQue | sliding20_step10_no_think | 9 | 66.3% | 39.5% | 99.2% |

## Reader Results

EM/F1 are answer-reader metrics. Reader R@5 uses each method's paper-facing reader/eval recall field: DAEC-selective `selector_retrieval_metrics`, SetR-faithful `overall_recomputed`, and RankGPT-style sliding `overall_recomputed`. Selector Support R@5 is reported separately above as a diagnostic, not as the reader main-table support metric.

| Dataset | Method | EM | F1 | Reader R@5 |
|---|---|---:|---:|---:|
| 2Wiki | DAEC-selective | 0.6420 | 0.7118 | 94.1% |
| 2Wiki | SetR-faithful | 0.6030 | 0.6746 | 88.3% |
| 2Wiki | RankGPT-style sliding | 0.6020 | 0.6659 | 91.2% |
| HotpotQA | DAEC-selective | 0.6200 | 0.7473 | 96.0% |
| HotpotQA | SetR-faithful | 0.6250 | 0.7435 | 92.3% |
| HotpotQA | RankGPT-style sliding | 0.5660 | 0.6845 | 87.8% |
| MuSiQue | DAEC-selective | 0.3530 | 0.4548 | 74.7% |
| MuSiQue | SetR-faithful | 0.3440 | 0.4467 | 65.9% |
| MuSiQue | RankGPT-style sliding | 0.3190 | 0.4093 | 62.5% |

## RankGPT Reader Sanity

| Dataset | EM | F1 | Pipeline Recall@5 | External title Recall@5 | Question mismatches | Unmatched docs |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.6020 | 0.6659 | 0.9117 | 0.9028 | 0 | 0 |
| HotpotQA | 0.5660 | 0.6845 | 0.8780 | 0.9500 | 0 | 0 |
| MuSiQue | 0.3190 | 0.4093 | 0.6254 | 0.7372 | 0 | 0 |

## Paired Bootstrap

Query-paired percentile bootstrap with 10,000 resamples. Deltas are left method minus right method.

| Dataset | Comparison | Metric | Left | Right | Delta | 95% CI | P(delta > 0) | Excludes 0 |
|---|---|---|---:|---:|---:|---:|---:|---|
| 2Wiki | DAEC-selective - RankGPT-style sliding | EM | 0.6420 | 0.6020 | +0.0400 | [+0.0170, +0.0630] | 0.9995 | yes |
| 2Wiki | DAEC-selective - RankGPT-style sliding | F1 | 0.7118 | 0.6659 | +0.0460 | [+0.0244, +0.0674] | 1.0000 | yes |
| 2Wiki | SetR-faithful - RankGPT-style sliding | EM | 0.6030 | 0.6020 | +0.0010 | [-0.0220, +0.0240] | 0.5145 | no |
| 2Wiki | SetR-faithful - RankGPT-style sliding | F1 | 0.6746 | 0.6659 | +0.0087 | [-0.0139, +0.0316] | 0.7732 | no |
| HotpotQA | DAEC-selective - RankGPT-style sliding | EM | 0.6200 | 0.5660 | +0.0540 | [+0.0330, +0.0750] | 1.0000 | yes |
| HotpotQA | DAEC-selective - RankGPT-style sliding | F1 | 0.7473 | 0.6845 | +0.0628 | [+0.0418, +0.0838] | 1.0000 | yes |
| HotpotQA | SetR-faithful - RankGPT-style sliding | EM | 0.6250 | 0.5660 | +0.0590 | [+0.0370, +0.0810] | 1.0000 | yes |
| HotpotQA | SetR-faithful - RankGPT-style sliding | F1 | 0.7435 | 0.6845 | +0.0590 | [+0.0380, +0.0802] | 1.0000 | yes |
| MuSiQue | DAEC-selective - RankGPT-style sliding | EM | 0.3530 | 0.3190 | +0.0340 | [+0.0050, +0.0620] | 0.9895 | yes |
| MuSiQue | DAEC-selective - RankGPT-style sliding | F1 | 0.4548 | 0.4093 | +0.0455 | [+0.0185, +0.0733] | 0.9994 | yes |
| MuSiQue | SetR-faithful - RankGPT-style sliding | EM | 0.3440 | 0.3190 | +0.0250 | [+0.0000, +0.0510] | 0.9699 | no |
| MuSiQue | SetR-faithful - RankGPT-style sliding | F1 | 0.4467 | 0.4093 | +0.0373 | [+0.0117, +0.0625] | 0.9983 | yes |

## MuSiQue Missing-Slice Headroom

The overall result is negative for replacing DAEC/SetR, but listwise selection still recovers a complementary subset of source-visible MuSiQue supports that DBEC/SetR miss.

| Source | Method | Missing titles | Queries | New@5 |
|---|---|---:|---:|---:|
| single_pass | rankgpt_rank5_no_think | 96 | 72 | 20.8% |
| single_pass | rankgpt_select5_no_think | 96 | 72 | 25.0% |
| sliding | rankgpt_sliding20_step10_no_think | 96 | 72 | 24.0% |

## Cost Profile

| Method | LLM calls/query | Sequential calls/query | Operational note |
|---|---:|---:|---|
| DAEC-selective | ~8.4 | partly parallelizable | Existing binding/extraction path; token usage should be reported from the DAEC run logs if needed. |
| SetR-faithful | 1 | 1 | One listwise selection call before selected-only reader context. |
| RankGPT-style single-pass | 1 | 1 | Cheaper but less faithful to RankGPT sliding-window behavior. |
| RankGPT-style sliding | 9 | 9 | Back-to-front windows are sequential; measured run used 27,000 window calls for 3,000 queries. |

Token usage was not emitted in the local vLLM outputs, so this report does not claim measured input/output token totals. The defensible cost claim here is call count and sequential dependency; exact token accounting should be collected separately if the paper needs a latency/cost table.

## Interpretation

- Sliding-window is the correct RankGPT-style local adaptation to report; single-pass is a useful ablation but not the named comparison.
- Under the controlled Qwen3-8B `/no_think` substrate, RankGPT-style sliding is below DAEC-selective and SetR-faithful on reader F1 across 2Wiki, HotpotQA, and MuSiQue.
- The MuSiQue missing-slice New@5 result remains important: listwise full-pool inspection can recover `24.0%` of the source-visible missing supports, so the fixed pool is not theoretically exhausted.
- The paper claim should be bounded: DAEC/DBEC outperforms this RankGPT-style local adaptation under the controlled substrate; do not claim it outperforms original GPT-3.5/4 RankGPT.
- With-thinking variants are not evaluated because the controlled substrate uses `/no_think` for all compared methods; cross-thinking-mode comparison is future work.

## Files

- Comparison table: `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/comparison.csv`
- Paired CI: `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/paired_ci.csv` / `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/paired_ci.md`
- Machine summary: `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/summary.json`
- Corrected selected pools: `run_logs/rankgpt_sliding_full1000_20260508/*.selected_pool.json`
