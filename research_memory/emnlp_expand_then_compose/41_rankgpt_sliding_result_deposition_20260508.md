# RankGPT-Style Sliding Result Deposition - 2026-05-08

## Purpose

This note consolidates the final RankGPT-style comparison and the connected MuSiQue fixed-pool diagnostics into paper-facing research memory.

It is a result deposition, not a new experiment plan. The goal is to preserve:

- what was actually tested;
- which result files are canonical;
- what can be claimed in the DBEC/DAEC paper;
- what should not be overclaimed;
- why more RankGPT/MuSiQue-fix experiments should stop for now.

## Executive Takeaway

The RankGPT-style comparison is now paper-ready.

Under the controlled Qwen3-8B `/no_think` substrate, the sliding-window RankGPT-style local adaptation is lower than DAEC-selective and SetR-faithful on reader F1 across all three full1000 datasets:

| Dataset | DAEC-selective F1 | SetR-faithful F1 | RankGPT-style sliding F1 | Decision |
|---|---:|---:|---:|---|
| 2Wiki | 0.7118 | 0.6746 | 0.6659 | DAEC wins; SetR vs RankGPT-style is a tie-range result |
| HotpotQA | 0.7473 | 0.7435 | 0.6845 | DAEC/SetR both win |
| MuSiQue | 0.4548 | 0.4467 | 0.4093 | DAEC/SetR both win |

But RankGPT-style listwise full-pool inspection still recovers a complementary subset of MuSiQue source-visible missing supports:

| Method | MuSiQue source-visible missing New@5 |
|---|---:|
| DBEC / DAEC | 0.0% |
| SetR-faithful | 0.0% |
| RankGPT single-pass rank5 | 20.8% |
| RankGPT single-pass select5 | 25.0% |
| RankGPT sliding20_step10 | 24.0% |

So the conclusion is not "listwise reranking is useless." The correct conclusion is:

```text
RankGPT-style listwise selection exposes complementary fixed-pool headroom,
but it does not replace DAEC/DBEC or SetR-faithful under the controlled
Qwen3-8B /no_think reader setting.
```

## Canonical Artifacts

Primary paper-ready report:

- `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/summary.md`
- `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/comparison.csv`
- `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/paired_ci.md`
- `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/paired_ci.csv`
- `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/summary.json`

Selector-level sliding-window report:

- `reports/rankgpt_fixed_pool_baseline_sliding_full1000_20260508/summary.md`
- `reports/rankgpt_fixed_pool_baseline_sliding_full1000_20260508/method_summary.csv`
- `reports/rankgpt_fixed_pool_baseline_sliding_full1000_20260508/selector_rows.csv`
- `reports/rankgpt_fixed_pool_baseline_sliding_full1000_20260508/musique_missing_summary.csv`

Single-pass RankGPT-style ablation:

- `reports/rankgpt_fixed_pool_baseline_full1000_1based_20260508/summary.md`

Updated MuSiQue synthesis:

- `reports/musique_failure_synthesis_20260507/summary.md`
- `reports/musique_failure_synthesis_20260507/evidence_table.csv`
- `reports/musique_failure_synthesis_20260507/decision_matrix.csv`
- `reports/musique_failure_synthesis_20260507/summary.json`

Implementation:

- `scripts/run_rankgpt_fixed_pool_baseline.py`
- `scripts/apply_rankgpt_selection_to_pool.py`
- `scripts/analyze_rankgpt_sliding_reader.py`
- `tests/test_rankgpt_fixed_pool_baseline.py`
- `tests/test_apply_rankgpt_selection_to_pool.py`

Commit:

- `d191f31 add rankgpt sliding reader comparison`

Important superseded artifact:

- Do not cite `reports/rankgpt_sliding_reader_full1000_20260508/`.
- It was generated before the selected-pool materialization dataset-filter bug was fixed.
- The corrected directory is `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/`.

Large local raw artifacts were intentionally not committed:

- `reports/rankgpt_fixed_pool_baseline_sliding_full1000_20260508/llm_cache.jsonl`
- `reports/rankgpt_fixed_pool_baseline_sliding_full1000_20260508/prompt_outputs.jsonl`
- `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/*.eval.json`
- `run_logs/rankgpt_sliding_full1000_20260508/*.selected_pool.json`
- `run_logs/rankgpt_sliding_full1000_20260508/*.trace.json`

## What Was Tested

### Controlled Substrate

All RankGPT-style runs use:

- fixed PropRAG pool100;
- Qwen3-8B local endpoints;
- `/no_think`;
- same Qwen3-8B reader setting as the DAEC/SetR comparisons;
- `qa_top_k=5`;
- `qa_doc_max_chars=2048`.

This is a controlled local adaptation, not a GPT-3.5/4 RankGPT reproduction.

### Single-Pass Variants

Two single-pass selector variants were run first:

- `rank5_no_think`: ask the model to output a top-5 ranking;
- `select5_no_think`: ask the model to directly select 5 support passages.

These are useful ablations, but they are not faithful enough to be named as RankGPT.

### Sliding-Window Variant

The final named comparison is:

- `sliding20_step10_no_think`;
- window size `20`;
- step `10`;
- rank range `0..100`;
- back-to-front windows: `(80,100), (70,90), ..., (0,20)`;
- 9 sequential LLM calls/query;
- each call ranks all passages in the local window;
- final reader top-5 comes from the final global permutation.

This closes the main reviewer attack that a single-pass listwise selector is not RankGPT-style.

## Selector-Level Results

Sliding-window improves over single-pass `rank5_no_think` on selector Support R@5 for all three datasets, but remains lower than DAEC/SetR on support metrics.

| Dataset | Variant | Calls/query | Support R@5 | Complete@5 | Parse ok |
|---|---:|---:|---:|---:|---:|
| 2Wiki | rank5_no_think | 1 | 89.6% | 78.7% | 100.0% |
| 2Wiki | select5_no_think | 1 | 88.5% | 75.2% | 100.0% |
| 2Wiki | sliding20_step10_no_think | 9 | 91.2% | 80.0% | 90.1% |
| HotpotQA | rank5_no_think | 1 | 86.6% | 76.2% | 100.0% |
| HotpotQA | select5_no_think | 1 | 85.0% | 73.3% | 100.0% |
| HotpotQA | sliding20_step10_no_think | 9 | 88.0% | 78.4% | 95.4% |
| MuSiQue | rank5_no_think | 1 | 65.8% | 37.4% | 100.0% |
| MuSiQue | select5_no_think | 1 | 65.2% | 36.6% | 100.0% |
| MuSiQue | sliding20_step10_no_think | 9 | 66.3% | 39.5% | 99.2% |

Interpretation:

- Sliding-window is the right paper-facing RankGPT-style variant.
- Single-pass remains useful only as a cheap ablation.
- The selector-level result already rejects the idea that local Qwen3-8B listwise reranking dominates DAEC/SetR overall.

## Reader Results

The corrected reader run uses selected pools materialized after fixing dataset filtering in `scripts/apply_rankgpt_selection_to_pool.py`.

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

Key paired-bootstrap F1 rows:

| Dataset | Comparison | Delta F1 | 95% CI | Decision |
|---|---|---:|---:|---|
| 2Wiki | DAEC-selective - RankGPT-style sliding | +0.0460 | [+0.0244, +0.0674] | DAEC wins |
| HotpotQA | DAEC-selective - RankGPT-style sliding | +0.0628 | [+0.0418, +0.0838] | DAEC wins |
| MuSiQue | DAEC-selective - RankGPT-style sliding | +0.0455 | [+0.0185, +0.0733] | DAEC wins |
| 2Wiki | SetR-faithful - RankGPT-style sliding | +0.0087 | [-0.0139, +0.0316] | tie-range |
| HotpotQA | SetR-faithful - RankGPT-style sliding | +0.0590 | [+0.0380, +0.0802] | SetR wins |
| MuSiQue | SetR-faithful - RankGPT-style sliding | +0.0373 | [+0.0117, +0.0625] | SetR wins |

Interpretation:

- DAEC-selective beats RankGPT-style sliding on all three datasets.
- SetR-faithful is statistically tied with RankGPT-style sliding on 2Wiki F1, but beats it on HotpotQA and MuSiQue.
- RankGPT-style sliding is not a stronger full1000 baseline under the controlled local substrate.

## MuSiQue Missing-Slice Interpretation

The MuSiQue source-visible missing-gold slice is the only strong positive signal for listwise selection:

| Method | Missing titles | Queries | New@5 |
|---|---:|---:|---:|
| dbec_selective | 96 | 72 | 0.0% |
| setr_faithful | 96 | 72 | 0.0% |
| rankgpt_rank5_no_think | 96 | 72 | 20.8% |
| rankgpt_select5_no_think | 96 | 72 | 25.0% |
| rankgpt_sliding20_step10_no_think | 96 | 72 | 24.0% |

This means:

- fixed-pool candidate generation is not theoretically exhausted;
- some source-visible MuSiQue supports are discoverable by direct listwise full-pool inspection;
- the signal is local and complementary;
- the signal does not translate into a full-method win after reader evaluation.

This should be used in the paper as limitation/future-work evidence, not as a new method direction.

## Engineering Lessons

### Sliding Window Was Necessary

Single-pass listwise selection is a useful ablation, but a reviewer could fairly object that it is not RankGPT-style. The sliding-window local adaptation closes that attack surface.

### Dataset Filtering Was A Real Bug

The original materialization keyed selector rows only by `query_index`. Because the selector CSV contains all three datasets, later rows could overwrite earlier rows.

Fix:

- add `--dataset`;
- filter rows by `dataset` / `base_dataset`;
- only then key by `query_index`.

Impact:

- old 2Wiki/HotpotQA reader numbers from `reports/rankgpt_sliding_reader_full1000_20260508/` are invalid;
- corrected results are in `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/`;
- MuSiQue old value happened to match because MuSiQue was last, but the old directory should still not be cited.

### Cost Claim Must Be Conservative

What is measured/reliable:

- single-pass RankGPT-style: 1 LLM call/query;
- sliding-window RankGPT-style: 9 sequential LLM calls/query;
- full sliding selector run: 27,000 window calls for 3,000 queries.

What is not measured:

- exact input/output token totals;
- exact wall-clock per-query latency under non-batched paper conditions.

Paper-safe cost wording:

```text
RankGPT-style sliding-window reranking requires 9 sequential listwise calls
per query over pool100 under our local adaptation, whereas DAEC uses a
different multi-call binding/extraction profile and SetR-faithful uses one
selection call. We report call structure rather than claiming exact token
cost parity.
```

Do not invent token totals unless a token-accounting run is added.

## Paper-Facing Claim Boundary

Allowed:

```text
Under the controlled Qwen3-8B /no_think substrate, DAEC-selective outperforms
our RankGPT-style sliding-window local adaptation on reader F1 across all
three datasets.
```

Allowed:

```text
RankGPT-style listwise full-pool inspection recovers a complementary subset
of MuSiQue source-visible supports missed by DAEC/SetR, suggesting remaining
fixed-pool headroom and possible future hybrid directions.
```

Allowed:

```text
We do not claim to reproduce GPT-3.5/4 RankGPT; this is a controlled local
adaptation over PropRAG pool100 using Qwen3-8B /no_think.
```

Not allowed:

```text
DBEC outperforms RankGPT.
```

Not allowed:

```text
RankGPT is not competitive.
```

Not allowed:

```text
Listwise reranking is useless.
```

Not allowed:

```text
Fixed-pool candidate generation is exhausted.
```

## Final Decision

Stop this experimental line for now.

Do not run:

- with-thinking RankGPT-style variant;
- GPT-4 / GPT-3.5 RankGPT;
- cross-pool RankGPT;
- prompt-tuned RankGPT variants;
- more MuSiQue-specific fixed-pool rescue pilots.

Use the current artifacts as:

- reviewer-defense baseline;
- failure-analysis evidence;
- future-work motivation for hybrid structured-composition plus listwise rescue.

Immediate next step:

- write the DBEC/DAEC paper.

The RankGPT-style baseline is now good enough to prevent a fair reviewer from saying "you ignored listwise LLM reranking," while preserving a clean and bounded claim.
