# SetR-faithful PropRAG Full1000

Date: 2026-05-07

Purpose: rerun the SetR-style PropRAG k20 baseline without rank-order fill-to-5 so the reader sees only LLM-selected passages. This matches the original SetR paper's variable-size/adaptive-context setting more closely than the previous `SetR + Fill@5` implementation.

## Implementation

- Selection JSONL is reused from `run_logs/setr_full1000_20260503`.
- `scripts/apply_setr_selection_to_pool.py --fallback_mode selected_only` keeps only parsed LLM-selected passages.
- Empty parse fallback is `--empty_fallback_top_n 1`, but it was never triggered in this run.
- Reader is still invoked with `--qa_top_k 5`; because the external pool contains only selected passages, the effective reader budget is the selected-only pool size.

## Selection Sanity

| Dataset | Records | Parse success | Empty fallback | Avg reader passages |
|---|---:|---:|---:|---:|
| 2Wiki | 1000 | 1000 | 0 | 2.577 |
| HotpotQA | 1000 | 1000 | 0 | 2.740 |
| MuSiQue | 1000 | 1000 | 0 | 3.622 |

## Main Results

| Dataset | DAEC-selective EM | DAEC-selective F1 | DAEC-selective R@5 | SetR-faithful EM | SetR-faithful F1 | SetR-faithful R@5 | dF1 DAEC-faithful | SetR + Fill@5 F1 | dF1 faithful-Fill@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.6420 | 0.7118 | 0.9410 | 0.6030 | 0.6746 | 0.8830 | +0.0372 | 0.6936 | -0.0190 |
| HotpotQA | 0.6200 | 0.7473 | 0.9605 | 0.6250 | 0.7435 | 0.9235 | +0.0038 | 0.7552 | -0.0117 |
| MuSiQue | 0.3530 | 0.4548 | 0.7469 | 0.3440 | 0.4467 | 0.6590 | +0.0081 | 0.4761 | -0.0294 |

## Paired Bootstrap

Query-paired percentile bootstrap, 10,000 resamples. Delta is DAEC-selective minus SetR-faithful.

| Dataset | Metric | DAEC-selective | SetR-faithful | Delta | 95% CI | P(delta > 0) | Excludes 0 |
|---|---|---:|---:|---:|---:|---:|---|
| 2Wiki | EM | 0.6420 | 0.6030 | +0.0390 | [0.0150, 0.0630] | 0.9995 | yes |
| 2Wiki | F1 | 0.7118 | 0.6746 | +0.0372 | [0.0154, 0.0595] | 0.9998 | yes |
| HotpotQA | EM | 0.6200 | 0.6250 | -0.0050 | [-0.0250, 0.0150] | 0.2910 | no |
| HotpotQA | F1 | 0.7473 | 0.7435 | +0.0038 | [-0.0150, 0.0226] | 0.6545 | no |
| MuSiQue | EM | 0.3530 | 0.3440 | +0.0090 | [-0.0170, 0.0360] | 0.7262 | no |
| MuSiQue | F1 | 0.4548 | 0.4467 | +0.0082 | [-0.0183, 0.0345] | 0.7207 | no |

## Interpretation

- The previous padded baseline was materially stronger than faithful selected-only SetR: Fill@5 adds +0.0117 to +0.0294 F1 depending on dataset.
- Correcting SetR to selected-only improves DAEC's relative position, but it does not create a three-dataset significant win. DAEC-selective significantly beats SetR-faithful on 2Wiki, while HotpotQA and MuSiQue remain statistical ties.
- The pre-specified 2Wiki 4-doc hard slice strengthens the composition story: DAEC-selective beats SetR-faithful by +0.1437 F1 with 95% CI [0.0879, 0.2010], and still beats the stronger `SetR + Fill@5` ablation by +0.0454 F1 with 95% CI [0.0028, 0.0894]. See `2wiki_4doc_hard_slice.md`.
- The under-selection mechanism is concentrated in deep slices: SetR-faithful selects fewer than 4 passages for 51.9% of 2Wiki 4-doc queries, while HotpotQA's 2-doc slice has only 3.2% under-selection. Conditioning on SetR selecting enough passages makes DAEC and SetR-faithful answer-tied, so the main mechanism claim should be sufficient structured coverage, not universal dominance after SetR has enough evidence. See `underselection_mechanism_slices.md`.
- Paper framing should therefore be: `SetR-faithful` is the primary SetR-style baseline aligned to adaptive context length; `SetR + Fill@5` is a budget-matched ablation, not the main reproduction.
- A bounded `SetR-Fill@10` ablation was added in `reports/setr_fill10_proprag_full1000_20260507/summary.md` to match the official repository's `convert_rankify.py` conversion-style `k=10` behavior. This should be framed as an official conversion-style sanity check, not as the paper-faithful SetR reader setting.
