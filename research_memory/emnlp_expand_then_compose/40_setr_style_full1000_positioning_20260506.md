# SetR-Style Full1000 Baseline Positioning - 2026-05-06

## Decision

The SetR-style baseline is already implemented and full1000-complete. Do not
reimplement or relaunch a new requirements-aware LLM selector before using this
result.

Paper label:

```text
SetR-style IRI selector (local) + rank-order fallback
```

Do **not** label it as an official SetR reproduction.

## Canonical Artifacts

Implementation:

- `scripts/export_setr_inputs.py`
- `scripts/run_setr_style_selector.py`
- `scripts/apply_setr_selection_to_pool.py`
- `scripts/run_setr_windowed_selector.py`
- `tests/test_setr_adapter.py`

Launcher and outputs:

- launcher: `run_logs/launch_setr_full1000_20260503.sh`
- run logs: `run_logs/setr_full1000_20260503/`
- eval reports: `reports/setr_full1000_20260503/`
- launcher status: `run_logs/setr_full1000_20260503/launcher.status`

Completion:

```text
[DONE] setr_full1000 2026-05-04T02:04:10+08:00
```

## Protocol

- Same 1000-query subsets and same Qwen3-8B reader path as DAEC full1000.
- Pools: Dense, HippoRAG, PropRAG pool100.
- Variants:
  - `setr_k20_doc768`: direct SetR-style IRI prompt over top20, 768 chars/doc.
  - `setr_k100_doc160`: direct prompt over top100, 160 chars/doc.
  - `setr_windowed_k50_doc768`: two-stage windowed top50 adaptation, 768 chars/doc.
- Prompt mechanism follows SetR's `selection_IRI` shape:
  1. list information requirements;
  2. find passages for each requirement;
  3. output `### Final Selection: [id] [id] ...`.
- The prompt allows an unlimited number of selected passages. The adapter
  frontloads parsed selections, then fills the remaining reader top5 slots from
  original rank order. This fallback is part of the local protocol and must be
  stated in paper tables.

## Main PropRAG Comparison

DAEC row is current `DAEC-LLM + CTL` PropRAG full1000:
`run_logs/daec_llm_wiki_title_proprag_full1000_20260503/`.

DAEC-selective row is the frozen query-level title-uniqueness gate:
`bind_conf_title_unique >= 0.88` uses binding, otherwise abstains to
`nobinding`. Fresh full1000 results are in
`run_logs/daec_selective_titleuniq_proprag_full1000_20260506/` and summarized
in `reports/daec_selective_binding_phase1_20260506/phase1_report.md`.

| Dataset | DAEC EM/F1/R@5 | SetR-style k20 EM/F1/R@5 | dF1 vs DAEC | SetR-windowed k50 EM/F1/R@5 | dF1 vs DAEC |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 0.642 / 0.712 / 0.941 | 0.624 / 0.694 / 0.942 | -0.018 | 0.620 / 0.692 / 0.941 | -0.019 |
| HotpotQA | 0.620 / 0.747 / 0.961 | 0.629 / 0.755 / 0.973 | +0.008 | 0.630 / 0.755 / 0.974 | +0.007 |
| MuSiQue | 0.337 / 0.436 / 0.727 | 0.377 / 0.476 / 0.755 | +0.040 | 0.365 / 0.467 / 0.746 | +0.032 |

Selective-binding comparison against the strongest SetR-style k20 row:

| Dataset | DAEC-selective EM/F1/R@5 | dF1 vs DAEC | dF1 vs SetR-style k20 | dR@5 vs SetR-style k20 |
|---|---:|---:|---:|---:|
| 2Wiki | 0.642 / 0.7118 / 0.9410 | +0.0000 | +0.0182 | -0.0013 |
| HotpotQA | 0.620 / 0.7473 / 0.9605 | +0.0000 | -0.0079 | -0.0125 |
| MuSiQue | 0.353 / 0.4548 / 0.7469 | +0.0189 | -0.0213 | -0.0078 |

Selective binding does not overturn the SetR-style result. It preserves DAEC's
2Wiki advantage, leaves HotpotQA unchanged, and narrows the MuSiQue F1 gap from
`-0.0402` to `-0.0213`. For support recall, use the unified title-multiset
audit below rather than mixing stored pipeline R@5 fields.

Paired bootstrap CI, answer F1:

| Dataset | Comparison | dF1 | 95% CI | Interpretation |
|---|---|---:|---:|---|
| 2Wiki | DAEC-selective - SetR-style k20 | +0.0182 | [-0.0019, +0.0379] | positive but not 95% significant |
| HotpotQA | DAEC-selective - SetR-style k20 | -0.0079 | [-0.0225, +0.0066] | statistically tied |
| MuSiQue | DAEC-selective - SetR-style k20 | -0.0212 | [-0.0443, +0.0018] | SetR-style positive trend, not 95% significant on F1 |
| MuSiQue | DAEC-selective - DAEC | +0.0189 | [+0.0076, +0.0309] | selective gate significantly repairs base DAEC |

Paper consequence: do not claim DAEC-selective significantly beats SetR-style.
The defensible claim is that DAEC-selective is a structured, interpretable
composer that is competitive with a strong local LLM set selector, with a clear
binding-abstention repair on MuSiQue and a large binding ablation gap on 2Wiki.

Unified title-multiset support R@5, DAEC-selective vs SetR-style k20:

| Dataset | DAEC-selective R@5 | SetR-style R@5 | dR@5 | 95% CI | Interpretation |
|---|---:|---:|---:|---:|---|
| 2Wiki | 0.9410 | 0.9425 | -0.0015 | [-0.0118, +0.0088] | tied |
| HotpotQA | 0.9625 | 0.9735 | -0.0110 | [-0.0200, -0.0020] | SetR higher support |
| MuSiQue | 0.7745 | 0.7837 | -0.0092 | [-0.0230, +0.0044] | tied |

This replaces earlier mixed-field support interpretations. Under one unified
title-based support metric, SetR-style remains tied or better on support, while
DAEC-selective's defensible edge is over Top5/IRCoT-style/LLM-direct and its
explicit binding ablation story.

Gold-support-count hard slice against SetR-style k20:

| Dataset | Slice | N | dF1 | F1 95% CI | dR@5 | R@5 95% CI | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| 2Wiki | `gold_doc_count>=3` (= 4-doc) | 235 | +0.0454 | [+0.0014, +0.0894] | +0.0106 | [-0.0106, +0.0319] | DAEC-selective wins answer F1 on the deep subset; support tied. |
| HotpotQA | `gold_doc_count>=3` | 0 | -- | -- | -- | -- | No deep-support slice; shallow/saturated contrast only. |
| MuSiQue | `gold_doc_count>=3` | 482 | -0.0083 | [-0.0431, +0.0260] | -0.0003 | [-0.0201, +0.0197] | Statistically tied; SetR-style not overturned. |

Hard-slice artifacts:

- script: `scripts/analyze_reviewer_baseline_hard_slices.py`
- report:
  `reports/reviewer_baseline_ci_20260506/hard_slice_gold_doc_count.md`

This is the strongest legal slice defense: 2Wiki's 4-document subset supports a
DAEC-selective deep-composition win over SetR-style, while HotpotQA has no such
subset and MuSiQue remains mixed/tied. Do not use it to claim a general
SetR-style defeat.

## Interpretation

SetR-style is a strong baseline, not a negative control.

Implications:

- DAEC cannot claim that explicit noisy-OR composition generally beats LLM
  set selection. It does not on full PropRAG HotpotQA/MuSiQue, although
  DAEC-selective significantly wins the 2Wiki 4-document hard slice.
- Generic "RAG set selection" is not a safe novelty claim. SetR owns much of
  that framing.
- DAEC should be positioned as a structured, interpretable, binding-aware
  fixed-pool composer rather than as a universally stronger set selector.
- The current-version `nobinding` refresh is now complete and confirms that
  binding is load-bearing on 2Wiki and still positive on HotpotQA, while
  MuSiQue benefits from abstention-aware binding.
- DAEC-selective should be framed as a robustness patch for ambiguous
  dependency resolution, not as a universal verifier or a general SetR-style
  replacement.
- HotpotQA support should be acknowledged directly: under unified title-based
  support, SetR-style is significantly higher on HotpotQA. DAEC's HotpotQA
  value is structured interpretability and consistency, not retrieval coverage.

Safe paper wording:

```text
SetR-style IRI is a strong prompt-based set selector that directly asks an LLM
to identify information requirements and choose passages. DAEC instead exposes
the dependency structure through a requirement-by-binding-by-document support
tensor and optimizes a transparent noisy-OR coverage objective. We therefore
compare against SetR-style IRI as a controlled local baseline, while treating
dependency binding as the key mechanism that must be ablated.
```

## Selection Statistics

PropRAG selection rows:

| Dataset | Variant | Rows | Parse Success | Mean Selected | Mean Fallback in Top5 | Selected < 5 |
|---|---|---:|---:|---:|---:|---:|
| 2Wiki | `setr_k20_doc768` | 1000 | 1000 | 2.58 | 2.48 | 959 |
| HotpotQA | `setr_k20_doc768` | 1000 | 1000 | 2.74 | 2.43 | 912 |
| MuSiQue | `setr_k20_doc768` | 1000 | 1000 | 3.62 | 1.70 | 771 |
| 2Wiki | `setr_windowed_k50_doc768` | 1000 | 996 | 2.23 | 2.77 | 993 |
| HotpotQA | `setr_windowed_k50_doc768` | 1000 | 999 | 2.28 | 2.73 | 957 |
| MuSiQue | `setr_windowed_k50_doc768` | 1000 | 997 | 2.79 | 2.23 | 887 |

This is why the table label must include rank-order fallback. The evaluated
reader top5 is not always five LLM-selected documents.

For the main PropRAG `setr_k20_doc768` baseline, the local SetR-style selector
is shallow on 2Wiki/HotpotQA:

| Dataset | Mean Selected | Mean Fallback in Top5 | Selected Positions from Original Top5 | pos0 Selected | pos1 Selected |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 2.58 | 2.48 | 86.5% | 88.5% | 72.3% |
| HotpotQA | 2.74 | 2.43 | 80.7% | 88.5% | 78.7% |
| MuSiQue | 3.62 | 1.70 | 57.2% | 75.6% | 58.0% |

Interpretation:

- On 2Wiki and HotpotQA, SetR-style IRI mostly confirms the retriever's top
  anchors and relies on rank-order fallback to complete the reader top5.
- On MuSiQue, it performs more meaningful deeper-position selection; only
  57.2% of selected positions come from the original top5.
- Therefore this baseline should be described as a hybrid prompt selector:
  `LLM-selected prefix + retriever-rank fallback`, not as a pure five-document
  set selector.

Metric caveat:

- R@5 values can differ depending on whether they are read from
  `overall_recomputed.Recall@5`, `setwise_selector_qa.selector_retrieval_metrics`,
  source-pool payload recall, or a custom title-based recomputation.
- Do not mix those fields in the same table. The canonical main comparison
  should use DAEC's selector metrics and SetR-style's evaluated selected-pool
  metrics under one explicitly named support-recall definition.

## Submission Action

Update main comparison candidates:

- Top5 baseline
- DAEC-LLM + CTL or DAEC-selective, depending on method-space budget
- SetR-style IRI local baseline
- IRCoT-style local baseline
- LLM-direct-title/snippet as simpler LLM selector controls

Main ablation priority is now satisfied for PropRAG full1000:

```text
current-version PropRAG full1000 nobinding refresh
```

Reason: binding is the cleanest mechanism-level distinction from SetR-style
information-requirement selection. The refreshed result supports the claim on
2Wiki/HotpotQA and exposes the MuSiQue over-constraint failure mode that
DAEC-selective addresses.

Escalation:

```text
SetR-style remains a strong baseline. The paper should claim mechanism
separation and robustness, not general superiority over LLM set selection.
```

## Cost Instrumentation Patch

Implemented after the full1000 run, 2026-05-06:

- `scripts/run_setr_style_selector.py` now stores per-call:
  - `usage.prompt_tokens`;
  - `usage.completion_tokens`;
  - `usage.total_tokens`;
  - `usage.finish_reason`;
  - `latency_s`;
  - `prompt_chars`;
  - `completion_chars`.
- `scripts/run_setr_windowed_selector.py` now stores stage-level fields and
  top-level aggregates:
  - `selector_call_count`;
  - `usage`;
  - `latency_s`;
  - `prompt_chars`;
  - `completion_chars`.

Existing `setr_full1000_20260503` outputs predate this patch, so exact API
token usage cannot be recovered for those rows. Use prompt-character estimates
for historical cost tables, and use recorded `usage` fields for new SetR-style
runs.

Cost-table unit caveat:

- In `reports/reviewer_baseline_costs_20260506/cost_table.md`,
  `prompt_tokens_per_q` is already aggregated per query. Do not multiply DAEC's
  `prompt_tokens_per_q` by its call count.
- DAEC is higher-call and slower than SetR-style k20 in these artifacts, while
  its measured binding prompt tokens/query are lower than SetR's reconstructed
  top20 prompt estimate. The defensible efficiency claim is not "cheaper than
  SetR"; it is "no extra retrieval and explicit/auditable structure."
