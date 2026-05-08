# DAEC/DBEC Depth x Gate-Resolvability Conditional Analysis - 2026-05-08

## Purpose

This note deposits the conditional mechanism analysis requested after the paper-story review.

The goal is to test whether the paper should claim:

```text
Explicit dependency binding helps conditionally, especially when the query has high support depth and the intermediate binding is resolvable.
```

This analysis is intentionally conservative:

- it reuses existing PropRAG full1000 outputs only;
- it makes no new LLM calls;
- it uses `gold_doc_count` as **support depth**, not manually annotated dependency depth;
- it uses DBEC-IG's pre-selection title-uniqueness gate as a **binding-resolvability proxy**, not an oracle identifiability label.

## Canonical Artifacts

Primary report:

- `reports/daec_depth_resolvability_20260508/summary.md`
- `reports/daec_depth_resolvability_20260508/slice_profile.csv`
- `reports/daec_depth_resolvability_20260508/method_summary.csv`
- `reports/daec_depth_resolvability_20260508/paired_ci.csv`
- `reports/daec_depth_resolvability_20260508/summary.json`

Implementation:

- `scripts/analyze_daec_depth_resolvability.py`
- `tests/test_daec_depth_resolvability.py`

Inputs:

- DBEC-IG / DAEC-selective PropRAG full1000:
  - `run_logs/daec_selective_titleuniq_proprag_full1000_20260506/`
- Ungated DBEC / DAEC PropRAG full1000:
  - `run_logs/daec_llm_wiki_title_proprag_full1000_20260503/`
- DBEC-nobinding PropRAG full1000:
  - `run_logs/daec_nobinding_proprag_full1000_20260506/`
- SetR-style k20:
  - `reports/setr_full1000_20260503/`
- SetR-faithful:
  - `reports/setr_faithful_proprag_full1000_20260507/`
- RankGPT-style sliding reader:
  - `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/`

Protocol:

- Fixed PropRAG pool100.
- Qwen3-8B `/no_think` controlled substrate.
- Reader `qa_top_k=5`, `qa_doc_max_chars=2048`.
- Query-paired percentile bootstrap with `10,000` resamples.
- `R5_TITLE` recomputed uniformly from final reader top-5 titles with title-multiset support recall.

## Main Conditional F1 Results

| Dataset | Slice | N | DBEC-IG | Nobinding | SetR-faithful | RankGPT-style | DBEC-IG - Nobind | 95% CI | DBEC-IG - SetR | 95% CI | DBEC-IG - RankGPT | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | all | 1000 | 0.7118 | 0.6120 | 0.6746 | 0.6659 | +0.0998 | [+0.0787, +0.1211] | +0.0372 | [+0.0155, +0.0596] | +0.0460 | [+0.0245, +0.0680] |
| 2Wiki | gate=bind | 698 | 0.6853 | 0.5422 | 0.6240 | 0.6194 | +0.1430 | [+0.1138, +0.1729] | +0.0613 | [+0.0342, +0.0888] | +0.0658 | [+0.0390, +0.0926] |
| 2Wiki | support_depth>=3, gate=bind | 233 | 0.9142 | 0.7811 | 0.7778 | 0.8598 | +0.1331 | [+0.0815, +0.1846] | +0.1364 | [+0.0825, +0.1931] | +0.0544 | [+0.0114, +0.0987] |
| HotpotQA | all | 1000 | 0.7473 | 0.7387 | 0.7435 | 0.6845 | +0.0086 | [-0.0045, +0.0213] | +0.0038 | [-0.0148, +0.0229] | +0.0628 | [+0.0424, +0.0846] |
| HotpotQA | gate=bind | 657 | 0.7630 | 0.7500 | 0.7590 | 0.6815 | +0.0130 | [-0.0065, +0.0320] | +0.0041 | [-0.0181, +0.0262] | +0.0815 | [+0.0544, +0.1100] |
| MuSiQue | all | 1000 | 0.4548 | 0.4458 | 0.4467 | 0.4093 | +0.0090 | [-0.0053, +0.0238] | +0.0082 | [-0.0185, +0.0343] | +0.0455 | [+0.0187, +0.0728] |
| MuSiQue | gate=bind | 371 | 0.4878 | 0.4636 | 0.5030 | 0.4649 | +0.0242 | [-0.0146, +0.0647] | -0.0152 | [-0.0581, +0.0281] | +0.0230 | [-0.0220, +0.0674] |
| MuSiQue | support_depth>=3, gate=bind | 159 | 0.3432 | 0.3332 | 0.3406 | 0.3419 | +0.0100 | [-0.0506, +0.0725] | +0.0026 | [-0.0661, +0.0707] | +0.0013 | [-0.0626, +0.0675] |
| MuSiQue | support_depth>=4, gate=bind | 52 | 0.1912 | 0.2588 | 0.2788 | 0.2275 | -0.0676 | [-0.1699, +0.0283] | -0.0876 | [-0.1955, +0.0160] | -0.0362 | [-0.1285, +0.0555] |
| MuSiQue | support_depth>=4, gate=abstain | 114 | 0.3140 | 0.3140 | 0.2465 | 0.2310 | +0.0000 | [+0.0000, +0.0000] | +0.0675 | [-0.0147, +0.1511] | +0.0830 | [+0.0060, +0.1621] |

## Interpretation

This is a useful experiment, but it should be written as a conditional mechanism result, not as a universal-depth result.

Supported:

- On 2Wiki high-support-depth gate-bind queries, explicit binding is clearly load-bearing.
- The 2Wiki hard-slice result is robust against DBEC-nobinding, SetR-faithful, and RankGPT-style sliding.
- Gate abstention exactly falls back to DBEC-nobinding on all three datasets: DBEC-IG - DBEC-nobinding is `0.0000` F1 with zero-width CI on `gate=abstain` slices.

Not supported:

- "Deep query" alone is not sufficient.
- "Gate-bind" alone is not sufficient.
- MuSiQue high-support-depth gate-bind does not replicate the 2Wiki advantage; DBEC-IG is lower than DBEC-nobinding and SetR-faithful on that small slice, with wide CIs.

Paper-safe framing:

```text
Explicit binding is strongly load-bearing on 2Wiki deep, gate-resolvable dependencies, but its value is conditional. MuSiQue shows the boundary: high support depth alone does not guarantee benefit when referents are ambiguous or the fixed 5-document reader budget is insufficient.
```

Use this to strengthen Section 5.3:

- Put 2Wiki deep gate-bind as the positive mechanism slice.
- Put MuSiQue deep gate-bind as the honest boundary slice.
- Keep the main claim as "conditional advantage", not "dependency binding universally solves deep multi-hop evidence selection".

## Claim Boundary

Allowed:

```text
DBEC-IG's explicit binding is load-bearing on the 2Wiki high-support-depth gate-bind slice: +0.133 F1 over DBEC-nobinding, +0.136 F1 over SetR-faithful, and +0.054 F1 over RankGPT-style sliding.
```

Allowed:

```text
The identifiability gate is a conservative fallback: on gate-abstain slices, DBEC-IG exactly matches DBEC-nobinding.
```

Not allowed:

```text
Explicit binding is universally better for all deep multi-hop queries.
```

Not allowed:

```text
The title-uniqueness gate is a ground-truth identifiability label.
```

Decision:

- Keep this result as a Section 5.3 mechanism/boundary table.
- Do not run another pilot unless the paper draft reveals a specific reviewer-facing gap.
