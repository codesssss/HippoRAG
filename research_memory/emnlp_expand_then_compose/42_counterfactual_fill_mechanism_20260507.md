# DBEC Counterfactual Fill Mechanism Audit

Date: 2026-05-07
Status: New evidence after locked framing; this should trigger a framing re-review before paper writing.

## Question

Can we strengthen the mechanism claim from post-hoc conditional analysis to an intervention-style diagnostic?

The tested slice is deliberately narrow:

- 2Wiki: `gold_doc_count>=4 and SetR count-underselected`, N=122
- MuSiQue: `gold_doc_count>=3 and SetR count-underselected`, N=106

All conditions use the same fixed PropRAG pool and Qwen3-8B reader. The intervention starts from SetR-faithful's selected documents, then fills the reader context to 5 documents in three ways:

- `rank_fill5`: fill remaining slots by original retrieval rank.
- `dbec_repair_fill5`: add DBEC-selected non-SetR documents, then rank-fill if needed.
- `oracle_fill5`: add SetR-missing gold supports, then rank-fill if needed. This is diagnostic only.

## Main Results

| Dataset | Method | F1 | dF1 vs SetR | Support recall | Support complete |
|---|---:|---:|---:|---:|---:|
| 2Wiki | SetR-faithful | 0.6330 | - | 0.6270 | 0.0% |
| 2Wiki | rank_fill5 | 0.8388 | +0.2058 | 0.8770 | 54.9% |
| 2Wiki | dbec_repair_fill5 | 0.9098 | +0.2769 | 0.9344 | 74.6% |
| 2Wiki | DBEC-selective | 0.9180 | +0.2851 | 0.9242 | 70.5% |
| 2Wiki | oracle_fill5 | 0.9508 | +0.3179 | 0.9795 | 91.8% |
| MuSiQue | SetR-faithful | 0.2274 | - | 0.4481 | 0.0% |
| MuSiQue | rank_fill5 | 0.3392 | +0.1117 | 0.6368 | 20.8% |
| MuSiQue | dbec_repair_fill5 | 0.2869 | +0.0594 | 0.6124 | 15.1% |
| MuSiQue | DBEC-selective | 0.3093 | +0.0819 | 0.6179 | 19.8% |
| MuSiQue | oracle_fill5 | 0.5405 | +0.3131 | 0.9175 | 73.6% |

Key paired deltas:

- 2Wiki `dbec_repair_fill5 - rank_fill5`: +0.0710 F1, CI [+0.0082, +0.1366]
- 2Wiki `dbec_repair_fill5 - DBEC-selective`: -0.0082 F1, CI [-0.0574, +0.0410]
- MuSiQue `dbec_repair_fill5 - rank_fill5`: -0.0523 F1, CI [-0.1175, +0.0094]
- MuSiQue `oracle_fill5 - rank_fill5`: +0.2013 F1, CI [+0.1260, +0.2819]

## Interpretation

The mechanism is stronger than the old post-hoc slice analysis in one respect: missing support documents are clearly a real bottleneck. `oracle_fill5` produces large gains on both datasets, so SetR under-selection is not just a harmless formatting artifact.

But the intervention also weakens an over-strong DBEC-specific story. A large fraction of the gain comes from simply filling SetR's under-filled reader budget:

- On 2Wiki, rank fill recovers about 72% of the DBEC-vs-SetR F1 gap.
- On MuSiQue, rank fill exceeds DBEC-selective on this slice.

The clean DBEC-specific repair signal is dataset-dependent:

- 2Wiki: DBEC repair is load-bearing. `dbec_repair_fill5` beats `rank_fill5` significantly and nearly matches DBEC-selective.
- MuSiQue: DBEC repair is not load-bearing beyond rank fill. The oracle upper bound is high, but DBEC's repair documents do not recover it.

## Paper Implication

Do not frame the mechanism as:

> DBEC fixes under-selection because dependency binding identifies the missing support chain.

That is too strong across datasets.

Safer and truer framing:

> Prompt-only adaptive selection has a measurable under-filled-context failure mode on deep multi-hop queries. Missing support documents are a real reader bottleneck, as shown by oracle fill interventions. DBEC provides a structured repair that is strongly effective on 2Wiki and explains most of its 2Wiki hard-slice gain, but this repair is less reliable on MuSiQue, where simple rank fill is already a strong budget control.

This is still useful mechanism evidence, but it changes the paper from "dependency repair is generally proven" to "under-selection is diagnosed, support absence is intervention-validated, and DBEC is one structured repair with clear dataset-dependent limits."

## Reviewer-Facing Consequence

This result is actually good for honesty:

- It prevents overclaiming a fragile semantic mechanism.
- It gives a clean budget-control narrative.
- It identifies an explicit future-work target: better repair scoring on MuSiQue-like compositional queries where the oracle gap remains large.

But it means DBEC cannot rely on the MuSiQue under-selection slice as a DBEC-specific mechanism win. MuSiQue should be used as evidence that support absence is real and that current DBEC repair is incomplete, not as evidence that DBEC repair dominates rank fill.

## Artifacts

- Script: `scripts/build_counterfactual_fill_pools.py`
- Script: `scripts/analyze_counterfactual_fill_mechanism.py`
- Eval outputs: `reports/counterfactual_fill_mechanism_20260507/*.eval.json`
- Summary: `reports/counterfactual_fill_mechanism_20260507/summary.md`
- Per-query rows: `reports/counterfactual_fill_mechanism_20260507/counterfactual_rows.csv`
- Paired CI table: `reports/counterfactual_fill_mechanism_20260507/paired_ci.csv`
