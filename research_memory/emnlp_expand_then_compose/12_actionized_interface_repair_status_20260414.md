# Actionized Interface Repair Status

Last updated: 2026-04-14

Successor branch note:

- This note freezes the `noisyor` stop decision.
- A later reviewer-driven successor controller branch is recorded separately in:
  - `research_memory/emnlp_expand_then_compose/13_tiered_witness_swap_impl_20260414.md`

## Scope

This note records the current status of the `action_swap` line after three consecutive stages:

1. non-learned actionization of the `bridge_append` assemble step
2. one-swap oracle decomposition under current vs relaxed legality
3. zero-shot `noisyor` controller audit

The goal is not to propose another new controller.
The goal is to freeze what is now established, what failed, and what should stop.

## Canonical files

- Main patch point and implementation:
  - `scripts/eval_causal_qwen3.py`
- Actionized selector unit tests:
  - `tests/test_setwise_selector.py`
- NoisyOR smoke runner:
  - `run_logs/action_swap_noisyor_smoke_20260414.sh`
- NoisyOR smoke summary:
  - `run_logs/action_swap_noisyor_smoke_20260414.summary.md`
- Oracle ceiling runner:
  - `scripts/run_action_swap_oracle_ceiling.py`
- Relaxed oracle reports:
  - `run_logs/musique_action_swap_oracle_relaxed_20260413.json`
  - `run_logs/musique_action_swap_oracle_relaxed_20260413.md`
  - `run_logs/2wiki_action_swap_oracle_relaxed_20260413.json`
  - `run_logs/2wiki_action_swap_oracle_relaxed_20260413.md`
- Zero-shot audit runner:
  - `scripts/audit_action_swap_noisyor.py`
- Zero-shot audit outputs:
  - `run_logs/musique_action_swap_noisyor_audit_20260414.json`
  - `run_logs/musique_action_swap_noisyor_audit_20260414.md`
  - `run_logs/musique_action_swap_noisyor_audit_20260414.csv`
  - `run_logs/2wiki_action_swap_noisyor_audit_20260414.json`
  - `run_logs/2wiki_action_swap_noisyor_audit_20260414.md`
  - `run_logs/2wiki_action_swap_noisyor_audit_20260414.csv`

## Frozen implementation definition

The implemented `action_swap_v0` family changes only the final `candidate_set -> final_front_positions` step inside the existing `bridge_append` branch.

Shared semantics:

- `S0` is the baseline-only CE scaffold
- action space is `keep ∪ {d ↔ i}`
- candidates come from bridge-appended docs only
- incumbents are chosen from the current scaffold bottom slots
- each query executes at most one swap
- swap is slot-preserving; reader order is not re-sorted after the action

This matters because it changes the decision unit from:

- `append docs, then union rerank`

to:

- `evaluate scaffold-conditioned actions, then rewrite the final front`

That is the concrete code-level repair of the previously identified `Expand -> Assemble` interface mismatch.

## Stage 1: Non-learned actionization result

Reference smoke summary:
- `run_logs/action_swap_noisyor_smoke_20260414.summary.md`

At `limit=100`, `qa_top_k=5`:

### MuSiQue

- `bridge_append_plus_ce`: `0.3200 / 0.3810`
- `action_swap_v0_dryrun`: `0.3200 / 0.3900`
- `action_swap_v0_judge`: `0.3200 / 0.3798`

### 2Wiki

- `bridge_append_plus_ce`: `0.4300 / 0.4901`
- `action_swap_v0_dryrun`: `0.4300 / 0.4658`
- `action_swap_v0_judge`: `0.4400 / 0.4805`

Interpretation:

- simply changing the execution unit from union rerank to single slot-preserving action already changes QA outcomes
- on MuSiQue, the dryrun gain suggests collateral harm from global union rerank
- on 2Wiki, the judge result suggests the main gain is selective drift filtering rather than actionization alone

This is positive evidence for the `actionized interface repair` diagnosis.
It is not yet evidence that a new controller should replace the main paper line.

## Stage 2: Oracle one-swap decomposition

Reference files:

- `run_logs/musique_action_swap_oracle_current_20260413.md`
- `run_logs/musique_action_swap_oracle_relaxed_20260413.md`
- `run_logs/2wiki_action_swap_oracle_current_20260413.md`
- `run_logs/2wiki_action_swap_oracle_relaxed_20260413.md`

The main decomposition is:

- `legality miss = relaxed oracle - current oracle`
- `policy miss = current oracle - actual policy`

### MuSiQue

- current oracle `ΔF1`: `+0.0211`
- relaxed oracle `ΔF1`: `+0.0961`
- legality miss: `+0.0750`

Interpretation:

- current legality is too hard
- many real positive swaps are excluded before policy selection even starts
- the current legality proxy behaves like pointwise CE dominance, not set utility

### 2Wiki

- current oracle `ΔF1`: `+0.0407`
- relaxed oracle `ΔF1`: `+0.0489`
- legality miss: `+0.0082`

Interpretation:

- legality is not the main bottleneck
- the main issue is action selection under contextual drift among already legal actions

Cross-dataset takeaway:

> MuSiQue is primarily an action-space recall problem.
> 2Wiki is primarily an action selection under drift problem.

This is stronger than a generic "judge is dataset-specific" observation because it decomposes the interface mismatch into two measurable error sources.

## Stage 3: Zero-shot NoisyOR controller result

The tested controller family was:

- `action_swap_noisyor_flat`
- `action_swap_noisyor_dep`

Design goal:

- keep the actionized `keep/swap` skeleton
- remove online LLM judge
- use query-side dependency graph plus local support scoring as a non-learned controller

Smoke result:

### MuSiQue

- `action_swap_noisyor_flat`: `0.3100 / 0.3698`
- `action_swap_noisyor_dep`: `0.3100 / 0.3698`
- executed swaps: `0`

### 2Wiki

- `action_swap_noisyor_flat`: `0.4200 / 0.4553`
- `action_swap_noisyor_dep`: `0.4200 / 0.4553`
- executed swaps: `0`

Immediate interpretation:

- the controller died before the dependency graph could matter
- the failure is not "dep graph too weak"
- the failure is that the support-to-utility surrogate never pushed actions over the decision margin

## Action-level audit result

Reference files:

- `run_logs/musique_action_swap_noisyor_audit_20260414.md`
- `run_logs/2wiki_action_swap_noisyor_audit_20260414.md`

The audit was intentionally local:

- no new reader runs
- no new controller variants
- only offline action scoring over:
  - relaxed oracle-positive swaps
  - dryrun executed swaps
  - judge executed swaps
  - sampled oracle-negative swaps

### Question A: are oracle-positive actions merely below margin, or mostly nonpositive?

Answer: mostly nonpositive.

MuSiQue:

- oracle-positive actions audited: `148`
- `flat_top2 <= 0`: `145 / 148`
- `dep_top2 <= 0`: `145 / 148`
- `flat_max <= 0`: `148 / 148`
- `flat_top2 <= 0.05`: `148 / 148`

2Wiki:

- oracle-positive actions audited: `55`
- `flat_top2 <= 0`: `55 / 55`
- `dep_top2 <= 0`: `55 / 55`
- `flat_max <= 0`: `55 / 55`
- `flat_top2 <= 0.05`: `55 / 55`

Interpretation:

- this is not just a margin calibration problem
- the current zero-shot utility surrogate largely fails to assign positive value even to true positive swaps

### Question B: do dryrun or judge positive actions get lifted by the surrogate?

Answer: almost never.

MuSiQue:

- dryrun positive executed swaps: `7`
- `flat_top2 > 0`: `1 / 7`

2Wiki:

- dryrun positive executed swaps: `11`
- `flat_top2 > 0`: `0 / 11`

Interpretation:

- the surrogate is not merely conservative
- it also fails to lift already validated positive actions

### Question C: does dependency gating materially change ranking?

Answer: almost not at all.

MuSiQue:

- dep-triggered queries: `30`
- flat vs dep ranking changed: `2`
- flat vs dep top-1 action changed: `1`

2Wiki:

- dep-triggered queries: `21`
- flat vs dep ranking changed: `1`
- flat vs dep top-1 action changed: `0`

Interpretation:

- dependency graph is not currently an effective controller component
- under the present support surrogate it behaves more like a mild numeric perturbation than a meaningful ranking mechanism

## What is now established

### 1. The interface mismatch diagnosis survives

The failed `noisyor` controller does not invalidate the earlier diagnosis.

What survives:

- `Expand` proposes structurally novel documents
- `Assemble` really needs scaffold-conditional action utility
- changing the execution unit to action-level decisions already changes outcomes
- oracle decomposition still shows measurable legality miss and policy miss

What failed:

- this particular zero-shot support-to-utility instantiation

### 2. Actionization is useful as analysis evidence

`action_swap_v0_dryrun` and `action_swap_v0_judge` should be treated as positive interventions:

- they show that `keep/swap` is a meaningful execution lens
- they help explain the difference between union-rerank collateral replacement and contextual drift filtering

They should not yet replace the paper mainline.

### 3. The current NoisyOR controller line should stop

The stop condition is already met:

- oracle-positive actions are mostly scored as nonpositive
- dryrun and judge positives are not lifted
- dependency graph barely changes rankings

So this controller should be written up as a useful failed instance, not rescued with more threshold tuning or more controller variants.

## Research consequence

The paper-facing main method remains:

- `bridge-aware Expand + CE-centered Assemble`

The actionized line should be written as analysis / intervention evidence:

1. actionized repair changes outcomes even without training
2. oracle decomposition reveals dataset-specific legality-vs-policy bottlenecks
3. a zero-shot dependency-aware utility controller fails because the support surrogate does not recover positive swap preference

That is a coherent negative result, not dead work.

## Engineering consequence

The current route should stay narrow.

Do:

- preserve the `action_swap_v0` implementation and tests
- preserve the oracle decomposition reports
- preserve the `noisyor` audit outputs as canonical evidence

Do not do next from this note:

- do not tune `epsilon`
- do not add more zero-shot controller variants
- do not expand the reader experiment surface for this line
- do not promote `noisyor_dep` into the main method

## Route from here

The route is now:

1. keep the paper mainline frozen on `bridge-aware Expand + CE-centered Assemble`
2. use `action_swap_v0_*` and oracle decomposition as analysis evidence for interface mismatch
3. treat `noisyor` as a failed zero-shot controller instance
4. only revisit controller learning if there is a separate reason to pursue scaffold-conditional utility as a future project

The route is not:

- keep iterating on this exact non-learned controller family
