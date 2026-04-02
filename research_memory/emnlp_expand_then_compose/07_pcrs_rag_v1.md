# PCRS-RAG V1 Branch Memo

Last updated: 2026-04-02

## Status

This note documents the new parallel branch around `requirement_beam`.

Current status:
- implementation landed
- tests for the new selector path passed locally through a Python harness
- no benchmark result files exist yet

This is not the current paper-facing mainline.

The current stable simple line remains:
- `bridge_beam + set_closure + pathcore_guard + reserve3 + dedup`

`PCRS-RAG V1` is a separate branch that tries to fix the current objective bias on `MuSiQue` without changing the HippoRAG retrieval backbone.

## Why This Branch Exists

Current diagnosis:
- the simple setwise objective is strongest on `2Wiki`
- it is much weaker on `MuSiQue`, especially the `3-doc` slice
- widening proposal shortlists did not materially change selections, which suggests the bottleneck is not only search width

Working hypothesis:
- the current objective is still too seed-centric and too tied to explicit entity bridge closure
- `MuSiQue` often needs intermediate semantic completion rather than only explicit seed-to-query entity reachability

So this branch reformulates the selector objective from:
- path-centric bridge closure

to:
- positive requirement support
- counterfactual leakage suppression
- Pareto-style set selection

## Branch Boundary

Branch:
- `feature/pcrs-rag-v1`

Implementation commits:
- `9407db3` `Add requirement beam selector core`
- `3c37b3e` `Add requirement beam cache and training scripts`

Files:
- `scripts/eval_causal_qwen3.py`
- `scripts/requirement_beam_utils.py`
- `scripts/build_requirement_cache.py`
- `scripts/train_requirement_setwise.py`
- `tests/test_setwise_selector.py`

Important boundary:
- this branch does not replace the current simple selector by default
- promotion requires benchmark evidence

## Method Summary

### 1. Offline Requirement Cache

Each query gets an offline cache entry with:
- positive requirements
- counterfactual requirement sets
- per-document requirement coverage scores

Current requirement types:
- `anchor`
- `bridge`
- `decision`

Current cache design is heuristic and intentionally lightweight:
- no dependency graph edges yet
- no online LLM judge in the selector loop
- no iterative repair loop

### 2. Set-Level Objective

The selector scores a candidate evidence set with two main axes:

1. `support_completeness`
   - smooth aggregation over `anchor / bridge / decision` support
2. `counterfactual_leakage`
   - soft worst-case support for alternative requirement sets

Derived diagnostics:
- `utility_margin = support_completeness - counterfactual_leakage`
- `utopia_distance`

### 3. Selector

New selector:
- `--setwise_selector requirement_beam`

Search structure:
- keep the widened deep pool
- preserve prefix anchors / reserves as before
- expand candidate states
- prune by Pareto dominance
- pick the final state by `utopia_distance`

Important nuance:
- proposal generation still reuses the old cheap bridge-aware candidate scorer as a pruning prior
- the new branch changes the state objective first, not the whole proposal layer

### 4. Optional Learned Prior

The branch also adds a lightweight matcher:
- trained offline from requirement-state deltas
- used only as an extra proposal prior in `learned` mode

It does not replace the set-level objective.

That is deliberate:
- first validate the new objective in oracle mode
- only then see whether a small learned prior helps proposal ordering

## New CLI Surface

### Build cache

Script:
- `scripts/build_requirement_cache.py`

Purpose:
- build the offline query-level requirement cache used by `requirement_beam`

### Run selector in oracle mode

Main flags:
- `--setwise_selector requirement_beam`
- `--setwise_requirement_cache_path <cache.json>`
- `--setwise_requirement_mode oracle`

### Train lightweight matcher

Script:
- `scripts/train_requirement_setwise.py`

Purpose:
- train a lightweight requirement matcher from cached requirement supervision

### Run selector in learned mode

Main flags:
- `--setwise_selector requirement_beam`
- `--setwise_requirement_cache_path <cache.json>`
- `--setwise_requirement_mode learned`
- `--setwise_requirement_model_path <model.joblib>`

## Validation Gates

This branch should be evaluated in order.

### Gate 1: Cache quality

Must check:
- positive requirement non-empty rate
- counterfactual set non-empty rate
- annotation pool sanity
- top-20 vs top-50 sensitivity

If cache quality is poor, stop here.

### Gate 2: Oracle selector

Run:
- `requirement_beam` with offline cache only

Goal:
- test whether the new objective has useful signal before any learned matcher is introduced

If oracle mode does not help, do not promote the branch.

### Gate 3: Learned matcher

Only after oracle mode is at least directionally useful:
- train the lightweight matcher
- rerun `requirement_beam` in learned mode

Goal:
- test whether proposal ordering improves without changing the state objective

## Risks

Main risks:
- cache quality is the largest risk
- requirement typing is still coarse
- proposal and state objectives are still not perfectly unified
- counterfactual sets are still heuristic rather than model-generated

This means the branch is currently:
- high-information
- medium engineering cost
- not paper-ready by default

## Promotion Rule

Do not replace the current simple line unless all of the following happen:

1. Oracle `requirement_beam` shows real value on at least one meaningful hard slice
2. The gain is not just a `2Wiki`-only artifact
3. The learned matcher does not destabilize the branch
4. The resulting story stays explainable enough for the paper

Until then, the correct project state is:
- paper mainline: current simple `pathcore_guard` stack
- parallel branch: `PCRS-RAG V1`

## Recommended Next Step

Run in this order:

1. build a first cache with `annotation_pool_k = 50`
2. run `requirement_beam` in oracle mode on a small `2Wiki` and `MuSiQue` slice
3. inspect mechanism metrics before EM/F1 only:
   - support completeness
   - counterfactual leakage
   - frontier size
4. train the lightweight matcher only if the oracle branch shows real signal
