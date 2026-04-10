# Coverage Probe And Control Status

Last updated: 2026-04-09

## Scope

This note consolidates the current state of the `coverage` line, the `width-matched CE control` line, and the active full-scale queue status.

It is meant to answer three practical questions:

- what has already been established with enough confidence to use in paper narrative
- which results should currently be treated as invalid or provisional
- what the ongoing full-scale control is actually trying to decide

## Canonical files

- Coverage append poisoning summary: `run_logs/coverage_frozen_atoms_musique_20260408.summary.md`
- Coverage anchored-v3 summary: `run_logs/coverage_anchored_musique_k5_20260408.summary.md`
- Coverage v2 trace note: `run_logs/coverage_v2_trace_diag_musique_k5_20260408v3.md`
- Coverage v3 trace note: `run_logs/coverage_v3_trace_diag_musique_k5_20260408v3.md`
- Coverage budget-gap summary: `run_logs/coverage_budget_gap_musique_k5_20260409.summary.md`
- Coverage budget-gap trace: `run_logs/coverage_budget_gap_musique_k5_20260409.trace.md`
- Top-5 width-matched control summary, limit=100: `run_logs/width_matched_control_top5_20260407.summary.md`
- MuSiQue top-7 width-matched control summary, limit=100: `run_logs/width_matched_control_musique_top7_20260407.summary.md`
- Baseline+CE attribution summary: `run_logs/bridge_append_ce_control_20260407.summary.md`
- Invalid pseudo-full summary, do not use: `run_logs/fullscale_width_matched_control_20260409.summary.md`
- Active real full queue log: `run_logs/fullscale_width_matched_control_20260409fullfix.log`
- Active priority-switch watcher: `run_logs/switch_to_priority_width_matched_control_20260409.log`

## Width-matched CE control naming

These names are easy to confuse, so keep the definitions explicit.

- `baseline_top5_plus_ce`:
  - baseline-only CE rerank
  - `expand_base_k = qa_top_k`
  - `append_max_docs = 0`
  - useful as a point of comparison, but not the fairest same-pool append control
- `baseline_top10_plus_ce`:
  - width-matched no-append CE control
  - `expand_base_k = 10`
  - `append_max_docs = 0`
- `bridge_append_plus_ce`:
  - same-pool pure CE rerank control
  - candidate pool = `top-10 baseline prefix + 3 bridge-appended docs`
  - `assemble_mode = cross_encoder`
  - no extra setwise objective after expansion; the final stage is CE score and sort over the candidate docs
- `random3_deep_plus_ce`:
  - matched random-append CE control with the same append width and CE rerank stage

Implementation note:
- in `scripts/eval_causal_qwen3.py`, `select_bridge_append_positions(...)` builds the candidate pool
- non-coverage runs then call `rerank_candidate_positions_for_assemble(...)`
- under `assemble_mode = cross_encoder`, that path computes per-doc CE scores and sorts the candidates, without an additional exact-search or setwise optimization layer

Cross-run comparability note:
- the `limit=100` width-matched control family and the valid `20260409fullfix` full-scale family use the same method definition
- they differ only in evaluation size and in whether baseline/retrieval reuse is enabled to save runtime

## What is already established

### 1. Same-pool setwise opportunity is real

The cleanest positive evidence is `MuSiQue`, `K=5`, `append=0`:

- baseline: `0.27 / 0.3348`
- append0 + CE: `0.31 / 0.3698`
- append0 + coverage: `0.33 / 0.4007`

Source:
- `run_logs/coverage_post_diagnostics_20260407.md`

Interpretation:
- coverage is not a fake reranker in the small-budget, same-pool regime
- there is a real setwise composition signal on `MuSiQue K=5`

### 2. Coverage-v1 append behavior was genuinely poisoned

Frozen-atoms results on `MuSiQue` show:

- `K=5, append=3, candidate_pool`: `0.28 / 0.3538`
- `K=5, append=3, baseline_prefix`: `0.32 / 0.4040`
- `K=7, append=3, candidate_pool`: `0.31 / 0.3881`
- `K=7, append=3, baseline_prefix`: `0.33 / 0.4065`

Source:
- `run_logs/coverage_frozen_atoms_musique_20260408.summary.md`

Interpretation:
- the v2 frozen-atoms change fixed a real bug
- the main bug was candidate-induced objective poisoning

### 3. v2 gains come mainly from suppressing appended docs

The v2 trace on `MuSiQue K=5` shows:

- v1 selected appended query rate: `0.64`
- v2 selected appended query rate: `0.20`
- v2 avg appended selected: `0.29`
- v2 nonzero unique frozen CovE gain doc rate: `0.0345`
- average overlap with append0: `4.7 / 5`
- effective decision layer: `CE 67`, `CovE 32`, `fallback 1`

Source:
- `run_logs/coverage_v2_trace_diag_musique_k5_20260408v3.md`

Interpretation:
- v2 improves mostly because it suppresses appended influence
- the current honest description is closer to `coverage-constrained CE assembly` than to `coverage-driven assembly`

### 4. v3 restored structural novelty but not QA utility

The baseline-anchored v3 result on `MuSiQue K=5` is:

- v1 `candidate_pool`: `0.28 / 0.3538`
- v2 `baseline_prefix`: `0.32 / 0.4040`
- v3 `baseline_anchored`: `0.28 / 0.3603`

The v3 trace shows:

- selected appended query rate: `0.54`
- avg appended selected: `0.78`
- nonzero unique anchored CovE gain doc rate: `0.9231`
- average overlap with append0: `4.21 / 5`
- replaced incumbents: `0.62`
- effective decision layer: `CE 61`, `CovE 38`, `fallback 1`

Sources:
- `run_logs/coverage_anchored_musique_k5_20260408.summary.md`
- `run_logs/coverage_v3_trace_diag_musique_k5_20260408v3.md`

Interpretation:
- v3 did not merely collapse back to v1 mechanically
- it did allow many appended docs with real anchored structural gain
- but those structural gains did not translate into answer utility
- the deeper problem is not only `anchor set too wide`; it is also `structural novelty != reader-useful novelty`

### 5. Budget-gap admissibility did not recover useful novelty

`MuSiQue K=5`, frozen scaffold:

- `baseline_prefix + off`: `0.3200 / 0.4040`
- `baseline_prefix + budget_gap`: `0.3300 / 0.4007`

Trace summary:

- budget-gap selected appended query rate: `0.01`
- budget-gap avg appended selected: `0.01`
- admissible appended query rate: `0.01`
- average overlap with append0: `4.8 / 5`
- average replaced incumbents: `0.03`

Sources:
- `run_logs/coverage_budget_gap_musique_k5_20260409.summary.md`
- `run_logs/coverage_budget_gap_musique_k5_20260409.trace.md`

Interpretation:
- budget-aware admissibility stays safe
- but in the current proposal distribution it almost completely removes appended docs
- this again points to an `Expand / proposal-interface` bottleneck, not an assemble-objective bottleneck

## What the CE control line already says

### MuSiQue top-5, limit=100

- baseline top-5: `0.2700 / 0.3348`
- baseline top-5 + CE: `0.3100 / 0.3577`
- baseline top-10 + CE: `0.3100 / 0.3698`
- random3_deep + CE: `0.2900 / 0.3524`
- bridge_append + CE: `0.3200 / 0.3810`

### MuSiQue top-7, limit=100

- baseline top-7: `0.3200 / 0.3924`
- baseline top-7 + CE: `0.3400 / 0.4007`
- baseline top-10 + CE: `0.3400 / 0.4034`
- random3_deep + CE: `0.3400 / 0.3805`
- bridge_append + CE: `0.3800 / 0.4385`

Sources:
- `run_logs/width_matched_control_top5_20260407.summary.md`
- `run_logs/width_matched_control_musique_top7_20260407.summary.md`
- `run_logs/bridge_append_ce_control_20260407.summary.md`

Interpretation:
- on the 100-query controls, `bridge_append + CE` is better than `random3_deep + CE`
- this already supports that the `Expand` side has nontrivial value
- however, these are still `limit=100` controls, so the paper-facing claim should wait for the real full-scale control

## What is invalid and should not be cited

The file family tagged `20260409full` is invalid as full-scale output.

Reason:
- `scripts/eval_causal_qwen3.py` defaults to `--limit 20`
- the original `fullscale_width_matched_control_20260409.sh` did not pass `--limit`
- the resulting JSONs and summaries therefore ran on only `20` examples

Symptom:
- suspiciously quantized EM values such as `0.3000`, `0.5500`, `0.5000`

Do not use:
- `run_logs/fullscale_width_matched_control_20260409.summary.md`
- any `20260409full` JSON under `outputs_step0_general_*/eval_reports/`

Use instead:
- the active rerun tagged `20260409fullfix`

## What the ongoing full-scale control is trying to decide

The most important remaining question is narrow:

- does real full-scale `bridge_append + CE` beat full-scale `random3_deep + CE` on `MuSiQue top-5`?

If yes:
- current `Expand` already has independent value
- the paper can keep `Expand + CE-centered assemble` as the main method
- coverage can be written as a probe / analysis contribution

If no:
- the next method iteration should move to `Expand / proposal gating`
- there is little reason to keep tuning assemble objectives

## Live engineering note: parse failures in the active full queue

From the current `20260409fullfix` log snapshot:

- `parse_failed_attempt1`: `624`
- `repair_failed`: `0`
- `HTTP 200`: `1835`
- retrieval progress ticks recorded: `1002`
- QA reading progress ticks recorded so far: `375`

Source:
- `run_logs/fullscale_width_matched_control_20260409fullfix.log`

Interpretation:
- parse failures are common on the first attempt
- the repair path is not visibly failing
- the current runtime bottleneck is not just retrieval; full QA reading is the dominant wall-clock cost
- this is an engineering throughput issue, not currently a scientific blocker

## Current practical conclusion

The current best paper-facing position is:

- main method: `Expand + CE-centered assemble`
- coverage: probe / diagnostic line, not the primary method line
- main negative result: fixing poisoning and restoring structural novelty still does not make appended novelty reader-useful under the current proposal distribution
- next method work, if any, should move to `Expand / proposal interface`, not `coverage v4/v5`

## Immediate pending item

Wait for the real `MuSiQue top-5` full baseline to finish, then switch automatically to the priority full-scale control queue:

- `baseline_top10_plus_ce`
- `random3_deep_plus_ce`
- `bridge_append_plus_ce`

Watcher:
- `run_logs/switch_to_priority_width_matched_control_20260409.sh`
