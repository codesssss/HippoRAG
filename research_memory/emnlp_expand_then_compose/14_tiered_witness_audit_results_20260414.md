# Tiered Witness Audit Results

Last updated: 2026-04-14

## Scope

This note records the first offline audit result for:

- `assemble_mode=action_swap_tiered_witness`

This audit was run before any new online smoke.
That ordering is intentional.
The branch should not consume more reader-side budget unless the offline scorer shows clear separation on oracle-positive actions.

## Canonical output files

### MuSiQue

- `run_logs/musique_action_swap_tiered_witness_audit_20260414.json`
- `run_logs/musique_action_swap_tiered_witness_audit_20260414.md`
- `run_logs/musique_action_swap_tiered_witness_audit_20260414.csv`

### 2Wiki

- `run_logs/2wiki_action_swap_tiered_witness_audit_20260414.json`
- `run_logs/2wiki_action_swap_tiered_witness_audit_20260414.md`
- `run_logs/2wiki_action_swap_tiered_witness_audit_20260414.csv`

## Execution commands used

### MuSiQue

`env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy CUDA_VISIBLE_DEVICES=3 .venv-hipporag/bin/python scripts/audit_action_swap_tiered_witness.py --dataset musique --oracle_relaxed_report run_logs/musique_action_swap_oracle_relaxed_20260413.json --query_report outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_action_swap_noisyor_dep_qatopk5_20260414noisyor.json --dryrun_report outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_action_swap_v0_dryrun_qatopk5_20260413impl.json --judge_report outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_action_swap_v0_judge_qatopk5_20260413impl.json --output_json run_logs/musique_action_swap_tiered_witness_audit_20260414.json --output_md run_logs/musique_action_swap_tiered_witness_audit_20260414.md --output_csv run_logs/musique_action_swap_tiered_witness_audit_20260414.csv --ce_device cuda:0 --negative_sample_size 40`

### 2Wiki

`env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy CUDA_VISIBLE_DEVICES=3 .venv-hipporag/bin/python scripts/audit_action_swap_tiered_witness.py --dataset 2wikimultihopqa --oracle_relaxed_report run_logs/2wiki_action_swap_oracle_relaxed_20260413.json --query_report outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_action_swap_noisyor_dep_qatopk5_20260414noisyor.json --dryrun_report outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_action_swap_v0_dryrun_qatopk5_20260413impl.json --judge_report outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_action_swap_v0_judge_qatopk5_20260413impl.json --output_json run_logs/2wiki_action_swap_tiered_witness_audit_20260414.json --output_md run_logs/2wiki_action_swap_tiered_witness_audit_20260414.md --output_csv run_logs/2wiki_action_swap_tiered_witness_audit_20260414.csv --ce_device cuda:0 --negative_sample_size 40`

## Headline result

The current tiered witness scorer did not separate positives from negatives offline.

Not "weakly separated".
Not "below margin".

It collapsed to zero gain on both audited datasets.

## MuSiQue

Selected actions audited:

- total: `197`
- oracle-positive: `148`
- oracle-negative sampled: `49`
- dryrun-selected: `16`
- judge-selected: `2`

Tier mode usage:

- heuristic tiers queries: `29`
- flat fallback queries: `40`

Gain statistics:

- oracle-positive median gain: `0.0`
- oracle-positive nonpositive rate: `148 / 148 = 100%`
- oracle-negative median gain: `0.0`
- oracle-negative nonpositive rate: `49 / 49 = 100%`
- dryrun-selected nonpositive rate: `16 / 16 = 100%`
- judge-selected nonpositive rate: `2 / 2 = 100%`

Ranking recall:

- top-1 oracle-positive hit rate: `57.97%` over `69` queries
- top-3 oracle-positive hit rate: `57.97%`

Interpretation:

- the scorer is not recovering positive swap preference
- even already executed dryrun / judge positives are not lifted
- `top1 == top3` indicates the ranking signal is effectively tied or degenerate, not usefully calibrated

## 2Wiki

Selected actions audited:

- total: `107`
- oracle-positive: `55`
- oracle-negative sampled: `52`
- dryrun-selected: `21`
- judge-selected: `3`

Tier mode usage:

- heuristic tiers queries: `14`
- flat fallback queries: `24`

Gain statistics:

- oracle-positive median gain: `0.0`
- oracle-positive nonpositive rate: `55 / 55 = 100%`
- oracle-negative median gain: `0.0`
- oracle-negative nonpositive rate: `52 / 52 = 100%`
- dryrun-selected nonpositive rate: `21 / 21 = 100%`
- judge-selected nonpositive rate: `3 / 3 = 100%`

Ranking recall:

- top-1 oracle-positive hit rate: `55.26%` over `38` queries
- top-3 oracle-positive hit rate: `55.26%`

Interpretation:

- this is the same failure shape as MuSiQue
- the current scorer still does not recover positive action preference
- heuristic tiering does not rescue the branch because the issue happens before action ranking becomes informative

## What failed

The current implementation fails before reader execution.

The failure point is:

- the local witness gain remains zero for oracle-positive swaps

Empirically, the branch does not even reach the earlier `noisyor` style question of:

- "is the action positive but below margin?"

Instead it fails at a stricter earlier stage:

- "does the scorer assign any positive local bottleneck gain at all?"

For many sample failures, the audit rows show:

- `bottleneck_tier_index = None`
- `target_facet_id = None`
- `swap_gain_vs_loss = 0.0`

So the controller often never identifies a bottleneck facet in the first place.

## Likely current failure mechanism

The evidence supports the following diagnosis for the current implementation:

1. the `min(relevance, answerability)` witness operator is too harsh in practice
2. the current ASRank-like answer-scent heuristic is too weak or too sparse on these facets
3. as a result, candidate witness gain frequently stays at zero even for oracle-positive swaps

This is a diagnosis about the current scorer instantiation.
It is not evidence that actionized interface repair is wrong.

## Decision

Do not run new online smoke for this branch.

The offline gate failed.
So the current `action_swap_tiered_witness` branch should stop here unless there is a separate decision to explicitly debug the witness scorer itself.

## Paper-facing consequence

This branch should not be promoted into the main method.

The canonical interpretation is:

- `action_swap_v0_*` still provides the positive intervention evidence for actionized interface repair
- oracle decomposition still provides the legality-miss / policy-miss diagnosis
- `tiered_witness` is another negative controller instance, now with a more local scorer than `noisyor`, but still failing to recover positive swap preference offline

## Recommended route from here

The route is narrow:

1. do not run reader-side smoke for `action_swap_tiered_witness`
2. do not add more heuristic gates on top of this implementation
3. only revisit this line if the task is explicitly to debug witness scoring offline

If revisited, the first debug target should be:

- the answerability channel and the `min(relevance, answerability)` collapse

not:

- more tier rules
- more action thresholds
- more online controller variants
