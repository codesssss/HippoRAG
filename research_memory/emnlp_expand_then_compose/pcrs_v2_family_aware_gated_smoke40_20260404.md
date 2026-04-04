# PCRS V2 Family-Aware Gated Smoke40 (2026-04-04)

## Goal

Validate whether a conservative, family-aware gate can recover value from `requirement_beam` without globally enabling the q6-style E stack.

Enabled stack behind the gate:

- `q6_factual`
- `city_state_alias`
- `allow_seed_target`

Baseline report:

- [requirement_beam_needunit_oracle_smoke40_conditional_bridge_seed_hybrid_varfix_clean_20260404.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke40_conditional_bridge_seed_hybrid_varfix_clean_20260404.json)

Gated report:

- [requirement_beam_needunit_oracle_smoke40_familygate_conservative_20260404.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke40_familygate_conservative_20260404.json)
- [requirement_beam_needunit_oracle_smoke40_familygate_conservative_20260404.md](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke40_familygate_conservative_20260404.md)

Subset E report reused by the wrapper:

- [requirement_beam_needunit_oracle_smoke40_familygate_subset_e_20260404.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke40_familygate_subset_e_20260404.json)

## Gate Rule

Conservative staged-closure gate:

- must be `partial_chain_closure_candidate`
- must **not** be `pool_coverage_gap`
- must **not** show the utility-rejection heuristic
- must have at least one `selected/final_only` gold
- must have at least one `pool_only` gold
- must have `gold_count >= 3`

Utility-rejection heuristic:

- any gold doc at `source` or `shortlist`
- with `best_support_completeness_gain >= 0.05`
- and `best_utility_margin_gain <= 0`

Implementation:

- default behavior remains unchanged
- gate is eval-only via [run_requirement_family_gated_eval.py](/mnt/nvme/code/HippoRAG/scripts/run_requirement_family_gated_eval.py)
- family logic lives in [requirement_family_gate_utils.py](/mnt/nvme/code/HippoRAG/src/hipporag/utils/requirement_family_gate_utils.py)

## Command

```bash
.venv-hipporag/bin/python scripts/run_requirement_family_gated_eval.py \
  --baseline_report outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke40_conditional_bridge_seed_hybrid_varfix_clean_20260404.json \
  --dataset musique \
  --limit 40 \
  --save_dir_root outputs_step0_general \
  --gate_mode conservative_staged_closure \
  --structure_relation_probe_mode q6_factual \
  --structure_continuity_probe_mode city_state_alias \
  --structure_seed_target_bridge_mode allow_seed_target
```

## Raw Results

Overall against the plain baseline:

- baseline EM/F1: `0.3750 / 0.4042`
- gated requirement_beam EM/F1: `0.3000 / 0.3304`
- delta: `-0.0750 / -0.0738`
- Recall@5 / Recall@10: `0.6375 / 0.7104`

Subset summaries:

- staged-closure subset (`n=9`):
  baseline EM/F1 = `0.4444 / 0.4444`
  gated EM/F1 = `0.4444 / 0.4889`
  delta = `+0.0000 / +0.0444`
- utility-rejection subset (`n=1`):
  baseline EM/F1 = `0.0000 / 0.0000`
  gated EM/F1 = `0.0000 / 0.0000`
  delta = `+0.0000 / +0.0000`
- pool-limited subset (`n=11`):
  baseline EM/F1 = `0.0909 / 0.1364`
  gated EM/F1 = `0.0000 / 0.0410`
  delta = `-0.0909 / -0.0954`
- non-staged subset (`n=31`):
  baseline EM/F1 = `0.3548 / 0.3925`
  gated EM/F1 = `0.2581 / 0.2843`
  delta = `-0.0968 / -0.1081`

## Critical Interpretation

The headline staged-closure gain needs a careful read:

- the gated smoke40 result `0.3304` is numerically identical to the existing varfix-clean `requirement_beam` smoke40 result
- the 9 gate-hit queries did **not** change relative to the existing requirement_beam baseline
- their improvement only appears when compared against the plain reader baseline, not against the already-existing requirement_beam smoke40 line

In other words:

> the family-aware gate did not unlock new q6-style wins on smoke40; it mostly reidentified a subset where requirement_beam was already doing relatively better than the plain baseline.

## Trigger Audit

Gate hit count:

- `9 / 40`

Actual trigger counts from the probe audit:

- `q6_factual`: `0`
- `city_state_alias`: `0`
- `allow_seed_target`: `0`

This is the strongest negative signal in the run.

It means:

- the conservative gate fired on nine queries
- but none of them actually exhibited the q6-style structural trigger pattern under the offline audit
- so the gate is still too coarse for “real q6-family activation”

Representative gate-hit traces that did not change:

- `How many times did plague occur in the place where Crucifixion's creator died?`
  baseline selector F1 = `0.0`
  gated selector F1 = `0.0`
  triggers = all false
- `Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`
  baseline selector F1 = `0.4`
  gated selector F1 = `0.4`
  triggers = all false
- `Who was second pick in the 1999 draft ...`
  baseline selector F1 = `1.0`
  gated selector F1 = `1.0`
  triggers = all false

## Stop-Loss Decision

Final verdict:

- `应切线`

Reason:

- overall F1 still trails baseline by `0.0738`
- staged-closure subset looks positive only relative to the plain baseline, not relative to the existing requirement_beam smoke40 result
- non-staged queries are still substantially worse
- the trigger audit shows the gate did not isolate real q6-style activations

## Practical Recommendation

For the 2026-04-06 stop-loss window:

- do **not** continue pushing `requirement_beam` as a baseline-chasing mainline
- downgrade this line to `analysis-only contribution`
- preserve the reusable pieces:
  - `varfix` on mainline as correctness
  - q6-style E stack as a mechanistic case-study tool
  - failure-family analysis tooling for diagnosis

The clean claim now is:

> q6 remains a valid staged-closure mechanism case study, but a conservative family-aware gate was not enough to convert that mechanism into smoke40-wide selector gains. This is not a strong baseline-beating path under the current time budget.
