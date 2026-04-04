# PCRS V2 Two-Query q6-Style Validation (2026-04-04)

## Scope

This note validates whether the q6 closure stack

- `q6_factual`
- `city_state_alias`
- `allow_seed_target`

transfers beyond q6 on a minimal two-query batch.

Validated queries:

1. `musique[13]`: `How many times did plague occur in the place where Crucifixion's creator died?`
2. `musique[5]`: `When was the region immediately north of the region where Israel is located and the location of the Battle of Qurah and Umm al Maradim created?`

Compared runs:

- baseline:
  [requirement_beam_needunit_oracle_plagueprobe_baseline_bridge_seed_hybrid_varfix_clean_20260404.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_plagueprobe_baseline_bridge_seed_hybrid_varfix_clean_20260404.json)
- E:
  [requirement_beam_needunit_oracle_plagueprobe_e_q6factual_cityalias_seedtarget_bridge_seed_hybrid_varfix_clean_20260404.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_plagueprobe_e_q6factual_cityalias_seedtarget_bridge_seed_hybrid_varfix_clean_20260404.json)
- baseline:
  [requirement_beam_needunit_oracle_saudiprobe_baseline_bridge_seed_hybrid_varfix_clean_20260404.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_saudiprobe_baseline_bridge_seed_hybrid_varfix_clean_20260404.json)
- E:
  [requirement_beam_needunit_oracle_saudiprobe_e_q6factual_cityalias_seedtarget_bridge_seed_hybrid_varfix_clean_20260404.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_saudiprobe_e_q6factual_cityalias_seedtarget_bridge_seed_hybrid_varfix_clean_20260404.json)

## Raw Table

| Query | Gold stage pattern under baseline | Baseline selector result | E result | Transfer verdict |
| --- | --- | --- | --- | --- |
| plague / Crucifixion | `selected + source + pool_only` | EM/F1=`0/0`, selected titles unchanged | identical to baseline | no transfer |
| Israel / Saudi Arabia | `selected + selected + pool_only + not_in_pool` | EM/F1=`0/0`, selected titles unchanged | identical to baseline | no transfer |

## Query 1: Crucifixion / plague

Gold titles:

- `Black Death`
- `Crucifixion (Titian)`
- `The Martyrdom of Saint Lawrence (Titian)`

Baseline stage pattern:

- `Black Death`: `selected`
- `Crucifixion (Titian)`: `source`
- `The Martyrdom of Saint Lawrence (Titian)`: `pool_only`

Observed selector output in both baseline and E:

- selected titles stayed exactly the same:
  `Crucifixion of Jesus`, `Black Death`, `Christian Ackermann`, `San Pietro in Montorio`, `Greece`
- selector answer stayed `0.`
- selector EM/F1 stayed `0 / 0`

Important exposure details:

- `Crucifixion (Titian)` stayed at `source`, with `best_scored_rank=3`, `best_combined_score=0.1844`
- it already had positive support gain (`0.1156`) but negative utility margin (`-0.0096`)
- `The Martyrdom of Saint Lawrence (Titian)` stayed `pool_only`, with `best_scored_rank=28` and zero gains

Interpretation:

- This is not the q6-style closure failure.
- The q6 stack does not help because the bottleneck is not missing factual-edge + alias + seed-target acceptance.
- The dominant symptom is closer to shortlist/final utility rejection on an art/creator/death chain, not a Minneapolis-style structural closure gap.

## Query 2: Israel / Saudi Arabia

Gold titles:

- `Geography of Saudi Arabia`
- `History of Saudi Arabia`
- `Israel`
- `Battle of Qurah and Umm al Maradim`

Baseline stage pattern:

- `Battle of Qurah and Umm al Maradim`: `selected`
- `Israel`: `selected`
- `Geography of Saudi Arabia`: `pool_only`
- `History of Saudi Arabia`: `not_in_pool`

Observed selector output in both baseline and E:

- selected titles stayed exactly the same:
  `Battle of Qurah and Umm al Maradim`, `Battle of Maroun al-Ras`, `Umayyad Caliphate`, `Israel`, `Kingdom of Israel (Samaria)`
- selector answer stayed `Not specified in the provided text.`
- selector EM/F1 stayed `0 / 0`

Important exposure details:

- `Israel` was already strong: `selected`, `best_scored_rank=1`, `best_support_completeness_gain=0.2105`
- `Geography of Saudi Arabia` stayed `pool_only`, with `pool_position=88`, `best_scored_rank=71`, and zero gains
- `History of Saudi Arabia` was absent from pool entirely

Interpretation:

- This query is not a clean q6-style closure target.
- The dominant blocker is pool coverage, not chain closure inside the current structure scorer.
- Since one gold doc is `not_in_pool`, q6-style structure fixes cannot rescue the full chain.

## Key Findings

1. Observation: both validation queries showed no difference between baseline and E.
   Interpretation: the q6 stack is not a universal repair for the entire `partial_chain_closure_candidate` bucket.
   Implication: q6 is a real family member, but only for a narrower subfamily.
   Next step: split failure-family analysis into at least three buckets: q6-style closure, shortlist/final utility rejection, and pool-coverage-limited cases.

2. Observation: the plague query superficially matched a staged pattern (`selected + source + pool_only`) but still did not respond to E.
   Interpretation: staged appearance alone is not enough; the underlying reasoning family matters.
   Implication: failure-family heuristics should not collapse all staged patterns into one closure class.
   Next step: add a sublabel for cases where source docs already have positive support gain but are rejected by margin/final utility.

3. Observation: the Saudi/Israel query contains an explicit `not_in_pool` gold doc (`History of Saudi Arabia`).
   Interpretation: this query is dominated by exposure/pool coverage, not structure closure.
   Implication: it should not be used as a selector-only closure validation target.
   Next step: exclude `not_in_pool` cases from the first closure-family validation batch, or at least separate them in reporting.

## Bottom Line

The two-query validation does not support broad generalization of the q6 stack. The current evidence is stronger for a narrower claim:

> q6 is not a singleton, but it represents a specific staged closure subfamily rather than the whole unresolved partial-chain-closure bucket.

Practical update:

- keep `varfix` on mainline
- keep `q6_factual + city_state_alias + allow_seed_target` eval-only
- refine the failure-family taxonomy before any attempt to generalize the E stack
