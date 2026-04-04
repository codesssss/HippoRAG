# PCRS-RAG V2 Shortlist Trace Diagnosis

Date: 2026-04-03

## Scope

This note diagnoses why `conditional third-hop` did not change the final evidence set in the clean `MuSiQue limit=10` smoke run on `qwen3-8b-train` / `8043`.

Files used:

- `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_clean_20260403.json`
- `research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_reuseopenie.json`

## Headline

For the current V2 branch, the third-hop signal is being neutralized by two upstream failures before it can change end-to-end behavior:

1. The `bridge` proposal stage still gates expansion with a very shallow top-4 shortlist.
2. The need-unit annotation layer assigns near-zero support to the true chain-closing documents on the key failure case.

Because both layers fail together, widening compile-time hop budget alone does not change the final finalists.

## Query 6

Question:

`Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`

### Ground truth support titles in top-100 pool

- `Southeast Library`: position `0`
- `Riverside Plaza`: position `29`
- `Minneapolis`: position `37`
- `Mississippi River`: position `42`

### Conditional V2 positive need units

- `designer_of`
- `death_place`
- `empties_into`
- `answer_slot`

The third relation hop is preserved in conditional mode:

- `pre_cap_relation_hop_count = 3`
- `post_cap_relation_hop_count = 3`
- `conditional_third_hop_allowed = true`

### Proposal shortlist diagnosis

Using the same top-100 pool and the current bridge proposal scoring path, the first-step bridge candidates after the reserved prefix are dominated by early-ranked novelty documents instead of true chain-closing support.

Top bridge candidates after reserved prefix:

1. `Oklahoma City` at pool position `4`
2. `The Hague City Hall` at pool position `6`
3. `Elliott Bay` at pool position `9`
4. `Stockholm Public Library` at pool position `10`

The true support docs needed for the chain are much deeper:

- `Riverside Plaza`: `29`
- `Minneapolis`: `37`
- `Mississippi River`: `42`

This means the real chain is not entering the shallow bridge proposal frontier early enough.

### Need-unit annotation diagnosis

Even when explicitly inspected, the true support docs receive zero need-unit gain in the current cache:

- `Riverside Plaza`
  - `support_completeness_gain = 0.0`
  - `utility_margin_gain = 0.0`
  - `doc_positive_mean = 0.0`
- `Minneapolis`
  - `support_completeness_gain = 0.0`
  - `utility_margin_gain = 0.0`
  - `doc_positive_mean = 0.0`
- `Mississippi River`
  - `support_completeness_gain = 0.0`
  - `utility_margin_gain = 0.0`
  - `doc_positive_mean = 0.0`

By contrast, the actually selected distractors have small but non-zero scores:

- `Oklahoma City`
  - `support_completeness_gain ≈ 0.0013`
  - `counterfactual_leakage_gain ≈ 0.0307`
- `Stockholm Public Library`
  - `support_completeness_gain ≈ 0.0013`
  - `counterfactual_leakage_gain ≈ 0.0307`

So for query 6, the third hop is not dying at compile-time gating. It survives compile-time, but the supporting documents behind that hop are both:

- too deep to enter the shallow bridge proposal frontier
- scored as zero-value by the current need-unit annotation layer

## Query 7

Question:

`When did the explorer reach the city where the headquarters of the only group larger than Vilaiyaadu Mankatha's record label is located?`

### Conditional V2 positive need units

- `publisher`
- `headquartered_in`
- `answer_slot`

Conditional mode blocks the third hop here:

- `pre_cap_relation_hop_count = 3`
- `post_cap_relation_hop_count = 2`
- `conditional_third_hop_allowed = false`
- reasons:
  - `third_relation_chain_not_high_confidence`
  - `non_contiguous_relation_chain`

This is consistent with the manual inspection: the dropped third hop was not a clean answer-reaching chain.

### Need-unit annotation diagnosis

Among inspected candidates:

- `Seattle` shows the only clearly positive margin signal
  - `support_completeness_gain ≈ 0.1426`
  - `counterfactual_leakage_gain ≈ 0.1236`
  - `utility_margin_gain ≈ 0.0190`
- `Sony Music` still gets zero signal
  - `support_completeness_gain = 0.0`
  - `utility_margin_gain = 0.0`

This supports the interpretation that conditional gating is not the main problem on query 7. The parser/compiler is already correctly refusing a noisy third hop.

## Mechanism Conclusion

The current failure is not best described as:

`conditional third-hop failed to help`

It is better described as:

`conditional third-hop is working, but the relevant documents are blocked or neutralized before that extra hop can affect the final evidence set`

More concretely:

- `query 6`: real third-hop support is lost to shallow proposal shortlist plus zero-value need-unit annotation
- `query 7`: the blocked third hop is noisy enough that conditional gating is likely correct

## Action Implication

Do not spend the next round on larger reader runs or more compile-time hop-budget sweeps.

Higher-value next steps are:

1. Improve need-unit annotation / scorer calibration so true support docs like `Riverside Plaza`, `Minneapolis`, and `Mississippi River` no longer receive zero signal.
2. Use the new selector trace fields to inspect whether shortlist widening would help after scorer quality improves.

## Code Support Added

`select_requirement_beam_positions()` now records, for each chosen beam step:

- `candidate_source_preview`
- `candidate_shortlist_preview`

These fields are attached to `selection_steps` so future runs can reveal, per step, which bridge candidates were available before expansion and which ones actually entered the beam.
