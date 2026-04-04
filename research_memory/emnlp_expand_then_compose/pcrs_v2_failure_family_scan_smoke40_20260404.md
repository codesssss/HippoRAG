# PCRS V2 Failure-Family Scan on MuSiQue Smoke40 (2026-04-04)

## Goal

After the q6 E run established a concrete three-part chain-closure mechanism, the next question was:

> is q6 an isolated special case, or does it belong to a broader failure family?

This scan intentionally does **not** mainline any fix. It only measures how often similar stage patterns appear in an existing small-sample report.

## Inputs

Existing report:

- `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke40_conditional_bridge_seed_hybrid_varfix_clean_20260404.json`

New analysis script:

- `scripts/analyze_requirement_failure_families.py`

Generated outputs:

- `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke40_conditional_bridge_seed_hybrid_varfix_clean_20260404.failure_families.json`
- `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke40_conditional_bridge_seed_hybrid_varfix_clean_20260404.failure_families.md`

## Method

The script reconstructs the runtime pool per query from the existing report and then recomputes exposure stages for **gold titles** using the saved selector trace:

- `not_in_pool`
- `pool_only`
- `source`
- `shortlist`
- `selected`
- `final_only`

Heuristic family tags:

- `pool_coverage_gap`
- `pool_to_source_gap`
- `source_to_shortlist_gap`
- `shortlist_to_selected_gap`
- `final_selected_but_answer_incomplete`
- `partial_chain_closure_candidate`

Important caveat:

- this is a **heuristic family scan**, not a proof that every tagged case has the same root cause as q6
- it is meant to answer prevalence / prioritization, not finalize the exact fix

## Commands

Self-check on smoke10 E:

```bash
.venv-hipporag/bin/python scripts/analyze_requirement_failure_families.py \
  --report outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_offlinehybrid_varfix_clean_e_q6factual_cityalias_seedtarget_20260404.json \
  --dataset musique \
  --limit 10 \
  --save_dir_root outputs_step0_general \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_offlinehybrid_varfix_clean_e_q6factual_cityalias_seedtarget_20260404.failure_families.json \
  --output_md outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_offlinehybrid_varfix_clean_e_q6factual_cityalias_seedtarget_20260404.failure_families.md
```

Smoke40 scan:

```bash
.venv-hipporag/bin/python scripts/analyze_requirement_failure_families.py \
  --report outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke40_conditional_bridge_seed_hybrid_varfix_clean_20260404.json \
  --dataset musique \
  --limit 40 \
  --save_dir_root outputs_step0_general \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke40_conditional_bridge_seed_hybrid_varfix_clean_20260404.failure_families.json \
  --output_md outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke40_conditional_bridge_seed_hybrid_varfix_clean_20260404.failure_families.md
```

## Smoke40 headline

Raw family counts over 40 queries:

- `final_selected_but_answer_incomplete`: 27
- `partial_chain_closure_candidate`: 24
- `pool_to_source_gap`: 22
- `source_to_shortlist_gap`: 9
- `pool_coverage_gap`: 6
- `shortlist_to_selected_gap`: 3

These raw counts are broad. A stricter unresolved subset is more informative:

- unresolved `partial_chain_closure_candidate`: 19
- unresolved `partial_chain_closure_candidate + pool_to_source_gap`: 15
- unresolved `partial_chain_closure_candidate + pool_to_source_gap + gold_count>=3`: 9
- unresolved `source_to_shortlist_gap`: 6

## What this says about q6-like failures

The q6 pattern is **not** an isolated one-off.

Even under this conservative scan:

- 15/40 queries show both:
  - a partial chain already reaching selected evidence
  - and at least one in-pool gold title still stuck at `pool_only`
- 9/40 queries show that same pattern with at least 3 gold titles

This is not proof that all 9 require the exact q6 fix stack, but it is enough to reject the hypothesis that q6 is just an exotic singleton.

## Representative unresolved candidates

Examples surfaced by the scan:

1. `Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`
   - gold: `Minneapolis`, `Mississippi River`, `Southeast Library`, `Riverside Plaza`
   - stage mix under the scanned smoke40 run:
     - `Southeast Library`: selected
     - `Minneapolis / Mississippi River / Riverside Plaza`: pool_only
   - this is the original q6 mechanism case

2. `When was the region immediately north of the region where Israel is located and the location of the Battle of Qurah and Umm al Maradim created?`
   - gold stage mix:
     - selected gold exists
     - `History of Saudi Arabia` and `Geography of Saudi Arabia` remain `pool_only`
   - looks like another partial-chain + pool-to-source failure

3. `How were the people from whom new coins were a proclamation of independence by the Somali Muslim Ajuran Empire expelled from the country between Thailand and A Lim's country?`
   - gold stage mix:
     - one gold selected
     - one gold shortlist
     - two gold pool_only
   - suggests a longer mixed-stage closure failure rather than a pure retrieval miss

4. `How many times did plague occur in the place where Crucifixion's creator died?`
   - gold stage mix:
     - one selected
     - one source
     - one pool_only
   - this looks closer to a staged promotion problem across multiple selector layers

## Decision

This scan supports the current strategy:

- do **not** mainline the q6 E stack yet
- do **not** jump back to final objective tuning
- do **not** let q7 drive selector design again

Instead:

- treat q6 as the first cleanly solved **mechanism template**
- treat the smoke40 counts as evidence that a broader **failure family** exists
- next effort should be a **small candidate-based validation pass**, not blind mainlining

Concretely, the right next step is:

- choose a small handful of the smoke40 candidates above
- check whether they exhibit the same three-part dependency pattern:
  - factual edge missing
  - continuity / alias gap
  - seed-target bridge acceptance gap

If several of them do, then the q6 stack becomes a candidate for broader controlled generalization.
