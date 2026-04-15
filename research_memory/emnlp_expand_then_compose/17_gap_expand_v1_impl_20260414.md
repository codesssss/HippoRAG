# Gap-Conditioned Expand V1 Implementation

Date: 2026-04-14

## Goal

Move the interface repair from controller-side action scoring to the expand stage:

- keep the existing `bridge_append + CE` assemble path
- replace generic deep append proposals with scaffold-conditioned gap-aware proposals
- keep the implementation inside the current `bridge_append` skeleton for fair comparison

This is the first implementation pass of:

- `Scaffold-conditioned Gap Expand + existing CE-centered assemble`

## What Was Implemented

### 1. New append policy

In `scripts/eval_causal_qwen3.py`:

- added `append_policy="gap_expand"`

The selector still preserves:

- baseline prefix
- append cap
- title dedup
- downstream CE rerank / final reader order

Only the appended candidate proposal order changes.

### 2. Heuristic gap detector

Added a lightweight gap detector:

- `detect_gap_expand_state(...)`

It infers one of:

- `bridge_entity`
- `target_attribute`
- `linking_relation`
- `flat_fallback`

Inputs:

- query
- baseline prefix positions
- baseline-covered entities
- query entities

Behavior:

- if query entities are still uncovered by the current scaffold, prefer `bridge_entity`
- if entities are mostly covered but relation-bearing lexical terms remain, prefer `linking_relation`
- otherwise prefer `target_attribute`
- if inference is unstable, or `gap_expand_mode=flat_fallback_only`, use `flat_fallback`

### 3. Micro-query builder

The gap detector emits up to `gap_expand_max_queries` query variants:

- raw question
- one gap-specific micro-query

Examples:

- bridge entity:
  - `Find the intermediate entity linking ...`
- target attribute:
  - `Find the target attribute or answer-bearing relation ...`
- linking relation:
  - `Find the linking relation between ...`

These are currently used as a proposal-scoring view over the existing deep pool, not as a new corpus-wide retrieval path.

### 4. Gap-aware rerank over existing bridge rows

Added:

- `rerank_gap_expand_candidates(...)`

Implementation choice:

- reuse the existing `score_bridge_candidates(...)` output
- add a gap-aware score on top of those rows
- sort appended proposals by:
  - `gap_combined_score`
  - then `gap_score`
  - then existing bridge combined score

This keeps the original structure score in play instead of replacing it.

Current gap-aware score uses:

- lexical overlap with emitted micro-queries
- overlap with current scaffold anchors
- overlap with uncovered query entities
- relation-term overlap
- existing structure / closure features as backoff

Admission rule:

- a candidate is allowed if `max(structure_score, gap_score) >= expand_min_structure_score`

This preserves the old append threshold semantics while allowing CE-negative but gap-positive bridge candidates to survive proposal filtering.

### 5. New CLI flags

Added:

- `--append_policy gap_expand`
- `--gap_expand_mode {heuristic,flat_fallback_only}`
- `--gap_expand_max_queries <int>`

### 6. New traces and summaries

Per-query `expand_assemble_trace` now includes:

- `gap_expand_enabled`
- `gap_expand_mode`
- `gap_expand_max_queries`
- `gap_type`
- `gap_mode`
- `gap_fallback_used`
- `gap_micro_queries`
- `gap_micro_query_count`
- `gap_anchors`
- `gap_candidate_positions`
- `gap_candidate_doc_ids`
- `gap_candidate_titles`
- `gap_candidate_ce_scores`
- `gap_candidate_selected_into_final`
- `gap_candidate_selected_count`
- `gap_candidate_final_front_rate`
- `gap_steps`

Bridge-append summary now also reports, when `append_policy=gap_expand`:

- `gap_expand_query_count`
- `gap_expand_fallback_query_count`
- `gap_expand_fallback_rate`
- `gap_type_distribution`
- `avg_gap_micro_query_count`
- `avg_gap_candidate_count`
- `avg_gap_candidates_selected_into_final`
- `gap_candidate_selected_query_rate`

## Validation

### Unit tests

Validated with:

```bash
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
  pytest tests/test_setwise_selector.py -q
```

Result:

- `161 passed`

Targeted new tests cover:

- gap detector choosing `bridge_entity`
- gap-aware rerank preferring a bridge-style candidate
- `select_bridge_append_positions(..., append_policy="gap_expand")`

### Notes on test environment

The repo test environment may inherit a SOCKS proxy variable. The OpenAI client construction tests in `tests/test_setwise_selector.py` fail during collection if those proxy env vars are left enabled and `socksio` is absent. This is environment-specific, not related to `gap_expand`.

## Recommended Smoke Commands

### MuSiQue smoke100

```bash
.venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
  --dataset musique \
  --limit 100 \
  --setwise_selector bridge_append \
  --setwise_pool_k 100 \
  --expand_base_k 10 \
  --append_max_docs 3 \
  --append_policy gap_expand \
  --gap_expand_mode heuristic \
  --gap_expand_max_queries 2 \
  --assemble_mode cross_encoder \
  --qa_top_k 5 \
  --ce_device cuda:0 \
  --output_json outputs_step0_general_musique/eval_reports/width_match_gap_expand_bridge_append_plus_ce_qatopk5_20260414impl.json
```

### 2Wiki smoke100

```bash
.venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
  --dataset 2wikimultihopqa \
  --limit 100 \
  --setwise_selector bridge_append \
  --setwise_pool_k 100 \
  --expand_base_k 10 \
  --append_max_docs 3 \
  --append_policy gap_expand \
  --gap_expand_mode heuristic \
  --gap_expand_max_queries 2 \
  --assemble_mode cross_encoder \
  --qa_top_k 5 \
  --ce_device cuda:0 \
  --output_json outputs_step0_general_2wikimultihopqa/eval_reports/width_match_gap_expand_bridge_append_plus_ce_qatopk5_20260414impl.json
```

## Recommended Experiment Order

1. Proposal audit first
   - compare `append_policy=bridge` vs `append_policy=gap_expand`
   - inspect:
     - `gap_type_distribution`
     - fallback rate
     - appended docs entering final front

2. CE mainline smoke
   - `baseline+CE`
   - `bridge_append+CE`
   - `gap_expand_bridge_append+CE`

3. Small dryrun comparison only after CE smoke
   - `gap_expand + action_swap_v0_dryrun`

4. Only if proposal-level signal is positive
   - move to atomic evidence units as a separate v1.5

## Current Scope Boundary

This implementation does **not** yet do:

- corpus-wide micro-query retrieval outside the current deep pool
- atomic evidence unit indexing
- answer-first repair
- any new controller or verifier

This is intentional. V1 is only:

- scaffold-conditioned expand over the current deep pool
- existing CE-centered assemble unchanged

## Smoke40 Result: 2Wiki

Executed on 2026-04-15 with cached local assets and retrieval cache:

- report:
  - `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_gap_expand_bridge_append_plus_ce_qatopk5_20260414smoke40.json`
- paired audit:
  - `run_logs/2wiki_gap_expand_smoke40_paired_20260415.json`
  - `run_logs/2wiki_gap_expand_smoke40_paired_20260415.md`

Primary QA:

- `bridge_append + CE` baseline on this smoke slice:
  - `EM 0.4300 / F1 0.4843`
- `gap_expand + bridge_append + CE`:
  - `EM 0.3750 / F1 0.4809`
- delta:
  - `EM -0.0550 / F1 -0.0034`

Selector / trace summary:

- `gap_type_distribution = {"linking_relation": 40}`
- `gap_expand_fallback_rate = 0.0`
- `avg_appended_doc_count = 1.375`
- `gap_candidate_selected_query_rate = 0.55`
- `avg_gap_candidates_selected_into_final = 0.675`
- `append_stop_reason_counts = {"append_cap_reached": 11, "structure_below_threshold": 29}`

Bucket breakdown:

- `2_doc`:
  - baseline `0.3750 / 0.4499`
  - method `0.4375 / 0.5548`
  - delta `+0.0625 / +0.1049`
- `4_doc`:
  - baseline `0.2500 / 0.2858`
  - method `0.1250 / 0.1854`
  - delta `-0.1250 / -0.1004`

## Paired Audit Takeaways

Paired comparison against the first 40 queries of
`width_match_bridge_append_plus_ce_qatopk5_20260409fullfix.json` shows:

- F1 win / tie / lose:
  - `2 / 34 / 4`
- EM win / tie / lose:
  - `0 / 37 / 3`
- selected-gap subset:
  - `22` queries
  - F1 `0 / 20 / 2`
- changed-evidence subset:
  - `19` queries
  - F1 `0 / 17 / 2`

The changed-evidence split matters most:

- there were **no true query-level wins** where `gap_expand` changed the final top-k and improved F1
- the two apparent wins came from unchanged evidence sets, so they are reader variance rather than selector gain
- the only true intervention effects on this smoke were **two regressions**

Representative true regressions:

1. `Which film has the director born later, Christ Walking On The Water or 45 Fathers?`
   - added:
     - `Roman Polanski`
     - `Jeethu Joseph`
   - removed:
     - `James Tinling`
     - `Benjamin Christensen`
   - F1: `1.0 -> 0.0`

2. `Which film has the director who was born later, Playing It Wild or I'll Be Going Now?`
   - added:
     - `Aditya Chopra`
   - removed:
     - `Ildikó Enyedi`
   - F1: `0.5714 -> 0.2222`

## Current Diagnosis

The failure is not infrastructure or keep-only collapse:

- `gap_expand` executed normally
- gap candidates frequently entered the final front
- retrieval recall moved slightly up on the smoke slice

The current failure mode is proposal semantics:

1. the heuristic gap detector collapses to `linking_relation` on every 2Wiki query in this slice
2. it never uses flat fallback, so there is no route diversity
3. once it actually changes the evidence set, it does not produce any observed QA gain on this smoke
4. the observed true interventions skew negative and are concentrated in the harder `4_doc` subset

So the current v1 conclusion is:

> the `gap_expand` interface shift is implemented and testable, but the current heuristic gap typing and rerank policy do not yet improve proposal utility on 2Wiki smoke; they mainly over-predict `linking_relation` and occasionally replace useful directors / bridge evidence with topical relation lookalikes.

## V2 Implementation: Typed-Abstaining Gap Expand

Date: 2026-04-15

This is a narrow follow-up to the failed `heuristic` V1.

The goal is not to change assemble or introduce a new controller. The goal is only:

- stop collapsing every query into `linking_relation`
- allow the proposal router to abstain when slot typing is unclear
- keep the original `bridge` top proposal intact
- add at most one typed alternate instead of replacing proposal ownership

### What Changed in Code

In `scripts/eval_causal_qwen3.py`:

- added `gap_expand_mode="typed_abstain"`
- extended `detect_gap_expand_state(...)` to emit:
  - `gap_type ∈ {role_relation, target_attribute, bridge_entity, abstain}`
  - `gap_slot`
  - `gap_slot_cues`
  - `gap_bridge_targets`
  - `gap_abstain_reason`
  - `fallback_used`
- added typed micro-query helpers:
  - `_find_gap_rule(...)`
  - `_resolve_gap_title_aligned_entities(...)`
  - `_resolve_gap_anchor_entities(...)`
  - `_build_gap_micro_queries(...)`
- added a typed rerank path in `rerank_gap_expand_candidates(...)`
  - uses local witness units
  - adds `gap_filter_passed`
  - adds `gap_best_witness_anchor`
  - adds `gap_best_witness_slot`
  - adds `gap_best_witness_query`
  - adds `gap_non_anchor_entity_gain`
- added `select_gap_expand_positions_typed_abstain(...)`
  - if typed routing is unstable, return `abstain` and fall back to raw `bridge_append`
  - otherwise:
    - first keep the original bridge top-1 as `bridge_primary`
    - then add at most one slot-aware `typed_alternate`
    - never let the alternate replace the primary proposal directly

### Additional Summary Fields

The bridge-append summary now also tracks:

- `gap_slot_distribution`
- `gap_abstain_reason_counts`

### Tests Added

`tests/test_setwise_selector.py` now covers:

- typed role-relation detection with anchor + slot
- abstention when no typed slot is available
- preserving bridge primary while adding one typed alternate
- bridge fallback when typed routing abstains

Validation:

- `env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy pytest tests/test_setwise_selector.py -k "gap_expand or typed_abstain" -q`
  - `7 passed`
- `env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy pytest tests/test_setwise_selector.py -q`
  - `165 passed`

## Smoke2 Sanity Check: 2Wiki with Local 8B

Executed on 2026-04-15 after fixing two runtime issues:

1. do not switch `llm_name` away from `qwen3-8b`, otherwise HippoRAG rebuilds a fresh asset directory instead of reusing
   `outputs_step0_general_2wikimultihopqa/qwen3-8b_VLLM__mnt_nvme_Qwen3-Embedding-8B`
2. do not pass the full `retrieval_cache_json` to a smaller `--limit`, because `eval_causal_qwen3.py` requires cache example count to match query count exactly

Final smoke2 command:

```bash
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
  OPENAI_API_KEY=EMPTY \
  .venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
  --dataset 2wikimultihopqa \
  --limit 2 \
  --save_dir outputs_step0_general \
  --llm_name qwen3-8b \
  --llm_request_name qwen3-8b-train \
  --llm_base_url http://localhost:8043/v1 \
  --embedding_name VLLM//mnt/nvme/Qwen3-Embedding-8B \
  --embedding_base_url http://localhost:8018/v1/embeddings \
  --causal_enabled false \
  --causal_engine_version v2 \
  --causal_v2_base_retrieval_mode legacy_fact_graph \
  --setwise_selector bridge_append \
  --setwise_pool_k 100 \
  --expand_base_k 10 \
  --append_max_docs 3 \
  --append_policy gap_expand \
  --gap_expand_mode typed_abstain \
  --gap_expand_max_queries 2 \
  --assemble_mode cross_encoder \
  --qa_top_k 5 \
  --ce_device cuda:3 \
  --output_json outputs_step0_general_2wikimultihopqa/eval_reports/width_match_gap_expand_bridge_append_plus_ce_qatopk5_20260415typedabstain_smoke2.json
```

Smoke2 report:

- `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_gap_expand_bridge_append_plus_ce_qatopk5_20260415typedabstain_smoke2.json`

Smoke2 QA:

- baseline QA in the run:
  - `EM 0.5000 / F1 0.7500`
- typed-abstain `bridge_append + CE`:
  - `EM 0.5000 / F1 0.5000`

The score on 2 examples is not meaningful. The important part is the trace behavior:

- query A produced:
  - `gap_type = target_attribute`
  - `gap_slot = death_date`
  - `gap_fallback_used = false`
  - a typed alternate with nonzero witness score entered the append trace
- query B produced:
  - `gap_type = abstain`
  - `gap_abstain_reason = no_typed_slot`
  - `gap_fallback_used = true`
  - no typed alternate was forced

So the key V2 routing hypothesis passed the first sanity check:

- typed routing no longer collapses to one universal label
- abstention is active
- fallback is active
- typed alternate generation is active

At the time of writing, the next step is a `limit=40` smoke on the same 8B path:

- `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_gap_expand_bridge_append_plus_ce_qatopk5_20260415typedabstain_smoke40.json`

That run is the one that will decide whether V2 is worth keeping beyond implementation.

## Smoke40 Result: 2Wiki Typed-Abstain V2

Executed on 2026-04-15 with the same local `qwen3-8b-train` reader path:

- report:
  - `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_gap_expand_bridge_append_plus_ce_qatopk5_20260415typedabstain_smoke40.json`

Primary QA within the run:

- baseline QA:
  - `EM 0.3500 / F1 0.4170`
- typed-abstain `gap_expand + bridge_append + CE`:
  - `EM 0.4500 / F1 0.5271`
- delta vs baseline inside the same run:
  - `EM +0.1000 / F1 +0.1101`

Selector / trace summary:

- `gap_expand_fallback_rate = 0.3`
- `gap_type_distribution = {"abstain": 12, "role_relation": 14, "target_attribute": 14}`
- `gap_slot_distribution = {"birthplace": 6, "death_date": 3, "director": 14, "education": 2, "nationality": 3}`
- `gap_abstain_reason_counts = {"no_typed_slot": 12}`
- `avg_appended_doc_count = 0.7`
- `avg_gap_candidate_count = 0.175`
- `avg_gap_candidates_selected_into_final = 0.125`
- `gap_candidate_selected_query_rate = 0.125`
- `append_stop_reason_counts = {"alternate_only_complete": 7, "append_cap_reached": 2, "no_typed_alternate": 4, "structure_below_threshold": 26, "typed_alternate_below_threshold": 1}`

Immediate interpretation:

- V2 did fix the V1 routing collapse:
  - it no longer predicts only `linking_relation`
  - abstention is active on `30%` of this slice
  - typed routes are split between `role_relation` and `target_attribute`
- V2 is also much more conservative than V1:
  - typed candidates only reach the final front on `12.5%` of queries
- the remaining open question is comparative, not infrastructural:
  - does this conservative typed routing beat or trail a same-slice `bridge_append + CE` control?

## Same-Slice Control: Bridge Append + CE

Executed on the same 40-query slice and same local 8B path:

- report:
  - `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260415smoke40.json`

Primary QA:

- baseline QA:
  - `EM 0.3250 / F1 0.4028`
- `bridge_append + CE`:
  - `EM 0.4500 / F1 0.5396`
- delta vs baseline in the same run:
  - `EM +0.1250 / F1 +0.1368`

This gives the clean V2 comparison target:

- typed-abstain:
  - `EM 0.4500 / F1 0.5271`
- bridge control:
  - `EM 0.4500 / F1 0.5396`
- delta:
  - `EM +0.0000 / F1 -0.0125`

## Paired Audit: Typed-Abstain V2 vs Bridge Control

Artifacts:

- `run_logs/2wiki_gap_expand_typedabstain_smoke40_paired_20260415.json`
- `run_logs/2wiki_gap_expand_typedabstain_smoke40_paired_20260415.md`

Overall:

- F1 win / tie / lose:
  - `1 / 37 / 2`
  - mean delta `-0.0125`
- EM win / tie / lose:
  - `1 / 38 / 1`
  - mean delta `0.0`
- selected-gap subset:
  - `5` queries
  - F1 `1 / 2 / 2`
- changed-evidence subset:
  - `6` queries
  - F1 `1 / 3 / 2`
- unchanged-evidence subset:
  - `34` queries
  - F1 `0 / 34 / 0`

Representative win:

- `Which film has the director who is older, God's Gift To Women or Aldri Annet Enn Bråk?`
  - `F1 0.0 -> 1.0`
  - added `Kenneth Branagh`
  - removed `Ingmar Bergman`

Representative regressions:

- `Which film has the director born later, Christ Walking On The Water or 45 Fathers?`
  - `F1 1.0 -> 0.0`
  - added `Roman Polanski`
  - removed `James Tinling`
- `What is the place of birth of the director of film The Return Of Swamp Thing?`
  - `F1 0.5 -> 0.0`
  - added `Roman Polanski`
  - removed `Jim Wynorski`

## Current V2 Diagnosis

V2 fixes the routing pathology but does not yet beat the bridge control:

1. the typed router is no longer degenerate
   - `role_relation` and `target_attribute` both fire
   - `abstain` fires on unclear cases
2. the abstaining fallback does reduce intervention rate
   - only `12.5%` of queries had a gap candidate enter final front
3. the residual failure is now localized
   - when V2 helps, it helps for the intended reason
   - when V2 hurts, it is still mostly a proposal semantics failure
   - the main bad alternate on this slice is `Roman Polanski`
4. on this smoke, V2 is close to bridge control but still slightly worse on F1

So the current typed-abstain conclusion is:

> V2 successfully repairs the V1 gap-typing collapse and activates abstaining fallback, but it does not yet outperform a same-slice `bridge_append + CE` control; the remaining error is proposal routing quality, not interface plumbing.

## V2 Bucketed Audit

Artifacts:

- `run_logs/2wiki_gap_expand_typedabstain_smoke40_paired_v2grouped_20260415.json`
- `run_logs/2wiki_gap_expand_typedabstain_smoke40_paired_v2grouped_20260415.md`

This follow-up audit only adds grouped summaries; it uses the same V2 and control reports as the paired audit above.

Grouped result:

- by `gap_type`:
  - `abstain`: `12` queries, all ties
  - `target_attribute`: `14` queries, all ties
  - `role_relation`: `14` queries, F1 `1 / 11 / 2`, mean delta `-0.0357`
- by `gap_slot`:
  - `director`: `14` queries, F1 `1 / 11 / 2`, mean delta `-0.0357`
  - all other typed slots were all ties on this smoke slice

This confirms the narrow next-step hypothesis:

> the remaining damage is concentrated in `role_relation / director`, not in `target_attribute`, not in `abstain`, and not in the CE-centered assemble path.

## V3 Implementation: Role-Relation Hardening Only

Date: 2026-04-15

This is the intentionally narrow V3 patch.

Goal:

- do not change controller logic
- do not change `target_attribute`
- do not change `bridge_entity`
- only harden `typed_abstain` under `role_relation`

Code changes:

- `scripts/eval_causal_qwen3.py`
  - added `_score_gap_witness_units(...)`
  - added `_role_relation_witness_tuple(...)`
  - tightened `role_relation` candidate admission inside `rerank_gap_expand_candidates(...)`
    - only local witness units (`title_plus_1sent` / `title_plus_2sent`) can satisfy the hard gate
    - the winning local witness must simultaneously show:
      - work anchor
      - role cue
      - non-anchor filler entity
  - added a current-role-witness reference veto inside `select_gap_expand_positions_typed_abstain(...)`
    - the typed alternate is rejected unless its local role witness is strictly better than the current best role witness already present in the scaffold / bridge-primary state
  - added trace fields for:
    - `gap_best_role_witness_joint`
    - `gap_best_role_anchor_slot`
    - `gap_best_role_non_anchor`
    - `gap_best_role_unit_type`
    - `role_reference_witness`
- `scripts/analyze_gap_expand_paired.py`
  - added grouped summaries:
    - `by_gap_type`
    - `by_gap_slot`
- `tests/test_setwise_selector.py`
  - added a role-relation rerank test that rejects topic-related pages lacking a valid local work-role witness
  - added a typed-abstain selection test that rejects a weaker `role_relation` alternate when the current role witness is already as strong or stronger

Validation:

- `env -u ALL_PROXY -u all_proxy -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy .venv-hipporag/bin/pytest tests/test_setwise_selector.py -q`
  - `167 passed`

## Smoke40 Result: 2Wiki Typed-Abstain V3

Artifacts:

- `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_gap_expand_bridge_append_plus_ce_qatopk5_20260415typedabstain_v3_smoke40.json`
- `run_logs/2wiki_gap_expand_typedabstain_v3_smoke40_paired_20260415.json`
- `run_logs/2wiki_gap_expand_typedabstain_v3_smoke40_paired_20260415.md`

Primary QA in the V3 run:

- baseline QA:
  - `EM 0.3500 / F1 0.4170`
- V3 typed-abstain:
  - `EM 0.4500 / F1 0.5396`

Compared against the same-slice bridge control:

- bridge control:
  - `EM 0.4500 / F1 0.5396`
- paired audit:
  - F1 `0 / 40 / 0`
  - EM `0 / 40 / 0`

Selector / trace summary:

- `gap_type_distribution = {"abstain": 12, "role_relation": 14, "target_attribute": 14}`
- `gap_expand_fallback_rate = 0.3`
- `avg_gap_candidate_count = 0.075`
- `avg_gap_candidates_selected_into_final = 0.0`
- `gap_candidate_selected_query_rate = 0.0`
- `append_stop_reason_counts = {"alternate_only_complete": 3, "append_cap_reached": 2, "no_typed_alternate": 4, "structure_below_threshold": 26, "typed_alternate_below_threshold": 5}`

Paired-audit interpretation:

- selected-gap subset:
  - `0` queries
- changed-evidence subset:
  - `2` queries
  - both ties
- by `gap_type`:
  - `role_relation`: all ties
  - `target_attribute`: all ties
  - `abstain`: all ties

What changed in V3 is not the final QA score. What changed is the intervention behavior:

- V2:
  - `selected_gap_rate = 0.125`
  - `role_relation/director` caused the only wins and the only losses
- V3:
  - `selected_gap_rate = 0.0`
  - all typed alternates were either blocked before final-front entry or lost in downstream CE

So the V3 hardening conclusion is:

> the `role_relation` hardening patch successfully removes the localized V2 regressions, but it does so by collapsing typed alternates back to a no-effect path relative to the same-slice `bridge_append + CE` control.

## Stop Condition

This was the intended narrow v3 stop test:

- if `role_relation` hardening produces wins over bridge control, keep the line alive
- if it only neutralizes losses by eliminating all effective alternates, stop the line

The smoke40 result falls into the second case.

Final status of the `gap_expand` line:

> `gap_expand` V1 failed because routing collapsed.
> `typed_abstain` V2 fixed routing collapse but still lost on `role_relation/director`.
> `role_relation` hardening V3 removed the losses, but only by returning to bridge-control-equivalent behavior.
> Under the current non-learned setup, this line does not justify further expansion.

## V4 Implementation: Unit-Typed-Abstain Gap Retrieval

Date: 2026-04-15

This is the first implementation of the new line:

> Scaffold-conditioned gap retrieval over finer evidence units.

The goal is explicitly not to revive controller logic. The goal is to move the
repair forward into proposal generation while preserving:

- `bridge_append`
- bridge-primary ownership
- downstream `bridge_append + CE` / actionized assemble

### What changed

In `scripts/eval_causal_qwen3.py`:

- added `gap_expand_mode="unit_typed_abstain"`
- extended `detect_gap_expand_state(...)`
  - `unit_typed_abstain` only admits:
    - `role_relation`
    - `target_attribute`
    - `abstain`
  - `bridge_entity` is disabled in this mode
  - unstable attribute anchors now abstain instead of silently re-routing bridge-style questions
- added finer evidence-unit helpers:
  - `build_gap_evidence_units(...)`
  - `rerank_gap_expand_units(...)`
  - `_gap_unit_rank_tuple(...)`
  - `_build_gap_doc_rows_from_positions(...)`
- added `select_gap_expand_positions_unit_typed_abstain(...)`
  - preserves the original bridge primary
  - builds `title + 1 sentence` / `title + 2 sentence` units over deep-pool docs
  - scores `micro-query × unit` with the existing CE when available
  - collapses unit scores back to parent docs
  - appends at most one unit-derived alternate
  - requires the alternate's best local witness to be strictly better than the current bridge primary witness
- bridge-append selector loading now reuses CE for proposal-side unit ranking when `gap_expand_mode=unit_typed_abstain`

### Trace additions

The new mode records unit-level diagnostics in the per-query selector trace:

- `gap_unit_count`
- `gap_unit_eligible_count`
- `gap_unit_score_source`
- `gap_unit_top_text`
- `gap_unit_top_parent_title`
- `gap_unit_top_parent_doc_id`
- `gap_unit_selected_parent_title`
- `gap_unit_selected_parent_doc_id`
- `gap_unit_selected_query_rate`
- `gap_unit_selected_query_score`
- `gap_unit_anchor_pass`
- `gap_unit_slot_pass`
- `gap_unit_non_anchor_pass`
- `gap_unit_value_pass`
- `gap_unit_selected_sentence_span`
- `gap_unit_bridge_primary_query_score`
- `gap_unit_bridge_primary_parent_title`
- `gap_unit_bridge_primary_parent_doc_id`

### Tests added

`tests/test_setwise_selector.py` now covers:

- evidence-unit construction with parent metadata and sentence spans
- `unit_typed_abstain` abstention on unstable bridge-style queries
- unit-level reranking that prefers a true local work-role witness over a topic-related person page
- preserving bridge primary while adding one unit-derived alternate
- rejecting a unit alternate when it is not strictly better than the current bridge-primary witness

Validation:

- `pytest tests/test_setwise_selector.py -k "unit_typed_abstain or gap_expand or build_gap_evidence_units" -q`
  - `11 passed`
- `env -u ALL_PROXY -u all_proxy -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy pytest tests/test_setwise_selector.py -q`
  - `172 passed`

### Current status

This completes the code path for the new unit-level expand repair line, but it
does not yet include a new smoke result.

The next step is intentionally narrow:

- offline paired audit first
- 2Wiki smoke40 only if unit-level separation is visible
- stop immediately if changed-evidence remains net negative

## V4 Smoke Result: `/no_think`-Hardened 2Wiki Smoke40

Date: 2026-04-15

### Why this rerun was necessary

The first `unit_typed_abstain` smoke run on local Qwen 8B was not a clean
method comparison because the reader path was still allowing explicit thinking
output. That introduced two practical problems:

- QA latency was much higher than expected
- retrieval-side chat completions emitted repeated parse warnings

So the experiment path was hardened before rerunning:

- `src/hipporag/llm/openai_gpt.py`
  - Qwen-family chat requests now automatically prefix the final user turn with
    `/no_think`
  - the prefix is applied before cache-key construction, so cache entries remain
    semantically correct
  - override is available via `HIPPORAG_QWEN_FORCE_NO_THINK=0`
- `tests/test_llm_request_name_alias.py`
  - added coverage for default `/no_think` injection
  - added coverage for explicit disable behavior

Validation:

- `env -u ALL_PROXY -u all_proxy -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy pytest tests/test_llm_request_name_alias.py tests/test_setwise_selector.py -q`
  - `177 passed`

### Rerun command

Candidate:

```bash
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
  .venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
  --dataset 2wikimultihopqa \
  --limit 40 \
  --llm_name qwen3-8b \
  --llm_request_name qwen3-8b-train \
  --llm_base_url http://localhost:8041/v1 \
  --embedding_name VLLM//mnt/nvme/Qwen3-Embedding-8B \
  --embedding_base_url http://localhost:8018/v1/embeddings \
  --causal_enabled false \
  --causal_engine_version v2 \
  --causal_v2_base_retrieval_mode legacy_fact_graph \
  --setwise_selector bridge_append \
  --setwise_pool_k 100 \
  --expand_base_k 10 \
  --append_max_docs 3 \
  --append_policy gap_expand \
  --gap_expand_mode unit_typed_abstain \
  --gap_expand_max_queries 2 \
  --assemble_mode cross_encoder \
  --qa_top_k 5 \
  --ce_device cuda:3 \
  --output_json outputs_step0_general_2wikimultihopqa/eval_reports/width_match_gap_expand_bridge_append_plus_ce_qatopk5_20260415unittypedabstain_smoke40_nothink.json
```

Fair control:

```bash
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
  .venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
  --dataset 2wikimultihopqa \
  --limit 40 \
  --llm_name qwen3-8b \
  --llm_request_name qwen3-8b-train \
  --llm_base_url http://localhost:8041/v1 \
  --embedding_name VLLM//mnt/nvme/Qwen3-Embedding-8B \
  --embedding_base_url http://localhost:8018/v1/embeddings \
  --causal_enabled false \
  --causal_engine_version v2 \
  --causal_v2_base_retrieval_mode legacy_fact_graph \
  --setwise_selector bridge_append \
  --setwise_pool_k 100 \
  --expand_base_k 10 \
  --append_max_docs 3 \
  --append_policy bridge \
  --assemble_mode cross_encoder \
  --qa_top_k 5 \
  --ce_device cuda:3 \
  --output_json outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260415smoke40_nothink.json
```

Paired audit:

```bash
python scripts/analyze_gap_expand_paired.py \
  --control_report outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260415smoke40_nothink.json \
  --candidate_report outputs_step0_general_2wikimultihopqa/eval_reports/width_match_gap_expand_bridge_append_plus_ce_qatopk5_20260415unittypedabstain_smoke40_nothink.json \
  --output_json run_logs/2wiki_gap_expand_unittypedabstain_smoke40_nothink_vs_bridge_nothink_paired_20260415.json \
  --output_md run_logs/2wiki_gap_expand_unittypedabstain_smoke40_nothink_vs_bridge_nothink_paired_20260415.md \
  --case_limit 8
```

### Observed runtime behavior

`/no_think` fixed the operational issue cleanly:

- retrieval no longer emitted the earlier cascade of parse-failed warnings
- reader throughput returned to normal
  - candidate run QA stage finished in about 19s for 40 queries
  - control rerun QA stage finished in about 18s for 40 queries

So `/no_think` is worth keeping as infrastructure hardening.

### Result summary

Candidate report:

- file:
  `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_gap_expand_bridge_append_plus_ce_qatopk5_20260415unittypedabstain_smoke40_nothink.json`
- primary QA:
  - `EM = 0.4000`
  - `F1 = 0.5176`
- gap summary:
  - `gap_type_distribution = {abstain: 12, role_relation: 14, target_attribute: 14}`
  - `fallback_rate = 0.30`
  - `selected_gap_count = 1`
  - `selected_gap_rate = 0.025`

Fair control report:

- file:
  `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260415smoke40_nothink.json`
- primary QA:
  - `EM = 0.4000`
  - `F1 = 0.5176`

Fair paired audit:

- file:
  `run_logs/2wiki_gap_expand_unittypedabstain_smoke40_nothink_vs_bridge_nothink_paired_20260415.json`
- shared queries: `40`
- overall:
  - F1: `0 win / 40 tie / 0 lose`
  - EM: `0 win / 40 tie / 0 lose`
  - mean delta: `0.0`
- subsets:
  - selected-gap subset: `1 query`, `0 win / 1 tie / 0 lose`
  - changed-evidence subset: `2 queries`, `0 win / 2 tie / 0 lose`

Note:

- an earlier paired comparison against the pre-`/no_think` bridge control showed
  an apparent negative delta
- that comparison is not method-clean and should not be used
- once the fair control is rerun under the same reader setting, the delta
  disappears entirely

### Interpretation

This rerun sharpens the conclusion.

`/no_think` was a real infrastructure bugfix, but it did not rescue the method.
After fixing the reader path and rerunning the proper fair control:

> `unit_typed_abstain` becomes behaviorally equivalent to same-slice
> `bridge_append + CE` on 2Wiki smoke40.

What changed with `/no_think`:

- system stability improved
- latency improved
- the comparison became trustworthy

What did not change:

- the method still does not create a measurable positive intervention
- the selected gap alternate rate remains tiny (`1/40`)
- no net win appears over fair bridge control

Final V4 status:

> Unit-level gap retrieval is a cleaner implementation than doc-level typed
> alternates, but under the current non-learned setup it still collapses to a
> no-effect path relative to fair `bridge_append + CE`.
