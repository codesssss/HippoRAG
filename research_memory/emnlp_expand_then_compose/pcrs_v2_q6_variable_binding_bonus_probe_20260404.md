# PCRS V2 q6 variable-binding bridge bonus probe - 2026-04-04

## What changed

- Added a default-off eval-only bridge bonus path for `requirement_beam`.
- New CLI flags in [scripts/eval_causal_qwen3.py](/mnt/nvme/code/HippoRAG/scripts/eval_causal_qwen3.py):
  - `--setwise_requirement_probe_bridge_bonus_mode {off,variable_binding}`
  - `--setwise_requirement_probe_bridge_bonus_weight <float>`
- Wired candidate-specific positive-score overrides through:
  - [scripts/requirement_beam_utils.py](/mnt/nvme/code/HippoRAG/scripts/requirement_beam_utils.py)
    - `compute_requirement_state_metrics(...)`
    - `compute_requirement_candidate_feature_rows(...)`
    - `merge_positive_score_overrides_by_position(...)`
  - [scripts/eval_causal_qwen3.py](/mnt/nvme/code/HippoRAG/scripts/eval_causal_qwen3.py)
    - `select_requirement_beam_positions(...)`
- Exposed bridge-bonus diagnostics in:
  - `candidate_source_preview`
  - `candidate_shortlist_preview`
  - `watch_title_trace`

Design scope:

- default behavior unchanged
- no parser / hop budget / set-level objective changes
- no retrieval redesign

## Validation

Commands:

```bash
.venv-hipporag/bin/python -m py_compile \
  scripts/requirement_beam_utils.py \
  scripts/eval_causal_qwen3.py \
  tests/test_setwise_selector.py

.venv-hipporag/bin/python -m pytest -q \
  tests/test_setwise_selector.py \
  tests/test_causal_utils.py
```

Result:

- `111 passed`

New focused test:

- `test_requirement_beam_variable_binding_bonus_can_promote_upstream_bridge`

It verifies:

- `bridge_bonus_mode=off` keeps the upstream bridge at zero gain
- `bridge_bonus_mode=variable_binding` gives it positive gain and shortlist promotion

## Runtime probe

Command:

```bash
.venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
  --dataset musique \
  --limit 10 \
  --save_dir outputs_step0_general \
  --llm_name qwen3-8b-train \
  --llm_base_url http://localhost:8043/v1 \
  --embedding_name VLLM//mnt/nvme/Qwen3-Embedding-8B \
  --embedding_base_url http://localhost:8018/v1/embeddings \
  --openie_mode online \
  --causal_enabled false \
  --causal_engine_version v2 \
  --causal_v2_graph_mode causal \
  --causal_v2_base_retrieval_mode legacy_fact_graph \
  --setwise_selector requirement_beam \
  --setwise_score_mode bridge \
  --setwise_pool_k 100 \
  --qa_top_k 5 \
  --setwise_anchor_count 2 \
  --setwise_reserve_top_m 3 \
  --setwise_non_anchor_title_dedup true \
  --setwise_beam_width 4 \
  --setwise_beam_expand_per_state 4 \
  --setwise_beam_projected_shortlist_factor 1 \
  --setwise_requirement_cache_path research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_reuseopenie_bridge_seed_hybrid.json \
  --setwise_requirement_mode oracle \
  --setwise_requirement_annotation_pool_k 20 \
  --setwise_requirement_reserve_policy fixed \
  --setwise_requirement_live_annotation_pool_k 100 \
  --setwise_requirement_live_annotation_score_mode hybrid \
  --setwise_requirement_live_atomic_model_path research_memory/emnlp_expand_then_compose/models/musique_need_unit_atomic_multiclass_hist_gbdt_pool100_limit10_bridge_seed.joblib \
  --setwise_requirement_live_source_expand_factor 3 \
  --structure_relation_probe_mode q6_factual \
  --structure_continuity_probe_mode city_state_alias \
  --setwise_requirement_probe_bridge_bonus_mode variable_binding \
  --setwise_requirement_probe_bridge_bonus_weight 0.6 \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_cityalias_varfix_varbinding06_20260404.json
```

Report:

- [requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_cityalias_varfix_varbinding06_20260404.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_cityalias_varfix_varbinding06_20260404.json)

Compared against:

- [requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_cityalias_varfix_20260404.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_cityalias_varfix_20260404.json)

## q6 mechanism result

Question:

- `Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`

Observed delta on `Riverside Plaza`:

- before:
  - `best_support_completeness_gain = 0.0`
  - `best_utility_margin_gain = 0.0`
  - `shortlist_preview_rank = 2`
  - no bridge bonus fields
- after:
  - `bridge_bonus_applied = true`
  - `bridge_bonus_unit_id = u_relation_hop_1`
  - `bridge_bonus_unit_predicate = death_place`
  - `bridge_bonus_score = 0.225`
  - `support_completeness_gain = 0.0552`
  - `utility_margin_gain = 0.0552`
  - `shortlist_preview_rank = 1`

Interpretation:

- This is the first clean evidence that the system now does more than merely expose `Riverside Plaza`.
- The probe made it count as useful upstream bridge evidence for q6.

Still unchanged:

- `Riverside Plaza` still does **not** enter final selected evidence
- `Minneapolis` stays `pool_only`
- `Mississippi River` stays `pool_only`
- q6 final answer remains wrong / unchanged

So the probe fixes a real scoring blind spot, but only for the first upstream bridge.

## Smoke10 aggregate

Compared to the prior `varfix` run:

- previous selector: `EM=0.2`, `F1=0.2733`, `Recall@5=0.4083`
- variable-binding probe: `EM=0.1`, `F1=0.1333`, `Recall@5=0.4083`

This aggregate result is negative.

## Why the aggregate should be read cautiously

The bridge bonus only actually fired on q6 in this smoke run:

- q6 had `bridge_bonus_applied=true`
- other watched mechanism queries remained without bonus activation

At the same time, non-q6 evidence sets still drifted between runs. For example:

- the Messi query changed its selected evidence set even though no bridge bonus fired there

So the smoke10 aggregate is not a clean measure of the local q6 mechanism improvement. The online eval path still has enough run-to-run instability that this probe should be read mainly as a mechanism test, not a promotion candidate.

## Conclusion

The exposure/continuity work was not wasted. After relation coverage, continuity, and variable preservation, this probe finally showed:

> q6 `Riverside Plaza` is not just visible; it can now receive positive atomic utility through a variable-binding-aware bridge bonus.

But the result also shows the remaining bottlenecks clearly:

- `source -> shortlist` for `Riverside Plaza` is now partly solved
- `shortlist -> final` is still unsolved for q6
- deeper chain docs (`Minneapolis`, `Mississippi River`) still never activate

So the current status is:

> The variable-binding bonus supports the upstream-bridge blind-spot hypothesis at q6 mechanism level, but it is too local and too unstable to justify mainline adoption.

## Next step

- Keep this path experimental only.
- Do not promote it to the clean baseline.
- If continuing the q6 line, the next useful probe should focus on why positive utility at `u_relation_hop_1 / death_place` does not propagate into final evidence, and why `Minneapolis` / `Mississippi River` still never become eligible for the same bridge mechanism.
