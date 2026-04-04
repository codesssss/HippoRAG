# PCRS V2 q6 variable preservation probe - 2026-04-04

## What changed

- Fixed need-unit slot normalization so variable tokens are preserved instead of being flattened into plain text.
- Added `normalize_need_unit_slot_value()` in [scripts/requirement_beam_utils.py](/mnt/nvme/code/HippoRAG/scripts/requirement_beam_utils.py).
- Updated these paths to preserve / recover `?x/?y/?ans`:
  - `build_need_unit()`
  - `_is_variable_token()`
  - `_canonicalize_variable_token()`
  - `extract_need_unit_atomic_features()`

Design choice:

- Did **not** change global `normalize_structure_text()`.
- Kept the fix local to need-unit / atomic-scoring paths.
- Also made read-time canonicalization backward-compatible so legacy cache entries containing `x/y/ans` are treated as variables during feature extraction.

## Validation

Commands:

```bash
.venv-hipporag/bin/python -m py_compile \
  scripts/requirement_beam_utils.py \
  tests/test_setwise_selector.py

.venv-hipporag/bin/python -m pytest -q \
  tests/test_setwise_selector.py \
  tests/test_causal_utils.py
```

Result:

- `110 passed`

New focused tests cover:

- compiled need-units preserve `?x/?y/?ans`
- `build_need_unit()` preserves canonical variable slots
- atomic feature extraction recovers legacy serialized variables like `x`

## Static q6 check

Existing cache entry in
[musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_reuseopenie_bridge_seed_hybrid.json](/mnt/nvme/code/HippoRAG/research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_reuseopenie_bridge_seed_hybrid.json)
still stores:

```text
u_relation_hop_0 southeast library designer_of x ?x
u_relation_hop_1 x death_place y ?y
u_relation_hop_2 y empties_into ans ?ans
u_answer_slot_3 ans return_as_answer ans ?ans
```

Recompiling the same q6 plan now produces:

```text
u_relation_hop_0 southeast library designer_of ?x ?x
u_relation_hop_1 ?x death_place ?y ?y
u_relation_hop_2 ?y empties_into ?ans ?ans
u_answer_slot_3 ?ans return_as_answer ?ans ?ans
```

So future cache rebuilds will serialize variables correctly, while the new read path can already recover legacy cache entries.

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
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_cityalias_varfix_20260404.json
```

Report:

- [requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_cityalias_varfix_20260404.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_cityalias_varfix_20260404.json)

Compared against:

- [requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_cityalias_20260404.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_cityalias_20260404.json)

## q6 delta

Question:

- `Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`

Observed change:

- `Riverside Plaza`
  - before: `stage=source`
  - after: `stage=shortlist`
  - still `best_scored_rank=1`, `structure_score=1.0`
  - `support_completeness_gain` remains `0.0`
  - `utility_margin_gain` remains `0.0`
  - still not selected into final evidence
- `Minneapolis`
  - unchanged: `pool_only`
  - still zero gain
- `Mississippi River`
  - unchanged: `pool_only`
  - still zero gain

Overall smoke10 metrics were unchanged:

- baseline pipeline: `EM=0.3`, `F1=0.35`
- selector: `EM=0.2`, `F1=0.2733`, `Recall@5=0.4083`

## Interpretation

This fix was real and not cosmetic:

- preserving / recovering variable slots is enough to change live selector behavior on q6
- specifically, `Riverside Plaza` now survives one more stage and reaches shortlist

But it is **not sufficient**:

- no positive gain was unlocked
- q6 final evidence and answer are unchanged
- `Minneapolis` and `Mississippi River` remain unpromoted

So the best current reading is:

> Variable token flattening was a real implementation defect and part of the bottleneck, but after fixing it the remaining failure is still upstream-bridge utility blindness. The system can now see `Riverside Plaza` a bit further downstream, but still does not reward it with positive support / margin gain.

## Next step

- Rebuild the smoke cache when convenient so the persisted units also use `?x/?y/?ans`.
- For mechanism work, the next target should not be more source-sort tuning.
- The next likely bottleneck is still the atomic utility definition for upstream bridge / variable-binding evidence.
