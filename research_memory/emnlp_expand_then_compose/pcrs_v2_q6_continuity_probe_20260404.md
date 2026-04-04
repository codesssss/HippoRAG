# PCRS V2 q6 continuity probe (city/state alias) - 2026-04-04

## What changed

- Added an eval-only continuity probe flag: `--structure_continuity_probe_mode {off,city_state_alias}`.
- The probe is default-off and only augments structure retrieval objects when enabled.
- Probe behavior:
  - derives high-confidence `city state -> bare city` aliases from normalized structure entities
  - adds bare-city aliases into `doc_idx_to_structure_entities`
  - adds `alias_city_state` edges in both directions
  - duplicates existing structure edges onto alias endpoints
- Also fixed structure-graph triple normalization so entities entering `_prepare_structure_retrieval_objects()` use `normalize_structure_text()` rather than raw `text_processing()`. This was required because values like `Minneapolis, Minnesota` were otherwise stored as `minneapolis  minnesota`, which prevented alias matching.

Touched code:

- [src/hipporag/utils/config_utils.py](/mnt/nvme/code/HippoRAG/src/hipporag/utils/config_utils.py)
- [src/hipporag/utils/causal_utils.py](/mnt/nvme/code/HippoRAG/src/hipporag/utils/causal_utils.py)
- [src/hipporag/HippoRAG.py](/mnt/nvme/code/HippoRAG/src/hipporag/HippoRAG.py)
- [scripts/eval_causal_qwen3.py](/mnt/nvme/code/HippoRAG/scripts/eval_causal_qwen3.py)
- [tests/test_causal_utils.py](/mnt/nvme/code/HippoRAG/tests/test_causal_utils.py)

## Commands run

```bash
.venv-hipporag/bin/python -m py_compile \
  src/hipporag/HippoRAG.py \
  src/hipporag/utils/causal_utils.py \
  tests/test_causal_utils.py \
  tests/test_setwise_selector.py

.venv-hipporag/bin/python -m pytest -q \
  tests/test_causal_utils.py \
  tests/test_setwise_selector.py

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
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_cityalias_20260404.json
```

## Validation

- `py_compile`: pass
- `pytest`: `107 passed`

## Main result

Report:

- [requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_cityalias_20260404.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_cityalias_20260404.json)

Compared against relation-only probe:

- [requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_20260404.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_20260404.json)

Smoke10 headline:

- Baseline pipeline metrics unchanged: `ExactMatch=0.3`, `F1=0.35`
- Selector metrics unchanged vs relation-only:
  - `selector_EM=0.2`
  - `selector_F1=0.2733`
  - `selector Recall@5=0.4083`

## q6 mechanism delta

Question:

- `Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`

Observed q6 stage behavior with `q6_factual + city_state_alias`:

- `Riverside Plaza` remains:
  - `stage=source`
  - `best_scored_rank=1`
  - `source_preview_rank=1`
  - `structure_score=1.0`
  - `support_completeness_gain=0.0`
  - `utility_margin_gain=0.0`
  - never reaches shortlist/final
- `Minneapolis` remains:
  - `stage=pool_only`
  - `best_scored_rank=41`
  - `structure_score=0.0`
  - zero gain
- `Mississippi River` remains:
  - `stage=pool_only`
  - `best_scored_rank=84/90`
  - `structure_score=0.0`
  - zero gain

Relative to relation-only probe, the q6 watch trace is effectively unchanged except for tiny novelty/combined-score noise. No stage transition changed.

## Interpretation

This probe fixed a real normalization issue and is correct as a continuity audit, but continuity alone did not move the live selector.

Current best reading:

1. q6 is no longer blocked only by missing predicate coverage.
2. Narrow city/state canonicalization also does not unlock nonzero atomic gain on the real trace.
3. The next bottleneck is likely not source admission anymore, but atomic scoring / requirement coverage semantics.

Most important confounder now visible in the cache/scorer interface:

- q6 positive need-units are serialized as:
  - `southeast library -> designer_of -> x`
  - `x -> death_place -> y`
  - `y -> empties_into -> ans`
- In the cache, `subject/object` use `x/y/ans` rather than `?x/?y/?ans`.
- In [scripts/requirement_beam_utils.py](/mnt/nvme/code/HippoRAG/scripts/requirement_beam_utils.py), atomic feature extraction treats a field as variable-like only when it starts with `?`.
- So the live atomic scorer is likely still treating `x/y/ans` as literal tokens/entities instead of variables, which can keep bridge-style units at zero support even when continuity is better.

## Bottom line

The continuity probe does **not** support “canonicalization alone fixes q6”.

More precise conclusion:

> The relation probe fixed the right graph layer, and the continuity probe fixed a real node-normalization gap, but q6 still does not gain support. The remaining blocker is now more likely in the atomic need-unit supervision/scoring interface, with variable serialization mismatch (`x/y/ans` vs `?x/?y/?ans`) as the leading concrete suspect.

## Recommended next step

- Keep `city_state_alias` as an eval-only probe.
- Next, patch the atomic scorer path so `x/y/ans` are canonicalized to `?x/?y/?ans` before variable-sensitive feature extraction and re-run q6 first.
