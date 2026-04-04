# PCRS-RAG V2 Selector Ceiling Probe Under Fixed Pool

Date: 2026-04-03

## Scope

This note records the next selector-ceiling diagnostic on the V2 need-unit line.

Goal:

> measure how much headroom remains if the selector is made nearly perfect within the current pool, before spending more effort on q7 final-stage objective tuning.

Boundaries kept fixed:

- parser / compiler unchanged
- conditional third-hop unchanged
- reserve / Pareto / utopia / set-level objective unchanged
- same widened live frontier substrate as the previous smoke10 run
- no retrieval redesign and no new learned model

## Code Changes

Files touched:

- `scripts/eval_causal_qwen3.py`
- `tests/test_setwise_selector.py`

What changed:

1. Added an eval-only flag:
   - `--setwise_requirement_probe_force_pool_gold_into_final`

2. Added a helper to resolve per-query gold titles against the current pool:
   - `resolve_query_pool_gold_titles(...)`

3. Requirement-beam trace now records:
   - `probe_force_pool_gold_into_final`
   - `probe_force_pool_gold_titles_requested`
   - `probe_force_pool_gold_titles_in_pool`
   - `probe_force_pool_gold_titles_missing_from_pool`
   - `forced_pool_gold_final_titles_applied`
   - aggregate selector-summary counts for how often this path actually injected titles

4. Exposure summary now treats any of these as `forced_into_final`:
   - `forced_final_titles_applied`
   - `forced_probe_final_titles_applied`
   - `forced_pool_gold_final_titles_applied`

## Verification

Commands:

```bash
.venv-hipporag/bin/python -m py_compile scripts/eval_causal_qwen3.py tests/test_setwise_selector.py
.venv-hipporag/bin/python -m pytest -q tests/test_setwise_selector.py
```

Result:

- `89 passed`

## Commands

### Q7 multi-doc forced-final

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
  --setwise_requirement_probe_force_final_titles 'Vilaiyaadu Mankatha,The Right Stuff Records,Sony Music' \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_forced_final_q7_multidoc_20260403.json
```

### Smoke10 pool-gold forced-final ceiling

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
  --setwise_requirement_probe_force_pool_gold_into_final true \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_force_pool_gold_final_20260403.json
```

## Aggregate Results

Reference widened frontier run:

- report:
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_frontier_widened_20260403.json`
- EM: `0.2`
- F1: `0.2733`
- Recall@5: `0.4083`

Q7 multi-doc forced-final:

- report:
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_forced_final_q7_multidoc_20260403.json`
- EM: `0.2`
- F1: `0.2733`
- Recall@5: `0.4333`

Pool-gold forced-final ceiling:

- report:
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_force_pool_gold_final_20260403.json`
- EM: `0.2`
- F1: `0.38`
- Recall@5: `0.6667`
- Recall@2: `0.5917`
- forced-pool-gold enabled queries: `10/10`
- queries with at least one pool gold title present: `10/10`
- queries where new pool-gold titles were actually injected into final: `7/10`
- total injected pool-gold titles: `12`

Interpretation:

- within the fixed pool, there is still meaningful selector headroom on evidence quality
- that headroom is much larger on recall / F1 than on EM
- the ceiling is not uniformly high enough to justify focusing only on selector logic

## Mechanism Cases

### Q7 ceiling remains blocked by pool coverage

Question:

`When did the explorer reach the city where the headquarters of the only group larger than Vilaiyaadu Mankatha's record label is located?`

Pool-gold forced-final on q7:

- gold titles:
  - `Vilaiyaadu Mankatha`
  - `The Right Stuff Records`
  - `Sony Music`
  - `Santa Monica, California`
- pool-in gold titles:
  - `Vilaiyaadu Mankatha`
  - `The Right Stuff Records`
  - `Sony Music`
- missing from pool:
  - `Santa Monica, California`
- newly forced into final:
  - `The Right Stuff Records`
  - `Sony Music`
- final top titles:
  - `Vilaiyaadu Mankatha`
  - `The Right Stuff Records`
  - `Sony Music`
  - `Look What I Almost Stepped In...`
  - `News World India`
- selector answer stayed wrong:
  - widened: `Not found in the provided texts.`
  - pool-gold final: `No information provided.`

Conclusion for q7:

> Even near-perfect final exposure within the current pool is not enough. q7 is still blocked by missing pool coverage, specifically `Santa Monica, California`.

This makes q7 a poor target for more final-stage objective tuning right now.

### Q6 shows real within-pool selector headroom

Question:

`Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`

Pool-gold forced-final on q6:

- gold titles:
  - `Southeast Library`
  - `Riverside Plaza`
  - `Minneapolis`
  - `Mississippi River`
- all four gold titles were already in pool
- newly forced into final:
  - `Riverside Plaza`
  - `Minneapolis`
  - `Mississippi River`
- final top titles:
  - `Southeast Library`
  - `Riverside Plaza`
  - `Minneapolis`
  - `Mississippi River`
  - `Colorado River (Texas)`
- answer moved:
  - widened: `Colorado River.`
  - pool-gold final: `Mississippi River.`
- F1 improved:
  - `0.4 -> 0.8`

Conclusion for q6:

> q6 is still strongly exposure-limited. Once the missing bridge chain is surfaced from within the current pool, the reader gets much closer to the correct answer.

### One clean win and several stubborn failures

Notable pool-gold forced-final changes:

- `What year did the publisher of Labyrinth end?`
  - forced: `Acornsoft`
  - `EM/F1: 0/0 -> 1/1`
- `When was Lady Godiva's birthplace abolished?`
  - forced: `Mercia`
  - still wrong
- `When was the start of the battle of the birthplace of the performer of III?`
  - forced: `III (Stanton Moore album)`, `Battle of New Orleans`
  - still wrong

This split matters:

- some queries are still mainly selector/exposure problems inside the current pool
- others are already limited by missing gold docs or by reader use of the surfaced evidence

## Bottom Line

This probe supports a more precise conclusion:

> The exposure-frontier hypothesis is supported, but only partially. Under the current pool, selector improvements can still substantially improve evidence quality and some answers, but the fixed-pool ceiling is not high enough to make selector-only work the dominant next step for all mechanism queries.

Practical next-step reading:

- q6-like cases still justify more `pool -> source` / `source -> shortlist` work
- q7-like cases are already limited by pool coverage
- therefore the next bottleneck is not a single stage:
  - some queries still need better selector exposure
  - others need stronger base retriever / pool coverage before final-stage tuning matters
