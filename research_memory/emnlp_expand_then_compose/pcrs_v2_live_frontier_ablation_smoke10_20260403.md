# PCRS-RAG V2 Live Frontier Ablation Smoke10

Date: 2026-04-03

## Scope

This note records a tightly scoped requirement-beam ablation on the V2 need-unit line.

Goal:

> test whether the current bottleneck is the live selector exposure frontier, not parser, hop budget, or set-level objective.

Boundaries kept fixed:

- parser / compiler unchanged
- conditional third-hop unchanged
- reserve / Pareto / utopia / set-level objective unchanged
- no new retrieval method, no HyDE, no iterative retrieval, no graph expansion

## Code Changes

Files touched:

- `scripts/eval_causal_qwen3.py`
- `tests/test_setwise_selector.py`
- existing passthrough update in `scripts/requirement_beam_utils.py`

What changed:

1. Added eval-only requirement-beam flags
   - `--setwise_requirement_live_annotation_pool_k`
   - `--setwise_requirement_live_annotation_score_mode`
   - `--setwise_requirement_live_atomic_model_path`
   - `--setwise_requirement_live_source_expand_factor`
   - `--setwise_requirement_exposure_watch_titles`

2. Live alignment now propagates atomic scorer bundle + score mode into rebuilt runtime annotations.

3. Added title exposure diagnostics for watched titles and gold titles:
   - `pool_only`
   - `source`
   - `shortlist`
   - `selected`

4. Kept widening conservative:
   - widened source frontier at eval time
   - but capped oracle shortlist back to `beam_expand_per_state`
   - this avoids turning oracle requirement-beam into a combinatorial sweep

## Verification

Commands:

```bash
python -m py_compile scripts/eval_causal_qwen3.py scripts/requirement_beam_utils.py tests/test_setwise_selector.py
.venv-hipporag/bin/python -m pytest -q tests/test_setwise_selector.py
```

Result:

- `82 passed`

## Smoke Commands

Baseline:

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
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_frontier_baseline_20260403.json
```

Widened frontier:

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
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_frontier_widened_20260403.json
```

Artifacts:

- baseline report
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_frontier_baseline_20260403.json`
- widened report
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_frontier_widened_20260403.json`

## Smoke Summary

Baseline:

- selector EM: `0.1`
- selector F1: `0.2167`
- selector Recall@5: `0.4583`
- avg support completeness: `0.2469`
- avg counterfactual leakage: `0.1304`

Widened:

- selector EM: `0.2`
- selector F1: `0.2733`
- selector Recall@5: `0.4083`
- avg support completeness: `0.2755`
- avg counterfactual leakage: `0.1289`

Interpretation:

- answer-side metrics improved
- recall@5 dropped
- this looks like a cleaner but more selective evidence frontier, not a pure recall win

## Mechanism Cases

### Q6

Question:

`Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`

Watched bridge docs:

- `Riverside Plaza`
- `Minneapolis`
- `Mississippi River`

Baseline:

- all three were `pool_only`
- none appeared in `candidate_source_preview`
- none appeared in `candidate_shortlist_preview`
- none were selected

Widened:

- still `pool_only`
- still absent from source
- still absent from shortlist
- still not selected

Conclusion for q6:

> widening the live frontier to source@12 with live hybrid annotation was not enough to expose the deep bridge chain.

This looks like an earlier proposal/source-ranking problem for this query, not a final-pick-only issue.

### Q7

Question:

`When did the explorer reach the city where the headquarters of the only group larger than Vilaiyaadu Mankatha's record label is located?`

Watched bridge docs:

- `The Right Stuff Records`
- `Sony Music`

Baseline:

- `The Right Stuff Records`: `pool_only`
- `Sony Music`: `pool_only`
- neither appeared in source or shortlist
- neither was selected

Widened:

- `The Right Stuff Records`: `source`
  - first seen at step `4`
  - seen again at step `5`
  - source ranks `6` and `5`
  - never promoted to shortlist
  - not selected
- `Sony Music`: still `pool_only`

Conclusion for q7:

> the widened frontier does surface at least one known deep bridge doc into candidate source, but it still dies before shortlist promotion.

## Bottom Line

The exposure-frontier hypothesis is **partially supported**.

What the ablation demonstrates:

1. parser is not the main issue
2. hop budget is not the main issue
3. the live frontier matters
   - `The Right Stuff Records` moved from `pool_only` to `source`
4. but the next bottleneck is now clearer:
   - for q7: `source -> shortlist` promotion
   - for q6: even `pool -> source` still fails under the current proposal ranking

Current best reading:

> the system is not mainly blocked at final pick. It is blocked first by exposure, and then by shortlist promotion once exposure begins to work.

So the next controlled step should target:

- shortlist promotion from widened source candidates
- not parser repair
- not hop-budget sweeps
- not set-level objective redesign
