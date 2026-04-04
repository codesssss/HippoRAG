# PCRS-RAG V2 VarFix Offline-Hybrid Q6 Matrix

Date: 2026-04-04

## Scope

This note records the narrow `A/B/C` smoke10 matrix requested after promoting variable serialization to a correctness fix.

Boundaries kept fixed:

- no live annotation widening
- no live source widening
- no parser/compiler redesign
- no hop-budget change
- no reserve / Pareto / utopia / set-level objective change
- no bridge bonus

The only moving parts in this matrix are:

- `A`: `varfix + offline-hybrid clean baseline`
- `B`: `A + --structure_relation_probe_mode q6_factual`
- `C`: `B + --structure_continuity_probe_mode city_state_alias`

## Artifacts

### Offline-hybrid cache

- cache
  - `research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_varfix_offlinehybrid_clean.json`
- top-level metadata
  - `annotation_score_mode = hybrid`
  - `atomic_model_path = research_memory/emnlp_expand_then_compose/models/musique_need_unit_atomic_multiclass_hist_gbdt_pool100_limit10_bridge_seed.joblib`
- per-doc annotation metadata
  - `score_mode = hybrid`
  - `scorer_version = need_unit_atomic_v1`

### Reports

- A
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_offlinehybrid_varfix_clean_a_20260404.json`
- B
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_offlinehybrid_varfix_clean_b_q6factual_20260404.json`
- C
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_offlinehybrid_varfix_clean_c_q6factual_cityalias_20260404.json`

## Commands Run

```bash
.venv-hipporag/bin/python scripts/build_need_unit_cache.py \
  --dataset musique \
  --limit 10 \
  --save_dir outputs_step0_general \
  --output_path research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_varfix_offlinehybrid_clean.json \
  --setwise_pool_k 100 \
  --annotation_pool_k 20 \
  --qa_top_k 5 \
  --llm_name qwen3-8b-train \
  --llm_base_url http://localhost:8043/v1 \
  --embedding_name VLLM//mnt/nvme/Qwen3-Embedding-8B \
  --embedding_base_url http://localhost:8018/v1/embeddings \
  --openie_mode online \
  --max_relation_hops 2 \
  --relation_hop_cap_mode conditional \
  --annotation_score_mode hybrid \
  --atomic_model_path research_memory/emnlp_expand_then_compose/models/musique_need_unit_atomic_multiclass_hist_gbdt_pool100_limit10_bridge_seed.joblib

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
  --setwise_requirement_cache_path research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_varfix_offlinehybrid_clean.json \
  --setwise_requirement_mode oracle \
  --setwise_requirement_annotation_pool_k 20 \
  --setwise_requirement_reserve_policy fixed \
  --setwise_requirement_exposure_watch_titles 'Southeast Library,Riverside Plaza,Minneapolis,Mississippi River,The Right Stuff Records,Sony Music' \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_offlinehybrid_varfix_clean_a_20260404.json

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
  --setwise_requirement_cache_path research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_varfix_offlinehybrid_clean.json \
  --setwise_requirement_mode oracle \
  --setwise_requirement_annotation_pool_k 20 \
  --setwise_requirement_reserve_policy fixed \
  --structure_relation_probe_mode q6_factual \
  --setwise_requirement_exposure_watch_titles 'Southeast Library,Riverside Plaza,Minneapolis,Mississippi River,The Right Stuff Records,Sony Music' \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_offlinehybrid_varfix_clean_b_q6factual_20260404.json

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
  --setwise_requirement_cache_path research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_varfix_offlinehybrid_clean.json \
  --setwise_requirement_mode oracle \
  --setwise_requirement_annotation_pool_k 20 \
  --setwise_requirement_reserve_policy fixed \
  --structure_relation_probe_mode q6_factual \
  --structure_continuity_probe_mode city_state_alias \
  --setwise_requirement_exposure_watch_titles 'Southeast Library,Riverside Plaza,Minneapolis,Mississippi River,The Right Stuff Records,Sony Music' \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_offlinehybrid_varfix_clean_c_q6factual_cityalias_20260404.json

.venv-hipporag/bin/python -m pytest -q tests/test_setwise_selector.py -k 'offline_hybrid_annotation_mode or canonical_variable_slots or legacy_serialized_variables or parser_trace_across_fallback'
```

## Aggregate Results

| Variant | Selector EM | Selector F1 | Recall@5 | live_annotation_apply | live_source_expand_apply |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: varfix + offline-hybrid | 0.1000 | 0.2167 | 0.4417 | 0 | 0 |
| B: A + q6_factual | 0.1000 | 0.2567 | 0.4667 | 0 | 0 |
| C: B + city_state_alias | 0.1000 | 0.2567 | 0.4667 | 0 | 0 |

Readout:

- The matrix stayed clean: all three runs kept `requirement_live_annotation_apply_count = 0` and `requirement_live_source_expand_apply_count = 0`.
- `q6_factual` gives a real but modest smoke10 lift: `F1 +0.0400`, `Recall@5 +0.0250`.
- Adding `city_state_alias` on top of `q6_factual` does not move aggregate smoke10 beyond B.

## Q6 Stage Readout

Question:

- `Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`

### A: varfix + offline-hybrid

- final selector titles:
  - `Southeast Library`
  - `Colorado River (Texas)`
  - `Gulf of Mexico`
  - `The Hague City Hall`
  - `Davenport Public Library`
- watch titles:
  - `Riverside Plaza`: `pool_only`, `best_scored_rank=36`, `best_combined_score=0.1647`, `support_gain=0.0`, `margin_gain=0.0`
  - `Minneapolis`: `pool_only`, `best_scored_rank=40`, `best_combined_score=0.1629`, `support_gain=0.0`, `margin_gain=0.0`
  - `Mississippi River`: `pool_only`, `best_scored_rank=85`, `best_combined_score=0.1410`, `support_gain=0.0`, `margin_gain=0.0`
- answer:
  - selector: `Gulf of Mexico.` with `F1=0.0`

### B: + q6_factual

- final selector titles:
  - `Southeast Library`
  - `Colorado River (Texas)`
  - `Gulf of Mexico`
  - `Riverside Plaza`
  - `The Hague City Hall`
- watch titles:
  - `Riverside Plaza`: `selected`, source rank `1`, shortlist rank `1`, `best_combined_score=0.7647`, `support_gain=0.1191`, `margin_gain=0.0449`
  - `Minneapolis`: still `pool_only`, `best_scored_rank=40`, `support_gain=0.0`
  - `Mississippi River`: still `pool_only`, `best_scored_rank=85`, `support_gain=0.0`
- answer:
  - selector: `Colorado River.` with `F1=0.4`

### C: + q6_factual + city_state_alias

- final selector titles:
  - `Southeast Library`
  - `Colorado River (Texas)`
  - `Gulf of Mexico`
  - `Riverside Plaza`
  - `The Hague City Hall`
- watch titles:
  - `Riverside Plaza`: `selected`, source rank `1`, shortlist rank `1`, `best_combined_score=0.7658`, `support_gain=0.1191`, `margin_gain=0.0449`
  - `Minneapolis`: still `pool_only`, `best_scored_rank=42`, `support_gain=0.0`
  - `Mississippi River`: still `pool_only`, `best_scored_rank=84`, `support_gain=0.0`
- answer:
  - selector: `Colorado River.` with `F1=0.4`

Q6 readout:

- `q6_factual` clearly fixes one earlier layer: `Riverside Plaza` is no longer invisible and now enters `source -> shortlist -> final`.
- The remaining q6 bottleneck is not source admission for `Riverside Plaza`; it is that the chain still does not continue through `Minneapolis / Mississippi River`.
- The narrow `city_state_alias` continuity probe does not unlock `Minneapolis` or `Mississippi River` in this clean offline-hybrid path.

## Q7 Sanity Check

Question:

- `When did the explorer reach the city where the headquarters of the only group larger than Vilaiyaadu Mankatha's record label is located?`

Across A/B/C:

- `The Right Stuff Records` stays `pool_only` with `best_scored_rank=5`, `support_gain=0.0`
- `Sony Music` stays `pool_only`
  - A/B: `best_scored_rank=13`
  - C: `best_scored_rank=16`
- selector answer remains `Not enough information.`

Readout:

- This matrix does not change the prior q7 conclusion.
- q7 remains a poor selector-driving case here; nothing in this narrow q6-focused probe changes its pool-limited character.

## Conclusion

The clean A/B/C result supports the current priority order:

1. `varfix` stays on mainline as a correctness fix.
2. `q6_factual` is still a necessary q6-specific mechanism repair.
3. `city_state_alias` did not add anything on top of `q6_factual` in this smoke10 clean path.

The main mechanism conclusion is now:

> In the clean offline-hybrid setting, q6 is no longer blocked purely by missing factual structure edges. `q6_factual` is sufficient to rescue `Riverside Plaza`, but the chain still fails to propagate through `Minneapolis / Mississippi River`. So the remaining q6 bottleneck is downstream of the repaired `Riverside Plaza` admission step and upstream of full bridge-chain closure.

This points back to the user-approved next priority:

- keep `varfix` in the clean line
- keep bridge bonus experimental only
- return q6 analysis to `exposure / source admission + factual predicate coverage`, not final-stage objective tuning
