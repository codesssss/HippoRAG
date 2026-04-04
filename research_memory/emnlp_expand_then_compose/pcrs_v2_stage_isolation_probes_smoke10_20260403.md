# PCRS-RAG V2 Stage-Isolation Probes Smoke10

Date: 2026-04-03

## Scope

This note extends the earlier live-frontier ablation with two tighter stage-isolation probes:

- `forced-source`: force selected watch titles into `candidate_source_preview`
- `forced-shortlist`: force selected watch titles into `candidate_shortlist_preview`

Everything else stayed fixed:

- parser / compiler unchanged
- conditional third-hop unchanged
- reserve unchanged
- Pareto / utopia / set-level objective unchanged
- no new retrieval method or learned model

The goal was to test whether the remaining bottleneck after frontier widening is:

- `pool -> source`
- `source -> shortlist`
- or `shortlist -> final`

## Commands

Both runs used the same substrate as the widened-frontier smoke:

- cache: `research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_reuseopenie_bridge_seed_hybrid.json`
- live atomic model: `research_memory/emnlp_expand_then_compose/models/musique_need_unit_atomic_multiclass_hist_gbdt_pool100_limit10_bridge_seed.joblib`
- watch titles: `Riverside Plaza,Minneapolis,Mississippi River,The Right Stuff Records,Sony Music`

Forced-source:

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
  --setwise_requirement_probe_force_source_titles 'Riverside Plaza,Minneapolis,Mississippi River,The Right Stuff Records,Sony Music' \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_forced_source_20260403.json
```

Forced-shortlist:

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
  --setwise_requirement_probe_force_shortlist_titles 'Riverside Plaza,Minneapolis,Mississippi River,The Right Stuff Records,Sony Music' \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_forced_shortlist_20260403.json
```

## Aggregate Summary

Baseline from the earlier note:

- baseline: EM `0.1`, F1 `0.2167`, Recall@5 `0.4583`
- widened: EM `0.2`, F1 `0.2733`, Recall@5 `0.4083`

Stage-isolation probes:

- forced-source: EM `0.2`, F1 `0.2333`, Recall@5 `0.4083`
- forced-shortlist: EM `0.2`, F1 `0.2733`, Recall@5 `0.4083`

Interpretation:

- widening still looks useful relative to clean baseline
- forcing source alone did not improve the widened result; it reduced F1
- forcing shortlist alone produced no aggregate change relative to widened

## Mechanism Cases

### Q6

Question:

`Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`

Watched titles:

- `Riverside Plaza`
- `Minneapolis`
- `Mississippi River`

Widened:

- all three stayed `pool_only`
- none reached source or shortlist

Forced-source:

- `Riverside Plaza`: `pool_only -> source`, but still not shortlist / final
- `Minneapolis`: `pool_only -> source`, but still not shortlist / final
- `Mississippi River`: `pool_only -> source -> shortlist -> selected`

Useful trace details:

- `Riverside Plaza`
  - source preview ranks: `13, 13`
  - best support gain: `0.0`
  - best utility margin gain: `0.0`
- `Minneapolis`
  - source preview ranks: `14, 14`
  - best support gain: `0.0`
  - best utility margin gain: `0.0`
- `Mississippi River`
  - shortlist preview rank: `1`
  - best support gain: `0.3602`
  - best utility margin gain: `0.149`

Forced-shortlist:

- no change from widened for q6
- because these titles still did not reach source under the non-forced-source path

Q6 conclusion:

> q6 is not just blocked at `pool -> source`. Even after forced source admission, two bridge titles still fail at `source -> shortlist`. Only the downstream body-of-water hop (`Mississippi River`) has enough atomic/set score to survive promotion.

Reader-side note:

- widened answer: `Colorado River.` with F1 `0.4`
- forced-source answer: `Gulf of Mexico.` with F1 `0.0`

So even partial bridge exposure did not improve the final answer on q6.

### Q7

Question:

`When did the explorer reach the city where the headquarters of the only group larger than Vilaiyaadu Mankatha's record label is located?`

Watched titles:

- `The Right Stuff Records`
- `Sony Music`

Widened:

- `The Right Stuff Records`: `source`
- `Sony Music`: `pool_only`
- neither entered shortlist or final evidence

Forced-source:

- `The Right Stuff Records`: stayed `source`, still not shortlist
- `Sony Music`: `pool_only -> source`, still not shortlist

Useful trace details:

- `The Right Stuff Records`
  - source preview ranks: `6, 5`
  - best support gain: `0.0292`
  - best utility margin gain: `-0.011`
- `Sony Music`
  - source preview ranks: `13, 14, 15, 16, 17, 13, 14, 15`
  - best support gain: `0.0544`
  - best utility margin gain: `-0.0215`

Forced-shortlist:

- `The Right Stuff Records`: `source -> shortlist`
- `The Right Stuff Records` still did **not** enter final selected evidence
- `Sony Music` still did not reach shortlist
- selector top titles remained unchanged
- answer remained `Not found in the provided texts.` with F1 `0.0`

Q7 conclusion:

> q7 now isolates a deeper bottleneck. `The Right Stuff Records` can be forced into shortlist, but final evidence selection still rejects it. So q7 has a `shortlist -> final` failure, not only an exposure failure.

## Updated Bottleneck Conclusion

The exposure-frontier hypothesis is supported, but it is not the whole story.

Current stage-isolation evidence points to at least three serial bottlenecks:

1. `pool -> source`
   - q6: `Riverside Plaza`, `Minneapolis`
   - q7: `Sony Music`
2. `source -> shortlist`
   - q6: `Riverside Plaza`, `Minneapolis`
   - q7 widened: `The Right Stuff Records`
3. `shortlist -> final`
   - q7 forced-shortlist: `The Right Stuff Records`

In short:

> the system is not only failing to expose bridge docs; it is also failing to promote some exposed bridge docs, and in at least one mechanism case it still refuses a bridge doc even after shortlist injection.

## Artifacts

- widened frontier report
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_frontier_widened_20260403.json`
- forced-source report
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_forced_source_20260403.json`
- forced-shortlist report
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_forced_shortlist_20260403.json`

## Next Step

The next minimal experiment should stay stage-local:

- keep parser / hop budget / objective fixed
- inspect why shortlist-injected bridge docs lose at final pick
- focus on the final-pick features that still assign weak utility to bridge docs already admitted into shortlist

This is now a cleaner target than broader scale-up.
