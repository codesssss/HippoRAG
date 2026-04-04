# PCRS-RAG V2 Q7 Forced-Final / Positive-Only Probes

Date: 2026-04-03

## Scope

This note records the next two q7-focused probes after the stage-isolation study:

- `forced-final`: inject `The Right Stuff Records` directly into the final reader evidence set
- `positive_only` shortlist: rank requirement-beam shortlist candidates by positive gain first, instead of utility margin first

I also reran `forced-shortlist` once after adding richer watch-title / finalist diagnostics so the q7 failure mode could be compared with the new traces.

All runs kept the same substrate:

- parser / compiler unchanged
- conditional third-hop unchanged
- reserve / Pareto / utopia objective unchanged
- live annotation: `hybrid`
- live source expansion: `3`
- no retrieval redesign and no new learned model

## Code Changes

Files touched:

- `scripts/eval_causal_qwen3.py`
- `tests/test_setwise_selector.py`

What changed:

1. Added eval-only `requirement_beam` flags:
   - `--setwise_requirement_probe_force_final_titles`
   - `--setwise_requirement_probe_shortlist_sort_mode {margin_first,positive_only}`

2. Extended requirement-beam traces with:
   - `support_completeness_after`
   - `counterfactual_leakage_after`
   - `utility_margin_after`
   - dominant positive unit / dominant counterfactual diagnostics
   - `beam_watch_title_finalists`

3. Extended exposure summary with:
   - `final_only`
   - `forced_into_final`

## Verification

Commands:

```bash
python -m py_compile scripts/eval_causal_qwen3.py tests/test_setwise_selector.py
.venv-hipporag/bin/python -m pytest -q tests/test_setwise_selector.py
```

Result:

- `87 passed`

## Commands

Forced-final:

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
  --setwise_requirement_probe_force_final_titles 'The Right Stuff Records' \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_forced_final_q7_20260403.json
```

Positive-only shortlist:

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
  --setwise_requirement_probe_shortlist_sort_mode positive_only \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_positive_only_shortlist_20260403.json
```

Forced-shortlist rerun for richer diagnostics:

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
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_forced_shortlist_rerun_20260403.json
```

## Aggregate Results

Reference:

- widened frontier: EM `0.2`, F1 `0.2733`, Recall@5 `0.4083`

New probes:

- forced-final q7 title: EM `0.1`, F1 `0.1733`, Recall@5 `0.4333`
- positive-only shortlist: EM `0.2`, F1 `0.2667`, Recall@5 `0.4583`
- forced-shortlist rerun: EM `0.2`, F1 `0.2733`, Recall@5 `0.4083`

Interpretation:

- forced-final clearly did **not** help; it hurt overall
- positive-only changed shortlist composition and Recall@5, but did not improve q7
- forced-shortlist reproducibly leaves q7 answer unchanged

## Q7 Mechanism Summary

Question:

`When did the explorer reach the city where the headquarters of the only group larger than Vilaiyaadu Mankatha's record label is located?`

### 1. Forced-final result

`The Right Stuff Records` moved to:

- stage: `final_only`
- final evidence: yes
- selected set: no

Reader evidence became:

- `The Right Stuff Records`
- `Look What I Almost Stepped In...`
- `News World India`
- `Vilaiyaadu Mankatha`
- `Love Around`

Reader output stayed wrong:

- selector answer: `that the information is not provided.`
- selector F1: `0.0`

Conclusion:

> Injecting `The Right Stuff Records` directly into the final reader evidence set is not sufficient to recover q7.

This argues against the simple hypothesis:

> “the only problem is that the selector never lets the bridge doc reach the reader.”

### 2. Positive-only shortlist result

Even with support-first shortlist sorting:

- `The Right Stuff Records` stayed at stage `source`
- it still did **not** enter shortlist
- q7 answer stayed wrong

The strongest q7 watch rows still looked weak:

- step 5 `The Right Stuff Records`
  - `support_completeness_gain = 0.0292`
  - `counterfactual_leakage_gain = 0.0401`
  - `utility_margin_gain = -0.011`
  - dominant positive unit: `u_relation_hop_1`
  - dominant predicate: `headquartered_in`
  - dominant unit was already partially covered:
    - `coverage_before = 0.5801`
    - `coverage_after = 0.6891`
    - `gain = 0.1091`
  - dominant counterfactual also rose:
    - `cf_1 before = 0.3013`
    - `cf_1 after = 0.3618`
    - `gain = 0.0605`

This is consistent with the earlier mechanism suspicion:

> `The Right Stuff Records` mostly improves an already partially covered downstream unit, while its counterfactual side rises at the same time, so margin stays non-positive.

### 3. Forced-shortlist rerun with finalist diagnostics

This rerun shows the most useful new detail:

- `The Right Stuff Records` does reach `shortlist`
- it also reaches a real finalist:
  - best containing finalist rank: `3`
  - selected titles:
    - `Look What I Almost Stepped In...`
    - `News World India`
    - `Vilaiyaadu Mankatha`
    - `Love Around`
    - `The Right Stuff Records`

But that finalist still loses to the best state:

- best containing finalist:
  - `support_completeness = 0.4781`
  - `counterfactual_leakage = 0.5186`
  - `utility_margin = -0.0405`
  - `utopia_distance = 0.7358`
- delta vs best state:
  - `support_completeness: +0.0291`
  - `counterfactual_leakage: +0.0400`
  - `utility_margin: -0.0109`
  - `utopia_distance: +0.0060`

Conclusion:

> On q7, the bridge doc is not merely dying before shortlist. It can survive into a finalist, but the finalist containing it is still slightly worse on the final objective because leakage grows more than support.

This is a cleaner statement than the earlier generic “shortlist -> final” diagnosis:

> q7 is now a **finalist-comparison / final-objective** problem, not just an exposure problem.

## Updated Conclusion

The two new probes point to three concrete takeaways:

1. `forced-final` failing means:
   - bridge-doc exposure to the reader is **not** by itself enough
   - either the bridge doc is insufficient alone, or the reader does not know how to use this bridge evidence in the current evidence mixture

2. `positive_only` failing means:
   - q7 is **not** mainly caused by margin-first shortlist sorting
   - support-first ranking still does not naturally promote `The Right Stuff Records`

3. `forced-shortlist` finalist diagnostics show:
   - a finalist containing `The Right Stuff Records` does exist
   - but it loses because the final objective sees slightly larger leakage growth than support gain

So the next bottleneck is no longer best described as:

> “the doc never gets exposed.”

It is now better described as:

> `The Right Stuff Records` can be exposed and can even reach a finalist, but under the current support-vs-leakage objective it remains a slightly worse state.

## Artifacts

- forced-final q7 report
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_forced_final_q7_20260403.json`
- positive-only shortlist report
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_positive_only_shortlist_20260403.json`
- forced-shortlist rerun report
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_forced_shortlist_rerun_20260403.json`

## Next Step

The next probe should stay very small and stay on q7:

- keep parser / retrieval / shortlist source frontier fixed
- inspect whether a tiny final-stage objective probe can reward bridge contribution without globally weakening the counterfactual axis

At this point, bigger-scale runs are still premature.
