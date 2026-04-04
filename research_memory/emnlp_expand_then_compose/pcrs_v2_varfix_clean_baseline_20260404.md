# PCRS-RAG V2 VarFix Clean Baseline

Date: 2026-04-04

## Scope

This note records the clean-baseline pass after promoting variable serialization to a correctness bugfix.

Boundaries kept fixed:

- no factual relation probe
- no continuity probe
- no bridge bonus
- no parser/compiler redesign
- no set-level objective change

The only intended semantic change is preserving variable slots as canonical `?x/?y/?ans` in the need-unit cache and downstream feature path.

## Artifacts

- smoke10 clean cache
  - `research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_varfix_clean.json`
- smoke10 clean report
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_varfix_clean_20260404.json`
- smoke40 clean cache
  - `research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit40_conditional_8043_varfix_clean.json`

## Commands Run

```bash
.venv-hipporag/bin/python scripts/build_need_unit_cache.py \
  --dataset musique \
  --limit 10 \
  --save_dir outputs_step0_general \
  --output_path research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_varfix_clean.json \
  --setwise_pool_k 100 \
  --annotation_pool_k 20 \
  --qa_top_k 5 \
  --llm_name qwen3-8b-train \
  --llm_base_url http://localhost:8043/v1 \
  --embedding_name VLLM//mnt/nvme/Qwen3-Embedding-8B \
  --embedding_base_url http://localhost:8018/v1/embeddings \
  --openie_mode online \
  --max_relation_hops 2 \
  --relation_hop_cap_mode conditional

.venv-hipporag/bin/python -m pytest -q tests/test_setwise_selector.py -k 'canonical_variable_slots or legacy_serialized_variables'

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
  --setwise_requirement_cache_path research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_varfix_clean.json \
  --setwise_requirement_mode oracle \
  --setwise_requirement_annotation_pool_k 20 \
  --setwise_requirement_reserve_policy fixed \
  --setwise_requirement_live_annotation_pool_k 100 \
  --setwise_requirement_live_annotation_score_mode hybrid \
  --setwise_requirement_live_atomic_model_path research_memory/emnlp_expand_then_compose/models/musique_need_unit_atomic_multiclass_hist_gbdt_pool100_limit10_bridge_seed.joblib \
  --setwise_requirement_live_source_expand_factor 3 \
  --setwise_requirement_exposure_watch_titles 'Southeast Library,Riverside Plaza,Minneapolis,Mississippi River,The Right Stuff Records,Sony Music' \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_varfix_clean_20260404.json

.venv-hipporag/bin/python scripts/build_need_unit_cache.py \
  --dataset musique \
  --limit 40 \
  --save_dir outputs_step0_general \
  --output_path research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit40_conditional_8043_varfix_clean.json \
  --setwise_pool_k 100 \
  --annotation_pool_k 20 \
  --qa_top_k 5 \
  --llm_name qwen3-8b-train \
  --llm_base_url http://localhost:8043/v1 \
  --embedding_name VLLM//mnt/nvme/Qwen3-Embedding-8B \
  --embedding_base_url http://localhost:8018/v1/embeddings \
  --openie_mode online \
  --max_relation_hops 2 \
  --relation_hop_cap_mode conditional
```

## Static Correctness Checks

### smoke10 cache

- entries: `10`
- positive need-units: `34`
- counterfactual need-units: `44`
- canonical variable slots observed:
  - subject vars: `54`
  - object vars: `78`
  - target vars: `78`
- bare legacy residues:
  - `subject=x/y/z/ans`: `0`
  - `object=x/y/z/ans`: `0`
  - `target_variable=x/y/z/ans`: `0`
  - `constraint.value=x/y/z/ans`: `0`

### q6 positive units in clean cache

- `u_relation_hop_0`: `southeast library --designer_of--> ?x`
- `u_relation_hop_1`: `?x --death_place--> ?y`
- `u_relation_hop_2`: `?y --empties_into--> ?ans`, with scope constraint `gulf of mexico`
- `u_answer_slot_3`: `?ans --return_as_answer--> ?ans`

### smoke40 cache

- entries: `40`
- positive need-units: `129`
- counterfactual need-units: `116`
- bare legacy residues:
  - `subject=x/y/z/ans`: `0`
  - `object=x/y/z/ans`: `0`
  - `target_variable=x/y/z/ans`: `0`
  - `constraint.value=x/y/z/ans`: `0`

### Dynamic safety check

- `tests/test_setwise_selector.py::test_build_need_unit_preserves_canonical_variable_slots`
- `tests/test_setwise_selector.py::test_extract_need_unit_atomic_features_recovers_legacy_serialized_variables`
- result: `2 passed`

This is enough to treat variable preservation as a correctness fix rather than an experimental probe.

## Smoke10 Result

### Aggregate

Compared against the prior no-probe bridge-seeded smoke10 run:

- old no-probe: EM `0.1`, F1 `0.2167`, Recall@5 `0.4583`
- varfix clean: EM `0.1`, F1 `0.24`, Recall@5 `0.4667`

Readout:

- EM is unchanged.
- F1 improves slightly.
- Recall@5 improves slightly.

So bugfix-only does produce natural end-to-end movement, but not a large one.

### q6

Question:

- `Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`

Observed under bugfix-only clean baseline:

- final selected titles:
  - `Southeast Library`
  - `Colorado River (Texas)`
  - `Gulf of Mexico`
  - `Mexico City`
  - `Charleston, South Carolina`
- exposure summary:
  - `Southeast Library`: `selected`
  - `Riverside Plaza`: `pool_only`
  - `Minneapolis`: `pool_only`
  - `Mississippi River`: `pool_only`

Interpretation:

- variable preservation alone does **not** push q6 bridge-gold docs into `candidate_source_preview`
- the q6 selector bottleneck remains upstream of shortlist/final
- q6 F1 rises from `0.0 -> 0.4`, but this is still not a bridge-chain closure win; the answer drifts from `Gulf of Mexico.` to `Colorado River.`

So the mechanism conclusion stays the same:

> bugfix-only makes the representation more correct, but does not by itself repair q6 exposure.

### q7

Question:

- `When did the explorer reach the city where the headquarters of the only group larger than Vilaiyaadu Mankatha's record label is located?`

Observed under bugfix-only clean baseline:

- final selected titles include:
  - `Vilaiyaadu Mankatha`
  - `The Right Stuff Records`
  - `Kathmandu`
- exposure summary:
  - `The Right Stuff Records`: `selected`
  - `Sony Music`: `source`
  - `Santa Monica, California`: `not_in_pool`

Interpretation:

- bugfix-only does create some natural downstream movement on q7-style bridge exposure
- but q7 remains pool-limited because `Santa Monica, California` is still missing

## Interim Conclusion

The variable serialization fix is real and should stay in the clean line.

What it already supports:

- the need-unit representation is now correct
- the atomic scorer can again see canonical variable slots
- bugfix-only gives a small smoke10 aggregate lift without adding any new probe logic

What it does **not** solve by itself:

- q6 still fails at exposure: `Riverside Plaza / Minneapolis / Mississippi River` remain `pool_only`
- therefore the next bottleneck is still not final-stage objective tuning

Current ablation order remains:

1. `bugfix only`
2. `bugfix + factual predicate probe`
3. `bugfix + factual predicate probe + bridge bonus`

## Smoke40 Result

Report:

- `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke40_conditional_bridge_seed_hybrid_varfix_clean_20260404.json`

Aggregate:

- selector EM `0.3000`
- selector F1 `0.3304`
- selector Recall@5 `0.5146`
- baseline EM `0.3750`
- baseline F1 `0.4042`

Important runtime note:

- `requirement_live_annotation_apply_count = 40`
- `requirement_live_source_expand_apply_count = 40`

So every smoke40 query triggered the ann20 -> ann100 live alignment path and the widened source frontier path during eval.

### q6 on smoke40

- selected titles:
  - `Southeast Library`
  - `Colorado River (Texas)`
  - `Gulf of Mexico`
  - `Mexico City`
  - `Charleston, South Carolina`
- exposure summary:
  - `Southeast Library`: `selected`
  - `Riverside Plaza`: `pool_only`
    - `best_scored_rank = 36`
    - `best_support_completeness_gain = 0.0`
    - `best_utility_margin_gain = 0.0`
  - `Minneapolis`: `pool_only`
    - `best_scored_rank = 40`
    - `best_support_completeness_gain = 0.0`
    - `best_utility_margin_gain = 0.0`
  - `Mississippi River`: `pool_only`
    - `best_scored_rank = 85`
    - `best_support_completeness_gain = 0.0`
    - `best_utility_margin_gain = 0.0`

Interpretation:

- smoke40 confirms the same bugfix-only conclusion seen in smoke10
- q6 bridge-gold docs are visible enough to get scores, but not enough to enter source
- variable preservation alone still does not repair q6 exposure

### q7 on smoke40

- selected titles:
  - `Look What I Almost Stepped In...`
  - `News World India`
  - `Vilaiyaadu Mankatha`
  - `Seattle`
  - `Love Around`
- exposure summary:
  - `The Right Stuff Records`: `shortlist`
    - `best_scored_rank = 6`
    - `best_support_completeness_gain = 0.0518`
    - `best_utility_margin_gain = -0.0001`
  - `Sony Music`: `pool_only`
    - `best_scored_rank = 14`
    - `best_support_completeness_gain = 0.0`
    - `best_utility_margin_gain = 0.0`
  - `Santa Monica, California`: `not_in_pool`

Interpretation:

- q7 still remains partly pool-limited
- unlike smoke10, bugfix-only no longer naturally pushes `Sony Music` to `source` on this larger slice
- `The Right Stuff Records` can reach `shortlist`, but not final evidence

## Consolidated Takeaway

Across both smoke10 and smoke40:

- variable serialization is correctly fixed
- bugfix-only produces some natural aggregate movement
- but it does not, by itself, unlock q6-style deep bridge exposure

So the fix should stay in the clean line, while the next mechanism work still belongs on:

1. exposure / source admission
2. factual predicate coverage
3. only then bridge-specific bonus logic
