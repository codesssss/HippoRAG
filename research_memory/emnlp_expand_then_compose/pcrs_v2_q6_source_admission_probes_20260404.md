# PCRS-RAG V2 Q6 Source Admission Probes

Date: 2026-04-04

## Scope

This note follows the q6 source-admission audit.

Goal:

1. test an eval-only `support_bonus` source ranking probe
2. test a pure source-width control (`cutoff=16`)
3. keep parser, hop budget, shortlist objective, reserve, and final objective unchanged

Everything below uses the same widened live-frontier substrate as the 2026-04-03 smoke10 runs:

- `musique`
- `setwise_selector=requirement_beam`
- `setwise_requirement_live_annotation_pool_k=100`
- `setwise_requirement_live_annotation_score_mode=hybrid`
- `setwise_requirement_live_atomic_model_path=research_memory/emnlp_expand_then_compose/models/musique_need_unit_atomic_multiclass_hist_gbdt_pool100_limit10_bridge_seed.joblib`
- `openie_mode=online`

## Code Changes

Files touched:

- `scripts/eval_causal_qwen3.py`
- `tests/test_setwise_selector.py`

What changed:

1. Added an eval-only source-ranking probe path for `requirement_beam`
   - `--setwise_requirement_probe_source_sort_mode {combined,support_bonus}`
   - `--setwise_requirement_probe_source_support_gain_weight <float>`
2. Added source-trace fields:
   - `source_sort_mode`
   - `source_support_gain_weight`
   - `source_sort_score`
   - `base_score`
3. Preserved default behavior:
   - default mode is still `combined`
   - no behavior change unless the new probe flag is set
4. Added a focused test showing `support_bonus` can rescue a low-prior bridge candidate into source

## Verification

Commands:

```bash
.venv-hipporag/bin/python -m py_compile scripts/eval_causal_qwen3.py tests/test_setwise_selector.py
.venv-hipporag/bin/python -m pytest -q tests/test_setwise_selector.py
```

Result:

- `90 passed`

## Commands

### Probe B: support-aware source ranking

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
  --setwise_requirement_probe_source_sort_mode support_bonus \
  --setwise_requirement_probe_source_support_gain_weight 0.5 \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_source_support_bonus_w05_20260404.json
```

### Probe A: cutoff=16 control

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
  --setwise_requirement_live_source_expand_factor 4 \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_source_cutoff16_20260404.json
```

## Aggregate Results

Reference widened baseline:

- report:
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_frontier_widened_20260403.json`
- EM: `0.2`
- F1: `0.2733`
- Recall@5: `0.4083`

Support-bonus source ranking:

- report:
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_source_support_bonus_w05_20260404.json`
- EM: `0.2`
- F1: `0.2333`
- Recall@5: `0.4333`

Cutoff=16 control:

- report:
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_source_cutoff16_20260404.json`
- EM: `0.2`
- F1: `0.2733`
- Recall@5: `0.4083`

Interpretation:

- `support_bonus` changes source exposure, but it does not improve aggregate QA on smoke10
- pure width increase from `12 -> 16` does not move aggregate metrics at all
- so q6 is not just a capacity problem

## Q6 Mechanism Read

Question:

`Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`

### Baseline widened

- `Riverside Plaza`: `pool_only`
- `Minneapolis`: `pool_only`
- `Mississippi River`: `pool_only`
- answer: `Colorado River.`

### Support-bonus probe

Key watch-title movement:

- `Riverside Plaza`
  - `scored_rank=30/37`
  - `source_rank=0`
  - `support_completeness_gain=0.0`
  - still `pool_only`
- `Minneapolis`
  - `scored_rank=33/42`
  - `source_rank=0`
  - `support_completeness_gain=0.0`
  - still `pool_only`
- `Mississippi River`
  - `scored_rank=86`
  - `source_rank=4`
  - `shortlist_rank=1`
  - `combined_score=0.1411`
  - `source_sort_score=0.3212`
  - `support_completeness_gain=0.3602`
  - `utility_margin_gain=0.149`
  - reaches `selected`

Outcome:

- answer changes to `Gulf of Mexico.`
- q6 now surfaces one deep bridge doc without any forced-source probe
- but the earlier bridge docs (`Riverside Plaza`, `Minneapolis`) remain completely blocked

Interpretation:

> A light need-unit-aware source bonus can rescue the downstream body-of-water bridge, but it does not solve the earlier bridge chain. q6 still has a primary `pool -> source` failure on the upstream bridge docs.

### Cutoff=16 control

- `Riverside Plaza`: still `pool_only`
- `Minneapolis`: still `pool_only`
- `Mississippi River`: still `pool_only`
- answer stays `Colorado River.`

Interpretation:

> Width alone is not enough. Simply expanding source capacity from 12 to 16 does not rescue q6.

## Q7 Side Read

Question:

`When did the explorer reach the city where the headquarters of the only group larger than Vilaiyaadu Mankatha's record label is located?`

Support-bonus:

- `The Right Stuff Records`: `pool_only`
- `Sony Music`: `pool_only`
- answer stays wrong

Cutoff=16:

- `The Right Stuff Records`: `source`
- `Sony Music`: `source`
- both still fail to enter shortlist
- answer changes surface form, but the query still remains wrong and still lacks `Santa Monica, California` in pool

Interpretation:

- q7 is still not a good selector-driving case
- even when source exposure improves, q7 remains pool-coverage-limited and downstream-limited

## Bottom Line

This round supports a narrower conclusion than the earlier generic q6 audit:

> q6 source admission is not primarily a top-12 capacity problem. A need-unit-aware source bonus can selectively rescue the downstream bridge (`Mississippi River`), but upstream bridge docs with `support_completeness_gain≈0` remain invisible. So the next bottleneck is still source-stage signal design, not just source width.

Practical reading:

1. keep q6 as the main selector mechanism case
2. do not spend the next step on pure cutoff expansion
3. if continuing on selector, focus on why upstream bridge docs still receive zero source-stage need-unit gain
