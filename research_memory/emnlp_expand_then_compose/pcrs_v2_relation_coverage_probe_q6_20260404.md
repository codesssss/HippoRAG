# PCRS-RAG V2 Relation Coverage Probe for Q6

Date: 2026-04-04

## Goal

Test the minimal, eval-only hypothesis:

> if q6 bridge predicates are promoted into the structure graph, does q6 stop failing at `pool -> source`, and does smoke10 improve without touching parser, hop budget, or set-level objective?

This probe keeps default behavior unchanged and only activates under:

- `--structure_relation_probe_mode q6_factual`

## Code Change

Added a flag-gated directed-predicate probe that extends structure-graph coverage for a narrow q6-focused factual family:

- attribution / creation
  - `designed by`
  - `created by`
  - `built by`
- spatial / flow
  - `opened in`
  - `lies on`
  - `drains into`
  - `flows into`
  - `empties into`

Implementation points:

- `src/hipporag/utils/config_utils.py`
- `src/hipporag/utils/causal_utils.py`
- `src/hipporag/HippoRAG.py`
- `scripts/eval_causal_qwen3.py`
- `tests/test_causal_utils.py`

Default mode remains `off`.

## Verification Commands

Syntax + focused tests:

```bash
.venv-hipporag/bin/python -m py_compile \
  src/hipporag/utils/config_utils.py \
  src/hipporag/utils/causal_utils.py \
  src/hipporag/HippoRAG.py \
  scripts/eval_causal_qwen3.py \
  tests/test_causal_utils.py

.venv-hipporag/bin/python -m pytest -q \
  tests/test_causal_utils.py \
  tests/test_setwise_selector.py
```

Result:

- `py_compile`: pass
- `pytest`: `106 passed`

Smoke10 probe run:

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
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_20260404.json
```

Baseline for comparison:

- `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_frontier_widened_20260403.json`

Probe report:

- `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_relation_probe_q6factual_20260404.json`

## Local Graph Audit After Probe

On the exact q6 gold docs identified in the current workspace:

- `doc_idx=133` `Southeast Library`
  - `('ralph rapson', 'southeast library', 0.85, 'factual_attribution')`
- `doc_idx=119` `Riverside Plaza`
  - `('ralph rapson', 'riverside plaza', 0.85, 'factual_attribution')`
  - `('riverside plaza', 'minneapolis minnesota', 0.8, 'factual_located_in')`
- `doc_idx=124` `Minneapolis`
  - `('minneapolis', 'mississippi river', 0.8, 'factual_located_on')`
- `doc_idx=120` `Mississippi River`
  - `('mississippi river', 'gulf of mexico', 0.85, 'factual_flows_to')`

So the probe does add the missing q6 factual edges into the structure graph.

## Smoke10 Outcome

Aggregate metrics:

- widened baseline selector:
  - `EM=0.2000`
  - `F1=0.2733`
  - `Recall@5=0.4083`
- relation probe selector:
  - `EM=0.2000`
  - `F1=0.2733`
  - `Recall@5=0.4083`

So the probe did **not** change smoke10 aggregate selector metrics.

## Q6 Mechanism Delta

Baseline widened q6 exposure:

- `Riverside Plaza`: `pool_only`
- `Minneapolis`: `pool_only`
- `Mississippi River`: `pool_only`

Probe q6 exposure:

- `Riverside Plaza`: `source`
  - `best_scored_rank=1`
  - `source_preview_ranks=[1, 1]`
  - `structure_score=1.0`
- `Minneapolis`: still `pool_only`
  - `best_scored_rank=41`
  - `structure_score=0.0`
- `Mississippi River`: still `pool_only`
  - `best_scored_rank=85`
  - `structure_score=0.0`

Important trace detail:

- `Riverside Plaza` now reaches `candidate_source_preview`, but still has:
  - `support_completeness_gain=0.0`
  - `utility_margin_gain=0.0`
  - no shortlist promotion

So the relation probe fixes the earlier binary graph-coverage block for one upstream bridge doc, but that alone does not make the requirement beam value it downstream.

## Interpretation

This probe supports a more specific staged diagnosis:

1. The earlier q6 graph audit was correct.
   The old structure graph was missing required factual edges.
2. Predicate coverage is a real blocker.
   Once enabled, `Riverside Plaza` moves from `pool_only -> source`.
3. Predicate coverage is not the only remaining blocker.
   `Minneapolis` and `Mississippi River` still do not reach source/shortlist.
4. The remaining break is now downstream of raw predicate recognition.

The strongest residual clue is the binding mismatch introduced by the newly added edge:

- `Riverside Plaza -> minneapolis minnesota`
- `Minneapolis -> mississippi river`

The chain is still not continuous at the normalized node level, so downstream reachability and need-unit gains stay near zero.

## Current Best Reading

After this probe, q6 looks like:

> graph coverage was a true first blocker, but once that blocker is removed, the next bottleneck becomes entity grounding / canonicalization continuity plus need-unit utility recognition for upstream bridge docs.

That is, the system is no longer failing only because the graph has no edge. It is now failing because the recovered edge does not yet translate into a continuous, scorer-visible bridge chain.

## Next Step

Do **not** go back to source-sort tuning yet.

The next correct-layer probe should target one of:

- entity/canonicalization continuity at the structure-graph level
  - e.g. `minneapolis minnesota` vs `minneapolis`
- or a narrowly scoped variable-binding / entity-grounding bonus only after continuity is confirmed

Current conclusion:

> the relation-coverage probe is informative and partially successful, but not sufficient. It upgrades q6 from “no graph edge” to “partial graph path with downstream continuity / utility blind spots.”
