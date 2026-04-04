# PCRS V2 q6 seed-target bridge scorer probe (2026-04-04)

## Goal

Test a minimal, default-off scorer fix for q6-style chain propagation:

- keep retrieval / parser / final objective unchanged
- keep `expand_directed_entities()` semantics unchanged
- only allow `score_candidate_docs_by_structure()` to accept an explicit bridge edge when:
  - the source is reachable from the current covered set, and
  - the target is already in the covered seed set

Flag:

- `--structure_seed_target_bridge_mode off|allow_seed_target`
- default: `off`

## Code changes

Targeted plumbing only:

- `src/hipporag/utils/causal_utils.py`
  - added `STRUCTURE_SEED_TARGET_BRIDGE_MODES`
  - added `normalize_structure_seed_target_bridge_mode(...)`
  - extended `score_candidate_docs_by_structure(..., seed_target_bridge_mode=...)`
  - when mode is `allow_seed_target`, set `target_support = 1.0` if the edge target is already in normalized seeds
- `src/hipporag/utils/config_utils.py`
  - added config field `structure_seed_target_bridge_mode`
- `scripts/eval_causal_qwen3.py`
  - added CLI flag `--structure_seed_target_bridge_mode`
  - plumbed the mode into:
    - `compute_candidate_feature_rows`
    - `score_bridge_candidates`
    - `compute_bridge_gate_decision`
    - `select_bridge_greedy_positions`
    - `select_bridge_beam_positions`
    - `select_requirement_beam_positions`
    - `score_evidence_state`
    - `apply_setwise_selector`
  - surfaced the mode in saved eval config / selector traces
- `tests/test_causal_utils.py`
  - added a focused unit test for the q6 pattern:
    - reachable source via alias edge
    - target already in seed set
    - legacy mode scores `{}`
    - `allow_seed_target` yields positive structure score

## Commands run

### Static / tests

```bash
python -m py_compile scripts/eval_causal_qwen3.py src/hipporag/utils/causal_utils.py src/hipporag/utils/config_utils.py tests/test_causal_utils.py
.venv-hipporag/bin/python -m pytest -q tests/test_causal_utils.py -k 'score_candidate_docs_by_structure'
.venv-hipporag/bin/python -m pytest -q tests/test_setwise_selector.py -k 'select_bridge or beam_closure or offline_hybrid_annotation_mode or canonical_variable_slots or legacy_serialized_variables or parser_trace_across_fallback'
```

Observed:

- `tests/test_causal_utils.py`: `3 passed`
- `tests/test_setwise_selector.py` subset: `17 passed`

### Smoke10 eval

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
  --setwise_requirement_cache_path research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_varfix_offlinehybrid_clean.json \
  --setwise_requirement_mode oracle \
  --setwise_requirement_annotation_pool_k 20 \
  --setwise_requirement_reserve_policy fixed \
  --structure_relation_probe_mode q6_factual \
  --structure_seed_target_bridge_mode allow_seed_target \
  --setwise_requirement_exposure_watch_titles 'Southeast Library,Riverside Plaza,Minneapolis,Mississippi River,The Right Stuff Records,Sony Music' \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_offlinehybrid_varfix_clean_d_q6factual_seedtarget_20260404.json
```

## Aggregate result

Reference smoke10 reports:

- A: `varfix clean`
- B: `varfix clean + q6_factual`
- C: `varfix clean + q6_factual + city_state_alias`
- D: `varfix clean + q6_factual + allow_seed_target`

| run | EM | F1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| A | 0.1000 | 0.2167 | 0.4417 | 0.6250 |
| B | 0.1000 | 0.2567 | 0.4667 | 0.6500 |
| C | 0.1000 | 0.2567 | 0.4667 | 0.6500 |
| D | 0.1000 | 0.2300 | 0.4917 | 0.6750 |

Readout:

- `allow_seed_target` improved retrieval recall over B/C
- but aggregate F1 regressed vs B/C
- this is not a clean overall win; it is a mechanism probe result

## q6 mechanism result

Query:

> Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?

### B (`q6_factual`) before the fix

- selected evidence:
  - `Southeast Library`
  - `Colorado River (Texas)`
  - `Gulf of Mexico`
  - `Riverside Plaza`
  - `The Hague City Hall`
- q6 answer: `Colorado River.`
- q6 F1: `0.4`
- exposure summary:
  - `Riverside Plaza`: `selected`
  - `Minneapolis`: `pool_only`
  - `Mississippi River`: `pool_only`

### D (`q6_factual + allow_seed_target`) after the fix

- selected evidence:
  - `Southeast Library`
  - `Colorado River (Texas)`
  - `Gulf of Mexico`
  - `Riverside Plaza`
  - `Mississippi River`
- q6 answer: `Mississippi River.`
- q6 F1: `0.8`
- exposure summary:
  - `Riverside Plaza`: `selected`
  - `Minneapolis`: `pool_only`
  - `Mississippi River`: `selected`

### Important q6 trace details

Step 4 in D:

- `Riverside Plaza`
  - `scored_rank=1`
  - `structure_score=1.0`
  - `support_completeness_gain=0.1191`
  - `utility_margin_gain=0.0449`
- `Mississippi River`
  - `scored_rank=2`
  - `structure_score=0.9781`
  - `support_completeness_gain=0.1507`
  - `utility_margin_gain=0.0518`
- `Minneapolis`
  - `scored_rank=44`
  - `structure_score=0.0`
  - `support_completeness_gain=0.0`
  - `utility_margin_gain=0.0`

Step 5 in D:

- `Mississippi River`
  - `scored_rank=1`
  - `structure_score=0.9882`
  - `support_completeness_gain=0.1288`
  - `utility_margin_gain=0.0397`
- `Minneapolis`
  - still `scored_rank=43`
  - still `structure_score=0.0`
  - still `support_completeness_gain=0.0`
  - still `utility_margin_gain=0.0`

## q7 sanity check

Query family:

> ... the only group larger than Vilaiyaadu Mankatha's record label ...

Observed in B and D:

- `The Right Stuff Records`: still `pool_only`
- `Sony Music`: still `pool_only`
- selector answer remains `Not enough information.`
- no sign that this fix helps q7

## Conclusion

This probe supports a narrow scorer-semantics diagnosis:

- the new flag **does** repair one real blind spot:
  - explicit bridge edges whose target is already covered can now score positively
- that repair is strong enough to change q6 downstream behavior:
  - `Mississippi River` moves from `pool_only` to `selected`
  - q6 answer improves from `Colorado River` to `Mississippi River`

But the probe also shows the remaining q6 bottleneck clearly:

- `Minneapolis` is still completely dark
- so the main unresolved break is **not** this seed-target acceptance rule
- the remaining break is still upstream:
  - continuity / canonicalization / binding into `Minneapolis`
  - or a scorer blind spot that never gives `Minneapolis` nonzero gain

Decision:

- keep this as a **default-off eval-only fix**
- do **not** promote to clean mainline yet
- next q6 diagnostic should stay focused on why `Minneapolis` remains `structure_score=0`
