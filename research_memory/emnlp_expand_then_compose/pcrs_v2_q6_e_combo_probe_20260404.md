# PCRS V2 q6 E combo probe (2026-04-04)

## Goal

Validate the first fully closed q6 combination:

- `structure_relation_probe_mode=q6_factual`
- `structure_continuity_probe_mode=city_state_alias`
- `structure_seed_target_bridge_mode=allow_seed_target`

This is the first configuration that should close:

- `Riverside Plaza -> Minneapolis, Minnesota`
- `Minneapolis, Minnesota -> Minneapolis`
- `Minneapolis -> Mississippi River`
- seed-target bridge acceptance for already-covered targets

## Run order

### 1. q6-only trace

Created a temporary single-query dataset:

- `reproduce/dataset/musique_q6probe.json`
- `reproduce/dataset/musique_q6probe_corpus.json` -> symlink to `musique_corpus.json`

Command:

```bash
.venv-hipporag/bin/python scripts/eval_causal_qwen3.py \
  --dataset musique_q6probe \
  --limit 1 \
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
  --structure_seed_target_bridge_mode allow_seed_target \
  --setwise_requirement_exposure_watch_titles 'Southeast Library,Riverside Plaza,Minneapolis,Mississippi River' \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_q6probe_conditional_offlinehybrid_varfix_clean_e_q6factual_cityalias_seedtarget_20260404.json
```

### 2. smoke10 E

Command:

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
  --structure_continuity_probe_mode city_state_alias \
  --structure_seed_target_bridge_mode allow_seed_target \
  --setwise_requirement_exposure_watch_titles 'Southeast Library,Riverside Plaza,Minneapolis,Mississippi River,The Right Stuff Records,Sony Music' \
  --output_json outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_offlinehybrid_varfix_clean_e_q6factual_cityalias_seedtarget_20260404.json
```

## q6-only result

Report:

- `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_q6probe_conditional_offlinehybrid_varfix_clean_e_q6factual_cityalias_seedtarget_20260404.json`

Observed:

- selected evidence:
  - `Southeast Library`
  - `Colorado River (Texas)`
  - `Gulf of Mexico`
  - `Mississippi River`
  - `Minneapolis`
- selector answer:
  - `Mississippi River.`
- selector F1:
  - `0.8`

Key mechanism result:

- `Minneapolis` is no longer dark
- in step 5:
  - `scored_rank=3`
  - `source_rank=3`
  - `shortlist_rank=3`
  - `structure_score=0.8339`
  - `support_completeness_gain=0.0973`
  - `utility_margin_gain=0.0573`

Exposure summary:

- `Riverside Plaza`: `shortlist`
- `Minneapolis`: `selected`
- `Mississippi River`: `selected`

Interpretation:

- the q6 chain is finally closed far enough for `Minneapolis` to receive nonzero structure support and nonzero utility
- this is the first direct evidence that the three-part combo is sufficient to light up the missing middle bridge on q6

## smoke10 result

Report:

- `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_offlinehybrid_varfix_clean_e_q6factual_cityalias_seedtarget_20260404.json`

### Aggregate comparison

| run | EM | F1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| A varfix clean | 0.1000 | 0.2167 | 0.4417 | 0.6250 |
| B + q6_factual | 0.1000 | 0.2567 | 0.4667 | 0.6500 |
| C + city_state_alias | 0.1000 | 0.2567 | 0.4667 | 0.6500 |
| D + allow_seed_target | 0.1000 | 0.2300 | 0.4917 | 0.6750 |
| E + alias + seed_target | 0.1000 | 0.2967 | 0.4917 | 0.6750 |

Readout:

- E is the best F1 among A/B/C/D/E
- E matches D on retrieval recall
- E improves F1 over D by recovering q6 chain propagation instead of only rescuing the downstream `Mississippi River` edge

## q6 in smoke10 E

Observed:

- selector answer:
  - `Mississippi River.`
- selector F1:
  - `0.8`
- selected evidence:
  - `Southeast Library`
  - `Colorado River (Texas)`
  - `Gulf of Mexico`
  - `Mississippi River`
  - `Minneapolis`

Exposure summary:

- `Riverside Plaza`: `shortlist`
- `Minneapolis`: `selected`
- `Mississippi River`: `selected`

This is stronger than D:

- D rescued `Mississippi River` but left `Minneapolis` dark
- E rescues both `Minneapolis` and `Mississippi River`

## q7 sanity check

Observed in smoke10 E:

- selector answer still `Not enough information.`
- `The Right Stuff Records`: still `pool_only`
- `Sony Music`: still `pool_only`

Interpretation:

- E is a q6-style chain closure repair
- it does not change the earlier conclusion that q7 remains pool/source exposure limited

## Conclusion

This E run is a successful mechanism validation.

What it supports:

- `q6_factual` alone is not enough
- `city_state_alias` alone is not enough
- `allow_seed_target` alone is not enough
- but the combination of all three closes the q6 bridge chain enough for:
  - `Minneapolis` to receive positive structure score
  - `Minneapolis` to enter source / shortlist / final selected evidence
  - `Mississippi River` to remain lit

Current recommendation:

- keep all three as **eval-only**
- do not mainline yet
- treat E as evidence that q6 remaining failure was indeed a multi-part chain-closure issue, not only a local ranking issue
