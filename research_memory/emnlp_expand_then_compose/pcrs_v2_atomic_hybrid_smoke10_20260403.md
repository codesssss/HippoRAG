# PCRS-RAG V2 Atomic Hybrid Smoke10 Note

Date: 2026-04-03

## Scope

This note records the first end-to-end smoke run after replacing the V2 atomic layer with the new hybrid scorer:

- atomic model: `research_memory/emnlp_expand_then_compose/models/musique_need_unit_atomic_multiclass_hist_gbdt_pool100_limit10.joblib`
- rebuilt hybrid caches:
  - `research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_reuseopenie_atomic_hybrid.json`
  - `research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_cap2_8043_reuseopenie_atomic_hybrid.json`
  - `research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_cap3_8043_reuseopenie_atomic_hybrid.json`
- smoke reports:
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_atomic_hybrid_20260403.json`
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_cap2_atomic_hybrid_20260403.json`
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_cap3_atomic_hybrid_20260403.json`

Environment:

- reader / rerank model: `qwen3-8b-train` on `http://localhost:8043/v1`
- embedding model: `VLLM//mnt/nvme/Qwen3-Embedding-8B` on `http://localhost:8018/v1/embeddings`
- judge service also available on `8045`, but not used here

## Headline

The hybrid atomic scorer is not ready for expansion.

The smoke10 run shows a clean failure pattern:

1. the new atomic layer changes the selected evidence sets a lot,
2. but it does not produce better QA,
3. and it largely washes out the `conditional / cap2 / cap3` budget differences.

So this round does **not** support moving to larger validation yet.

## What Was Run

### 1. Atomic training

Command:

```bash
.venv-hipporag/bin/python scripts/train_need_unit_scorer.py \
  --dataset musique \
  --limit 10 \
  --train_queries 8 \
  --eval_queries 2 \
  --split_seed 13 \
  --save_dir outputs_step0_general \
  --requirement_cache_path research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_reuseopenie.json \
  --output_dir research_memory/emnlp_expand_then_compose/models \
  --training_task atomic_multiclass \
  --model_type hist_gbdt \
  --include_counterfactual_samples true \
  --llm_base_url http://localhost:8043/v1 \
  --llm_name qwen3-8b-train \
  --embedding_base_url http://localhost:8018/v1/embeddings \
  --embedding_name VLLM//mnt/nvme/Qwen3-Embedding-8B \
  --openie_mode online
```

Artifacts:

- `research_memory/emnlp_expand_then_compose/models/musique_need_unit_atomic_multiclass_hist_gbdt_pool100_limit10.joblib`
- `research_memory/emnlp_expand_then_compose/models/musique_need_unit_atomic_multiclass_hist_gbdt_pool100_limit10.json`
- `research_memory/emnlp_expand_then_compose/models/musique_need_unit_atomic_multiclass_hist_gbdt_pool100_limit10.rows.jsonl`

### 2. Hybrid cache rebuild

We rebuilt all three smoke10 caches with:

- `scripts/annotate_need_unit_support.py`
- `--score_mode hybrid`
- `--atomic_model_path research_memory/emnlp_expand_then_compose/models/musique_need_unit_atomic_multiclass_hist_gbdt_pool100_limit10.joblib`
- `--openie_mode online`

### 3. End-to-end smoke

All three runs used the same old clean smoke configuration:

- `--setwise_selector requirement_beam`
- `--setwise_requirement_mode oracle`
- `--setwise_pool_k 100`
- `--setwise_anchor_count 2`
- `--setwise_reserve_top_m 3`
- `--setwise_non_anchor_title_dedup true`
- `--setwise_beam_width 4`
- `--setwise_beam_expand_per_state 4`
- `--setwise_beam_projected_shortlist_factor 1`

Only the requirement cache path changed.

## Core Results

### End-to-end QA

Baseline retrieval+reader stayed the same in all runs:

- baseline EM = `0.3`
- baseline F1 = `0.35`

Selector QA after hybrid atomic scoring:

- conditional: EM `0.1`, F1 `0.2`
- cap2: EM `0.1`, F1 `0.2`
- cap3: EM `0.1`, F1 `0.2`

This means:

- all three hybrid runs are still substantially worse than baseline
- all three hybrid runs collapse to the same end result

### Retrieval-side selector metrics

Compared with the previous clean V2 reports:

- clean selector Recall@5: `0.4917`
- hybrid selector Recall@5: `0.4083`

So the hybrid atomic layer changes the selected set aggressively, but the selected set is worse by gold-support recall.

### Requirement-state summaries

Old clean vs new hybrid:

- conditional
  - clean support/leakage/frontier: `0.2235 / 0.2424 / 4.0`
  - hybrid support/leakage/frontier: `0.2380 / 0.1324 / 4.55`
- cap2
  - clean support/leakage/frontier: `0.2230 / 0.2381 / 3.95`
  - hybrid support/leakage/frontier: `0.2274 / 0.1161 / 4.65`
- cap3
  - clean support/leakage/frontier: `0.2290 / 0.2417 / 3.9`
  - hybrid support/leakage/frontier: `0.2393 / 0.1387 / 4.3`

This is the same failure pattern across all three variants:

- support completeness rises slightly
- leakage drops a lot
- frontier gets wider
- but selected evidence recall gets worse
- final QA remains bad

So the new atomic layer is not fixing the original mismatch. It is just pushing the selector into a different but still bad regime.

## Atomic Model Diagnosis

The training report already explains why the hybrid scorer is fragile.

From `musique_need_unit_atomic_multiclass_hist_gbdt_pool100_limit10.json`:

- training labels:
  - `nei`: `1024`
  - `full_support`: `164`
  - `contradiction`: `11`
  - `bridge_support`: `1`
- eval labels:
  - `nei`: `279`
  - `full_support`: `16`
  - `contradiction`: `5`
  - `bridge_support`: `0`

So the first hybrid model is almost not trained on bridge cases at all.

That shows up directly in the rebuilt hybrid cache:

- positive atomic items: `660`
- argmax label counts:
  - `nei`: `520`
  - `full_support`: `109`
  - `contradiction`: `29`
  - `bridge_support`: `2`
- positive items with `bridge_support_prob > 0.05`: `2 / 660`

Only two positive atomic items get meaningful bridge mass, both from the same query:

- query about the group larger than `Vilaiyaadu Mankatha`'s record label
- docs: `Vilaiyaadu Mankatha` and `YouTube`

This means the scorer is still behaving almost entirely as:

- full support
- contradiction
- NEI

and not as a real bridge-aware utility scorer.

## Evidence-Set Behavior

The hybrid scorer does move the selector.

Compared with the previous clean runs:

- conditional changed `10 / 10` query-level selected sets
- cap2 changed `10 / 10`
- cap3 changed `9 / 10`

Compared with baseline retrieval:

- all three hybrid runs changed `10 / 10` query-level selected sets

But the three hybrid variants are still close to one another:

- conditional vs cap2: `2 / 10` selected sets differ
- conditional vs cap3: `2 / 10` selected sets differ

So once the hybrid atomic layer is on, the budget variants matter much less than the scorer failure itself.

## Concrete Regressions

Two easy regressions from the conditional hybrid report:

### Query 1

Question:

`When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?`

Baseline top titles:

- `Lionel Messi`
- `FC Barcelona`
- `FC Barcelona`
- `Lionel Messi`
- `FC Barcelona`

Hybrid selector top titles:

- `Lionel Messi`
- `FC Barcelona`
- `List of international goals scored by Lionel Messi`
- `List of La Liga top scorers`
- `List of Spanish football champions`

This looks like topical football support, not chain-closing evidence.

### Query 4

Question:

`What year did the publisher of Labyrinth end?`

Baseline top titles:

- `Warlocked`
- `The Devil's Labyrinth`
- `Labyrinth (1984 video game)`
- `The Sense of an Ending`
- `Acornsoft`

Hybrid selector top titles:

- `Warlocked`
- `The Devil's Labyrinth`
- `Labyrinth (1984 video game)`
- `List of Little House on the Prairie books`
- `The Sense of an Ending`

This is a direct regression away from the needed publisher/eol chain.

## Mechanism-Level Conclusion

This smoke run supports the following conclusion:

> The current hybrid atomic scorer is not yet improving need-unit alignment.
> It changes evidence sets strongly, but because bridge supervision is nearly absent, it mainly suppresses leakage and reshuffles topical support instead of rewarding true bridge completion.
> As a result, the selector still drifts toward distractors, and the `conditional / cap2 / cap3` budget differences become secondary.

## Decision

Do **not** move to 40-query or larger validation with this hybrid scorer.

The next step should be upstream repair of atomic supervision, not longer reader runs.

Most valuable next actions:

1. fix bridge-support supervision
   - current training split has only `1` bridge label
   - hybrid scorer cannot learn the intended behavior in this regime

2. harden weak/pseudo labeling
   - especially bridge vs NEI
   - especially hard negatives with bridge-like lexical overlap but broken variable chains

3. rerun the same smoke10 gate after label repair
   - only expand if the selector starts improving shortlist/source quality and no longer drops Recall@5 this sharply

## Minor Operational Note

When two hybrid annotation jobs were run in parallel, one path briefly failed to reuse `graph.pickle` with `Ran out of input` and fell back to rebuilding the legacy substrate.

This did not crash the job, but it suggests:

- parallel cache rebuilds can race on shared graph reuse
- later large rebuilds should prefer sequential execution if exact runtime stability matters
