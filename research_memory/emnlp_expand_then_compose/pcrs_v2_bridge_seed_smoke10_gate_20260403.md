# PCRS-RAG V2 Bridge-Seed Smoke10 Gate

Date: 2026-04-03

## Scope

This note records the first smoke10 gate after injecting bridge-seeded supervision into the V2 atomic scorer.

Artifacts produced in this round:

- rebuilt caches
  - `research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_cap2_8043_reuseopenie_bridge_seed_hybrid.json`
  - `research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_cap3_8043_reuseopenie_bridge_seed_hybrid.json`
- smoke reports
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_20260403.json`
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_cap2_bridge_seed_hybrid_20260403.json`
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_cap3_bridge_seed_hybrid_20260403.json`

Environment:

- reader / rerank: `qwen3-8b-train` on `http://localhost:8043/v1`
- embeddings: `VLLM//mnt/nvme/Qwen3-Embedding-8B` on `http://localhost:8018/v1/embeddings`
- judge available on `8045`, not used here

## Headline

Bridge-seeded supervision repairs the atomic layer, but it still does not survive the selector source frontier.

The evidence is now clean:

1. the atomic layer is no longer flat on `bridge_support`,
2. end-to-end smoke improves slightly over the old hybrid run,
3. but the critical bridge documents still never enter `candidate_source_preview` or `candidate_shortlist_preview`,
4. so `conditional / cap2 / cap3` still collapse to the same end result.

This means the current bottleneck is no longer parser fallback or hop cap.

It is now:

> `annotation frontier + proposal/source frontier`

more concretely:

> the bridge-seeded scorer can rescue bridge-like documents **inside the annotated slice**, but the true deep bridge docs for the key failure cases still do not receive live source exposure inside the top-100 selector run.

## Result Summary

### End-to-end metrics

Baseline remained unchanged in all runs:

- baseline EM = `0.3`
- baseline F1 = `0.35`

Bridge-seeded selector results:

- conditional: EM `0.1`, F1 `0.2167`, Recall@5 `0.4583`
- cap2: EM `0.1`, F1 `0.2167`, Recall@5 `0.4583`
- cap3: EM `0.1`, F1 `0.2167`, Recall@5 `0.4583`

Comparison against the previous hybrid smoke:

- old hybrid: EM `0.1`, F1 `0.2`, Recall@5 `0.4083`
- bridge-seeded: EM `0.1`, F1 `0.2167`, Recall@5 `0.4583`

So the supervision repair does help a little, but it does **not** fix the selector.

### Budget variants

`conditional / cap2 / cap3` now differ in some selected distractors, but their metrics are still identical:

- no EM separation
- no F1 separation
- no Recall@5 separation

So the system is still not in a state where hop-budget ablations are the right main axis.

## Mechanism Evidence

### Atomic gate improvement is real

From the rebuilt caches:

- old conditional hybrid
  - positive items with bridge argmax: `2`
  - positive items with `bridge_support_prob > 0.05`: `2`
- bridge-seeded conditional
  - positive items with bridge argmax: `21`
  - positive items with `bridge_support_prob > 0.05`: `23`
- bridge-seeded cap2
  - positive items with bridge argmax: `21`
  - positive items with `bridge_support_prob > 0.05`: `23`
- bridge-seeded cap3
  - positive items with bridge argmax: `24`
  - positive items with `bridge_support_prob > 0.05`: `26`

This confirms the bridge supervision is reaching the atomic scorer.

### Query 6 still fails before shortlist

Question:

`Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`

Known gold support docs inside top-100 pool:

- `Southeast Library`
- `Riverside Plaza`
- `Minneapolis`
- `Mississippi River`

What improved:

- `Southeast Library` now receives strong bridge-style atomic mass in the rebuilt cache.

What did not improve:

- in all bridge-seeded smoke reports, the only gold doc that survives selection is still `Southeast Library`
- none of `Riverside Plaza / Minneapolis / Mississippi River` appears in:
  - `candidate_source_preview`
  - `candidate_shortlist_preview`

Representative bridge-seeded source previews still look like:

- `Oklahoma City`
- `The Hague City Hall`
- `Elliott Bay`
- `Stockholm Public Library`
- `Davenport Public Library`

That is the same failure family as before: the real chain-closing docs do not even reach the beam expansion frontier.

### Query 7 still fails before shortlist

Question:

`When did the explorer reach the city where the headquarters of the only group larger than Vilaiyaadu Mankatha's record label is located?`

Known gold support docs:

- `The Right Stuff Records`
- `Santa Monica, California`
- `Vilaiyaadu Mankatha`
- `Sony Music`

What improved:

- in the rebuilt cache, `The Right Stuff Records` now gets strong bridge-support mass.

What did not improve:

- in all bridge-seeded smoke reports, only `Vilaiyaadu Mankatha` appears in the selected evidence set
- none of `The Right Stuff Records / Santa Monica, California / Sony Music` appears in:
  - `candidate_source_preview`
  - `candidate_shortlist_preview`

The live source frontier is still dominated by:

- `Manchester City F.C.`
- `Richmond, Virginia`
- `Vijay Eswaran`
- `Seattle`
- `Love Around`

So this is no longer an atomic-scoring-only problem. The selector never surfaces the right bridge docs for the atomic gains to matter.

## Interpretation

The bridge-seeded repair changed the system state in a useful way, but it exposed a harder bottleneck:

> the current requirement cache annotates only a shallow `ann20` slice, while the real bridge docs for key failures live deeper in the top-100 pool.

As a result:

- the atomic scorer gets stronger,
- but it only gets to score the shallow annotated slice well,
- and the deeper true bridge docs still behave like zero-information candidates during live selection.

That is why:

- smoke metrics improve a bit over the old hybrid run,
- but q6/q7 still do not recover,
- and hop-budget variants still do not separate.

## Decision

Do **not** spend the next round on:

- parser tuning
- more hop-cap sweeps
- larger 40/100 end-to-end validation

These are no longer the critical path.

## Next Action

The next round should explicitly target the frontier mismatch:

1. expand supervision from `ann20` to the live selector frontier
   - either raise annotation coverage for deeper pool docs
   - or do targeted on-demand annotation for source-preview candidates and deep bridge candidates
2. keep the bridge-seeded atomic model fixed for one round
3. recheck q6/q7 first
   - the gate is simple:
   - `Riverside Plaza / Minneapolis / Mississippi River / The Right Stuff Records / Sony Music`
     must appear in `candidate_source_preview` or `candidate_shortlist_preview`
4. only rerun larger smoke after that gate passes

## One-line Conclusion

> The bridge-seeded supervision repair succeeded at the atomic layer, but the live selector is still bottlenecked by shallow annotation coverage and source-frontier exposure. The next round should move upward to frontier coverage, not backward to parser or hop-cap tuning.
