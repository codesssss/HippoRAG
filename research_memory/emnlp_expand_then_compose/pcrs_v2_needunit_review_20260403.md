# PCRS V2 Need-Unit Review

Date: 2026-04-03

## Scope

This note reviews the current `PCRS-RAG V2` implementation on the `feature/pcrs-rag-v1` branch, with the review centered on one question:

> did the branch actually migrate from lexical requirements to semantic need units?

It also records one MuSiQue smoke experiment:

- cache build: `research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_pool100_ann50_limit40.json`
- eval report: `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke40_v2_legacyfact_pool100_ann50_reserve3.json`

## Code Review Findings

### 1. The V2 schema is present, but the parser is still mostly heuristic

The branch now emits:

- `qdmr_steps`
- `positive_need_units`
- `counterfactual_sets`

But the actual parser remains heuristic:

- `build_qdmr_steps_and_need_units()` synthesizes both steps and units from the same local rules
- `_extract_relation_predicates()` still falls back to focus-term concatenation
- constraints are extracted from token matches rather than step-level semantic decomposition

So this is not yet:

- teacher step plan
- deterministic compiler to structured units

It is better than V1 structurally, but still not a clean semantic parser.

### 2. Many relation predicates are still lexical chunks, not stable relation slots

On the MuSiQue-40 cache:

- total positive units: `127`
- relation-hop units: `41`
- lexical-looking relation-hop predicates: `33`
- lexical relation-hop rate: `0.8049`

Examples:

- `person_goals`
- `publisher_end`
- `birthplace_abolished`
- `region_immediately`
- `body_water`
- `company_succeeded`

This is the clearest signal that V2 is still carrying V1-style lexical behavior inside a structured schema.

### 3. Counterfactual generation is structurally upgraded, but mostly absent in practice

The unit-level transforms exist:

- `role_swap`
- `predicate_shift`
- `temporal_shift`
- `constraint_flip`

But on MuSiQue-40:

- queries with any counterfactual sets: `8 / 40`
- queries with zero counterfactual sets: `32 / 40`
- average counterfactual sets per query: `0.375`

Transform counts:

- `role_swap`: `5`
- `predicate_shift`: `5`
- `constraint_flip`: `4`
- `temporal_shift`: `1`

This means the negative axis is not broadly active. The implementation is better than pure entity swap, but coverage is too sparse for the method to consistently behave like counterfactual set retrieval.

### 4. The scorer moved to the right interface, but not yet to true support classification

`score_need_unit_support()` now outputs:

- `alignment_score`
- `support_prob`
- `contradiction_prob`
- `nei_prob`
- `coverage_score`

That matches the intended V2 interface. But the internals are still overlap-style heuristics:

- subject token/entity overlap
- predicate token overlap
- object token overlap
- simple opposing-predicate string hits
- simple constraint string hits

So the branch is not yet doing a real support-vs-contradiction judgment in the FEVER/QED sense. It is still mostly lexical similarity plus a small contradiction heuristic.

### 5. The selector boundary is mostly respected

This part is in good shape.

The branch did not quietly rewrite the beam logic. The main search skeleton still looks the same, and V2 is mainly changing:

- state metrics
- candidate feature rows
- cache/schema plumbing

This is good, because it keeps the experiment focused on whether the new need-unit signal is useful.

### 6. Learned mode is not the current blocker, but it is not fully semantically clean either

The V2 feature namespace is separate and the learned/oracle paths are cleanly split at runtime.

However:

- `train_need_unit_scorer.py` still reuses `build_requirement_training_rows()` from the old requirement training pipeline

So the learned path is not yet a clean end-to-end V2-native training stack.

## Cache Quality Snapshot

From `musique_need_unit_cache_pool100_ann50_limit40.json`:

- average QDMR steps per query: `3.175`
- average positive need units per query: `3.175`
- average counterfactual sets per query: `0.375`
- malformed unit count: `0`
- suspicious wh-subject rate: `0.0157`
- entity-locator subjects with noisy surface forms: `9 / 40`

Noisy entity examples:

- `messi s`
- `erik hort s`
- `lady godiva s`
- `s o jos`
- `u s`
- `young man luther s`

This means the explicit wh-word bug is mostly reduced, but entity surface normalization is still noisy enough to distort unit quality.

## MuSiQue-40 Oracle Result

Run:

- dataset: `musique`
- limit: `40`
- selector: `requirement_beam`
- cache: `musique_need_unit_cache_pool100_ann50_limit40.json`
- mode: `oracle`
- pool: `100`
- reserve: `3`
- projected shortlist factor: `1`

Result:

- baseline EM: `0.375`
- baseline F1: `0.4016`
- selector EM: `0.3500`
- selector F1: `0.3667`
- EM delta: `-0.0250`
- F1 delta: `-0.0349`

Bucket view:

- `2_doc`: F1 delta `-0.0634`
- `3_doc`: F1 delta `0.0`
- `4_doc`: F1 delta `0.0`

Mechanism indicators:

- average support completeness: `0.4941`
- average counterfactual leakage: `0.0999`
- average frontier size: `2.425`
- average finalist leakage range: `0.0187`
- median finalist leakage range: `0.0`
- nonzero best-leakage queries: `8 / 40`

Interpretation:

- the branch still underperforms the baseline
- the negative axis is almost inactive
- the leakage behavior matches the cache fact that only `8 / 40` queries have counterfactual sets at all

So this run does not support the claim that V2 already behaves like semantic need-unit counterfactual retrieval.

## Bottom Line

The branch is a real structural migration, but not yet a semantic migration.

What is already true:

- V2 has a distinct schema
- V2 has unit-level counterfactual machinery
- V2 has a support/contradiction interface
- V2 keeps the selector boundary mostly clean

What is not yet true:

- parser outputs are not yet reliably semantic need units
- relation predicates are still mostly lexical chunks
- counterfactual coverage is too sparse
- support scoring is still heuristic overlap, not real support discrimination

In short:

> V2 currently looks like “V1-style signals inside a better schema”, not yet “semantic need-unit retrieval”.
