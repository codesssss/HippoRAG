# PCRS-RAG V2 Bridge Probe Seed Round 1

Date: 2026-04-03

## Scope

This note records the first bridge-first supervision repair round after the failed hybrid smoke10 run.

Files added in this round:

- bridge label spec:
  - `research_memory/emnlp_expand_then_compose/pcrs_v2_bridge_label_spec_20260403.md`
- bridge probe miner:
  - `scripts/mine_bridge_probe_examples.py`
- mined probe rows:
  - `research_memory/emnlp_expand_then_compose/data/musique_bridge_probe_20260403.jsonl`
- mined probe preview:
  - `research_memory/emnlp_expand_then_compose/data/musique_bridge_probe_20260403.md`
- seed labels:
  - `research_memory/emnlp_expand_then_compose/data/musique_bridge_probe_seed_labels_20260403.jsonl`
- bridge-seeded atomic model:
  - `research_memory/emnlp_expand_then_compose/models/musique_need_unit_atomic_multiclass_hist_gbdt_pool100_limit10_bridge_seed.joblib`
- bridge-seeded atomic report:
  - `research_memory/emnlp_expand_then_compose/models/musique_need_unit_atomic_multiclass_hist_gbdt_pool100_limit10_bridge_seed.json`
- bridge-seeded rebuilt cache:
  - `research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_reuseopenie_bridge_seed_hybrid.json`

## What Was Done

### 1. Defined bridge support tightly

We wrote a bridge label spec that separates:

- `full_support`
- `bridge_support`
- `nei`
- `contradiction`

The critical rule is:

`bridge_support` must preserve variable-chain continuity, not just topical relatedness.

### 2. Mined a bridge-first probe set

Using:

- the conditional hybrid cache
- the conditional hybrid smoke10 report
- gold docs from `MuSiQue limit=10`

the probe miner exported:

- `59` selected probe rows

Bucket breakdown:

- `17` bridge positives
- `8` bridge-like hard negatives
- `24` clean NEI
- `10` full-support controls

### 3. Generated training-ready seed labels

The high-confidence seed label file contains:

- `48` total rows
- `6` `bridge_support`
- `32` `nei`
- `10` `full_support`

This is still small, but it is already enough to move bridge supervision from near-zero to non-trivial.

### 4. Retrained the atomic multiclass model with seed labels

Compared with the original atomic report:

- old train `bridge_support`: `1`
- new train `bridge_support`: `19`

Train label source counts:

- old:
  - `heuristic`: `438`
  - `weak`: `762`
- new:
  - `llm`: `35`
  - `weak`: `745`
  - `heuristic`: `420`

This confirms the seed labels are actually entering training and changing the label distribution.

## Upstream Gate Result

We did not rerun smoke yet.

We first checked whether the bridge-seeded model changes the atomic layer itself.

### Positive atomic distribution

Old hybrid cache:

- positive atomic items: `660`
- argmax `bridge_support`: `2`
- positive items with `bridge_support_prob > 0.05`: `2`

Bridge-seeded rebuilt cache:

- positive atomic items: `660`
- argmax `bridge_support`: `21`
- positive items with `bridge_support_prob > 0.05`: `23`

This is the first real sign that the bridge axis is no longer flat.

## Key Example Changes

### Query: Southeast Library

Old hybrid cache:

- doc: `Southeast Library`
- unit: `u_relation_hop_0` / `designer_of`
- `support_prob = 0.000147`
- `bridge_support_prob = 0.0`
- `coverage_score = 0.0`

Bridge-seeded cache:

- same doc and unit
- `support_prob = 0.64353`
- `bridge_support_prob = 0.990027`
- `coverage_score = 0.273496`

This is exactly the type of bridge doc we wanted to rescue.

### Query: Vilaiyaadu Mankatha

Old hybrid cache:

- doc: `Vilaiyaadu Mankatha`
- unit: `u_relation_hop_1` / `headquartered_in`
- `support_prob = 0.652442`
- `bridge_support_prob = 0.989084`

Bridge-seeded cache:

- same doc and unit
- `support_prob = 0.680107`
- `bridge_support_prob = 0.870645`

This means the known bridge case survived the supervision change.

### Query: The Right Stuff Records

Old hybrid cache:

- doc: `The Right Stuff Records`
- unit: `u_relation_hop_1` / `headquartered_in`
- `support_prob = 0.0`
- `bridge_support_prob = 0.0`

Bridge-seeded cache:

- same doc and unit
- `support_prob = 0.611575`
- `bridge_support_prob = 0.940884`

This is a strong recovery signal for previously invisible bridge candidates.

## Current Interpretation

This round does not prove that end-to-end QA is fixed.

But it does prove something important:

> once bridge supervision is injected, the atomic layer starts assigning non-trivial bridge mass to real middle-hop documents.

That is exactly the missing precondition from the failed hybrid smoke10 run.

## Decision

The supervision repair is working well enough to justify the next gate.

The next step should be:

1. rebuild `cap2` and `cap3` caches with the bridge-seeded model
2. check whether bridge docs now enter shortlist/source previews
3. only then rerun smoke10

Do not expand to `40` yet.

## Open Risks

1. The seed bridge set is still small:
   - only `6` high-confidence `bridge_support` overrides
2. Some bridge-positive probe rows remain `medium` confidence and should be reviewed before a second training round.
3. We have not yet verified whether these atomic improvements survive all the way through shortlist and QA.
