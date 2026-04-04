# PCRS-RAG V2 Q6 Canonicalization / Continuity Audit

Date: 2026-04-04

## Goal

Follow up the relation-coverage probe and answer:

> after factual structure edges are added, why does q6 still fail with `gain=0` for the upstream bridge docs?

## Scope

This audit uses the eval-only probe mode:

- `structure_relation_probe_mode=q6_factual`

and inspects the exact q6 gold docs in the current workspace.

## Exact Probe Edges on Q6 Gold Docs

After rebuilding structure objects with the probe enabled:

- `Southeast Library`
  - `ralph rapson -> southeast library`
- `Riverside Plaza`
  - `ralph rapson -> riverside plaza`
  - `riverside plaza -> minneapolis minnesota`
- `Minneapolis`
  - `minneapolis -> mississippi river`
- `Mississippi River`
  - `mississippi river -> gulf of mexico`

So the factual relation coverage is present.

## Continuity Check

### Attribution side

The attribution side is aligned:

- both `Southeast Library` and `Riverside Plaza` normalize `Ralph Rapson` to the same node
  - `ralph rapson`

This means the designer bridge is no longer missing because of predicate classification.

### City / location side

The city transition is not aligned:

- `Riverside Plaza`
  - target node: `minneapolis minnesota`
- `Minneapolis`
  - source node: `minneapolis`

These are different structure nodes under the current normalization.

Current normalization only does:

- lowercase
- punctuation stripping
- whitespace collapse

It does **not** merge:

- `minneapolis minnesota`
- `minneapolis`

So the graph now contains a partial path, but not a continuous node-level chain.

### Downstream river side

The downstream river side is aligned:

- `Minneapolis`
  - `minneapolis -> mississippi river`
- `Mississippi River`
  - `mississippi river -> gulf of mexico`

So once the city node is grounded continuously, the rest of the downstream chain should be much closer to usable.

## What This Explains

This continuity break explains the current selector behavior:

- `Riverside Plaza` can now get a high `structure_score`
  - it is seen as adjacent to the current covered set through `ralph rapson`
- but it still gets:
  - `support_completeness_gain = 0`
  - `utility_margin_gain = 0`

because the path it opens does not yet connect into the downstream unit chain through the scorer-visible node inventory.

In other words:

> the graph can now see `Riverside Plaza` as structurally nearby, but it still cannot consume it as a continuous variable-binding bridge because `minneapolis minnesota` and `minneapolis` remain disconnected.

## Diagnosis

At the current stage, q6 is best described as:

1. missing predicate coverage: partially fixed
2. node-level continuity / canonicalization: still broken
3. upstream bridge utility recognition: still likely weak, but not yet cleanly testable until node continuity is fixed

## Practical Next Step

The next correct-layer probe should be a minimal, eval-only canonicalization continuity probe, not more source-sort tuning.

The most targeted case is:

- merge or alias-match `city, state` and bare `city` variants in the structure graph / structure scoring path

For q6 specifically:

- `minneapolis minnesota` <-> `minneapolis`

Only after that continuity probe should we reassess whether:

- `Riverside Plaza`
- `Minneapolis`

still have `gain=0`.

If they do, then the next bottleneck is more clearly an upstream bridge / variable-binding scoring blind spot.
