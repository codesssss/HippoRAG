# BEEP Day -1B: MuSiQue 4-hop Manual Hop-Type Audit

- Date: `2026-04-29`
- Dataset/pool: `run_logs/dense_pool_exports_full1000_20260424/musique_dense_pool100.json`
- Split: first 10 MuSiQue 4-hop examples inside limit=100
- Decision: `STOP_ENTITY_BRIDGE_CEILING_FAIL`

## Gate

Proceed only if:

```text
>=60% hop transitions are entity_bridge
and >=6/10 queries have all required transitions representable by entity bridges.
```

## Summary

| Metric | Value |
|---|---:|
| Queries audited | 10 |
| Estimated hop transitions | 30 |
| Entity-bridge transitions | 14 |
| Entity-bridge transition rate | 0.4667 |
| Fully entity-bridge queries | 1 |
| Fully entity-bridge query rate | 0.1000 |

The audit fails both thresholds. Most failing transitions require geography,
relation selection, comparison, attribute/event interpretation, or same-name
disambiguation rather than simple entity co-occurrence.

## Cases

| # | query_idx | Short diagnosis | Labels |
|---:|---:|---|---|
| 1 | 5 | Saudi/Israel/Kuwait geography and "created" date require region relation selection, not just shared entities. | `relation_selection`, `comparison_operator`, `attribute_or_event` |
| 2 | 6 | Southeast Library -> Ralph Rapson -> Minneapolis -> Mississippi River is mostly bridgeable by entities. | `entity_bridge`, `entity_bridge`, `entity_bridge` |
| 3 | 7 | Sony/Universal/Santa Monica path has entity bridges, but "only group larger" is a relation-selection step. | `entity_bridge`, `relation_selection`, `entity_bridge` |
| 4 | 11 | Portuguese/Ottoman/Myanmar/Laos path depends on country-between relation and historical expulsion/defeat relation. | `entity_bridge`, `relation_selection`, `relation_selection` |
| 5 | 24 | Mexico/North America/John Cabot/Sebastian Cabot is partially bridgeable but requires continent and child relation selection. | `relation_selection`, `entity_bridge`, `entity_bridge` |
| 6 | 30 | Springfield Missouri vs Springfield Illinois requires same-name disambiguation and state-capital relation. | `relation_selection`, `relation_selection`, `attribute_or_event` |
| 7 | 33 | Near East/Shamal/Saudi path is dominated by geography and region interpretation. | `relation_selection`, `comparison_operator`, `attribute_or_event` |
| 8 | 39 | MLB/World Series/draft path has entity overlaps but "after it" and league/competition mapping require relation selection. | `entity_bridge`, `relation_selection`, `entity_bridge` |
| 9 | 49 | Sony/Universal/Santa Monica path is bridgeable but "only group larger" remains relation selection. | `entity_bridge`, `relation_selection`, `entity_bridge` |
| 10 | 51 | Mingus/Arizona/Phoenix/IndyCar path requires birth-state and largest-city comparison; entity-only bridge is insufficient. | `entity_bridge`, `relation_selection`, `comparison_operator` |

## Decision

Stop BEEP as an entity-only retrieval mainline. Do not implement oracle bridge
smoke, non-oracle bridge quality, bridge mention retrieval, or BridgeRAG-lite.

The failure mode is upstream of implementation: MuSiQue 4-hop support chains are
not reliably representable as document-entity co-occurrence paths.

