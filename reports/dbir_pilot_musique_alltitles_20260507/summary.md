# D-BIR MuSiQue Pilot

This pilot uses a local lexical corpus retriever. It is a structural feasibility test, not the final PropRAG-substrate experiment. Expansion metrics are computed on expansion-produced pools only; the original fixed pool is reported as a baseline and is not unioned into D-BIR outputs.

## Decision

- Decision: `stop_or_pivot`
- Query-primary Slice A: `False`
- Top-m per retrieval call: `10`
- Expanded K: `100`
- Max iterations: `4`

## Policy Summary

| Slice | Policy | Titles | Q | Exp@50 | Exp@100 | Final New@5 | Pool-absent@100 | calls/q | slot coverage | rank improvement |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| source_pool_absent | fixed_pool_top100 | 25 | 19 | 0.0% | 0.0% | 0.0% | 0.0% | 0.00 | 0.0% | 0.0 |
| source_pool_absent | rank_expansion | 25 | 19 | 20.0% | 28.0% | 8.0% | 28.0% | 1.00 | 0.0% | 0.0 |
| source_pool_absent | independent_demand | 25 | 19 | 28.0% | 28.0% | 8.0% | 28.0% | 3.12 | 100.0% | 0.0 |
| source_pool_absent | context_iterative_lite | 25 | 19 | 12.0% | 12.0% | 8.0% | 12.0% | 4.00 | 0.0% | 0.0 |
| source_pool_absent | dbir_det | 25 | 19 | 12.0% | 12.0% | 8.0% | 12.0% | 3.12 | 100.0% | 0.0 |
| source_visible_not_rank_or_dbec | fixed_pool_top100 | 96 | 72 | 72.9% | 100.0% | 2.1% | 0.0% | 0.00 | 0.0% | 4.0 |
| source_visible_not_rank_or_dbec | rank_expansion | 96 | 72 | 14.6% | 20.8% | 5.2% | 0.0% | 1.00 | 0.0% | -7.4 |
| source_visible_not_rank_or_dbec | independent_demand | 96 | 72 | 8.3% | 8.3% | 2.1% | 0.0% | 2.95 | 99.7% | 19.0 |
| source_visible_not_rank_or_dbec | context_iterative_lite | 96 | 72 | 19.8% | 19.8% | 5.2% | 0.0% | 4.00 | 0.0% | 4.5 |
| source_visible_not_rank_or_dbec | dbir_det | 96 | 72 | 16.7% | 16.7% | 2.1% | 0.0% | 2.95 | 100.0% | 3.2 |

## Interpretation Rules

- Strong go requires D-BIR to beat the equal-budget context-iteration proxy, reach strong Slice-A expanded recall/final New@5, and recover nontrivial Slice-B pool-absent gold.
- If D-BIR only matches independent demand retrieval, dependency-bound control is not yet contributing enough.
- If expanded recall is high but Final New@5 is weak, the next bottleneck is composition rather than expansion.

## Files

- Expanded rows: `reports/dbir_pilot_musique_alltitles_20260507/expanded_pool_rows.csv`
- Query traces: `reports/dbir_pilot_musique_alltitles_20260507/slot_trace.jsonl`
- Policy summary: `reports/dbir_pilot_musique_alltitles_20260507/policy_summary.csv`
- Full summary: `reports/dbir_pilot_musique_alltitles_20260507/summary.json`

