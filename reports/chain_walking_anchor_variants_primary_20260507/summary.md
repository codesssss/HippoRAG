# Chain-Walking Anchor Variant Probe

This offline diagnostic reuses the chain-walking LLM entity outputs and reranks the fixed MuSiQue pool100 under stricter title-only policies. It makes no LLM or reader calls.

## Policy Comparison

| Policy | New @5 | New @10 | New @20 | unique @5 vs current | lost @5 vs current | mean probe rank | mean improvement | positive-score rate | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| current_all | 10.5% | 18.6% | 19.8% | 0 | 0 | 28.5 | 5.9 | 70.9% | weak_future_work |
| title_strict | 7.0% | 9.3% | 3.5% | 0 | 3 | 33.8 | 0.6 | 10.5% | weak_future_work |
| title_only | 10.5% | 18.6% | 11.6% | 2 | 2 | 32.3 | 2.0 | 25.6% | weak_future_work |
| demand_entity_title | 9.3% | 17.4% | 11.6% | 2 | 3 | 33.0 | 1.4 | 25.6% | weak_future_work |

## Source-Rank Buckets

| Policy | Bucket | Titles | New @5 | New @10 | New @20 | mean improvement |
|---|---|---:|---:|---:|---:|---:|
| current_all | source_rank_0_20 | 34 | 17.6% | 23.5% | 2.9% | -7.9 |
| current_all | source_rank_51_99 | 24 | 4.2% | 12.5% | 20.8% | 27.6 |
| current_all | source_rank_21_50 | 28 | 7.1% | 17.9% | 39.3% | 4.0 |
| title_strict | source_rank_0_20 | 34 | 14.7% | 17.6% | 2.9% | -0.3 |
| title_strict | source_rank_51_99 | 24 | 4.2% | 8.3% | 8.3% | 4.8 |
| title_strict | source_rank_21_50 | 28 | 0.0% | 0.0% | 0.0% | -2.0 |
| title_only | source_rank_0_20 | 34 | 17.6% | 20.6% | 2.9% | -3.4 |
| title_only | source_rank_51_99 | 24 | 4.2% | 8.3% | 8.3% | 8.4 |
| title_only | source_rank_21_50 | 28 | 7.1% | 25.0% | 25.0% | 3.2 |
| demand_entity_title | source_rank_0_20 | 34 | 17.6% | 17.6% | 2.9% | -4.7 |
| demand_entity_title | source_rank_51_99 | 24 | 0.0% | 8.3% | 8.3% | 8.2 |
| demand_entity_title | source_rank_21_50 | 28 | 7.1% | 25.0% | 25.0% | 3.1 |

## Interpretation

- Best New@5 policy: `current_all`.
- If `title_strict` beats `current_all`, the earlier probe was hurt by body/token noise.
- If `demand_entity_title` beats `current_all`, demand-conditioned reformulation has a measurable signal without new LLM calls.
- If all policies stay below 15% New@5, this remains a weak future-work signal rather than a main-method redesign trigger.

## Key Deltas

| Comparison | d New@5 | d New@10 | d New@20 |
|---|---:|---:|---:|
| title_strict - current_all | -3.5% | -9.3% | -16.3% |
| demand_entity_title - current_all | -1.2% | -1.2% | -8.1% |

## Top Improvements For Best Policy

- q281 `Myanmar` rank 83 -> 0 via `Myanmar` from `Star Cola` (title_exact)
- q668 `History of Sacramento, California` rank 73 -> 8 via `California` from `Atwell Mill Grove` (title_substring)
- q954 `Adult contemporary music` rank 72 -> 10 via `NBC` from `The Biggest Loser (season 1)` (body_token_overlap)
- q690 `John C. Petersen` rank 69 -> 8 via `Appleton, Wisconsin` from `Erik Jensen (American football)` (body_substring)
- q568 `Casa Loma` rank 68 -> 12 via `Toronto, Ontario` from `Danko Jones` (body_substring)
- q55 `Saudi Arabia` rank 75 -> 20 via `Kingdom of Saudi Arabia` from `History of Saudi Arabia` (title_substring)
- q704 `Geography of Saudi Arabia` rank 72 -> 22 via `Persian Gulf` from `Battle of Qurah and Umm al Maradim` (body_substring)
- q815 `Adult contemporary music` rank 81 -> 35 via `AVA Radio Company` from `Edward Fokczyński` (body_token_overlap)
- q74 `Mississippi River` rank 50 -> 5 via `Minneapolis, Minnesota` from `Write This Down (band)` (body_token_overlap)
- q946 `Casa Loma` rank 71 -> 27 via `Casa Natal del General Santander` from `Casa Natal del General Santander` (title_token_overlap)
- q358 `North Carolina` rank 73 -> 34 via `Coeburn, Virginia` from `WGCK-FM` (body_token_overlap)
- q647 `Jews` rank 79 -> 42 via `United State` from `Space Race` (body_token_overlap)

## Files

- Variant rows: `reports/chain_walking_anchor_variants_primary_20260507/variant_rows.csv`
- Policy summary: `reports/chain_walking_anchor_variants_primary_20260507/policy_summary.csv`
- Policy comparison: `reports/chain_walking_anchor_variants_primary_20260507/policy_comparison.csv`
- Full summary: `reports/chain_walking_anchor_variants_primary_20260507/summary.json`

