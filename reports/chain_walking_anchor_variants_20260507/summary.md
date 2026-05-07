# Chain-Walking Anchor Variant Probe

This offline diagnostic reuses the chain-walking LLM entity outputs and reranks the fixed MuSiQue pool100 under stricter title-only policies. It makes no LLM or reader calls.

## Policy Comparison

| Policy | New @5 | New @10 | New @20 | unique @5 vs current | lost @5 vs current | mean probe rank | mean improvement | positive-score rate | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| current_all | 9.4% | 16.7% | 17.7% | 0 | 0 | 30.4 | 5.6 | 68.8% | weak_future_work |
| title_strict | 6.2% | 8.3% | 3.1% | 0 | 3 | 35.6 | 0.4 | 9.4% | weak_future_work |
| title_only | 9.4% | 16.7% | 10.4% | 2 | 2 | 34.4 | 1.6 | 22.9% | weak_future_work |
| demand_entity_title | 8.3% | 15.6% | 10.4% | 2 | 3 | 34.9 | 1.1 | 22.9% | weak_future_work |

## Source-Rank Buckets

| Policy | Bucket | Titles | New @5 | New @10 | New @20 | mean improvement |
|---|---|---:|---:|---:|---:|---:|
| current_all | source_rank_51_99 | 29 | 3.4% | 10.3% | 17.2% | 26.3 |
| current_all | source_rank_0_20 | 35 | 17.1% | 22.9% | 2.9% | -7.7 |
| current_all | source_rank_21_50 | 32 | 6.2% | 15.6% | 34.4% | 1.6 |
| title_strict | source_rank_51_99 | 29 | 3.4% | 6.9% | 6.9% | 3.9 |
| title_strict | source_rank_0_20 | 35 | 14.3% | 17.1% | 2.9% | -0.3 |
| title_strict | source_rank_21_50 | 32 | 0.0% | 0.0% | 0.0% | -1.9 |
| title_only | source_rank_51_99 | 29 | 3.4% | 6.9% | 6.9% | 6.8 |
| title_only | source_rank_0_20 | 35 | 17.1% | 20.0% | 2.9% | -3.5 |
| title_only | source_rank_21_50 | 32 | 6.2% | 21.9% | 21.9% | 2.5 |
| demand_entity_title | source_rank_51_99 | 29 | 0.0% | 6.9% | 6.9% | 6.6 |
| demand_entity_title | source_rank_0_20 | 35 | 17.1% | 17.1% | 2.9% | -4.8 |
| demand_entity_title | source_rank_21_50 | 32 | 6.2% | 21.9% | 21.9% | 2.5 |

## Interpretation

- Best New@5 policy: `current_all`.
- If `title_strict` beats `current_all`, the earlier probe was hurt by body/token noise.
- If `demand_entity_title` beats `current_all`, demand-conditioned reformulation has a measurable signal without new LLM calls.
- If all policies stay below 15% New@5, this remains a weak future-work signal rather than a main-method redesign trigger.

## Key Deltas

| Comparison | d New@5 | d New@10 | d New@20 |
|---|---:|---:|---:|
| title_strict - current_all | -3.1% | -8.3% | -14.6% |
| demand_entity_title - current_all | -1.0% | -1.0% | -7.3% |

## Top Improvements For Best Policy

- q281 `Myanmar` rank 83 -> 0 via `Myanmar` from `Star Cola` (title_exact)
- q668 `History of Sacramento, California` rank 73 -> 8 via `California` from `Atwell Mill Grove` (title_substring)
- q954 `Adult contemporary music` rank 72 -> 10 via `NBC` from `The Biggest Loser (season 1)` (body_token_overlap)
- q690 `John C. Petersen` rank 69 -> 8 via `Appleton, Wisconsin` from `Erik Jensen (American football)` (body_substring)
- q735 `WWNQ` rank 88 -> 29 via `South Carolina` from `Zubly Cemetery` (body_substring)
- q568 `Casa Loma` rank 68 -> 12 via `Toronto, Ontario` from `Danko Jones` (body_substring)
- q55 `Saudi Arabia` rank 75 -> 20 via `Kingdom of Saudi Arabia` from `History of Saudi Arabia` (title_substring)
- q704 `Geography of Saudi Arabia` rank 72 -> 22 via `Persian Gulf` from `Battle of Qurah and Umm al Maradim` (body_substring)
- q815 `Adult contemporary music` rank 81 -> 35 via `AVA Radio Company` from `Edward Fokczyński` (body_token_overlap)
- q74 `Mississippi River` rank 50 -> 5 via `Minneapolis, Minnesota` from `Write This Down (band)` (body_token_overlap)
- q946 `Casa Loma` rank 71 -> 27 via `Casa Natal del General Santander` from `Casa Natal del General Santander` (title_token_overlap)
- q358 `North Carolina` rank 73 -> 34 via `Coeburn, Virginia` from `WGCK-FM` (body_token_overlap)

## Files

- Variant rows: `reports/chain_walking_anchor_variants_20260507/variant_rows.csv`
- Policy summary: `reports/chain_walking_anchor_variants_20260507/policy_summary.csv`
- Policy comparison: `reports/chain_walking_anchor_variants_20260507/policy_comparison.csv`
- Full summary: `reports/chain_walking_anchor_variants_20260507/summary.json`

