# Chain-Walking Binding Probe

This diagnostic tests whether upstream-document entity extraction can move MuSiQue missing gold support titles from deep fixed-pool ranks into top-k. It does not call the reader and does not change DBEC selection.

## Overall

| Metric | Value |
|---|---:|
| Target queries | 72 |
| Target missing gold titles | 96 |
| Prompt rows | 384 |
| Parse ok rate | 99.0% |
| Entity outputs | 396 |
| New Recall@5 | 9.4% |
| New Recall@10 | 16.7% |
| New Recall@20 | 17.7% |
| Mean source rank | 36.0 |
| Mean probe rank | 30.4 |
| Mean rank improvement | 5.6 |
| Decision | weak_future_work |

## Rank Shift

| Bucket | Titles | New @5 | New @10 | New @20 | mean source rank | mean probe rank | mean improvement | positive-score rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 96 | 9.4% | 16.7% | 17.7% | 36.0 | 30.4 | 5.6 | 68.8% |
| source_rank_51_99 | 29 | 3.4% | 10.3% | 17.2% | 73.6 | 47.3 | 26.3 | 69.0% |
| source_rank_0_20 | 35 | 17.1% | 22.9% | 2.9% | 10.9 | 18.7 | -7.7 | 74.3% |
| source_rank_21_50 | 32 | 6.2% | 15.6% | 34.4% | 29.4 | 27.9 | 1.6 | 62.5% |

## Top Rank Improvements

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

- Probe rows: `reports/chain_walking_binding_probe_20260507/probe_rows.csv`
- Entity outputs: `reports/chain_walking_binding_probe_20260507/entity_outputs.jsonl`
- Rank summary: `reports/chain_walking_binding_probe_20260507/rank_shift_summary.csv`
- Full summary: `reports/chain_walking_binding_probe_20260507/summary.json`

