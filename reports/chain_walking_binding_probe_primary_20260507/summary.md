# Chain-Walking Binding Probe

This diagnostic tests whether upstream-document entity extraction can move MuSiQue missing gold support titles from deep fixed-pool ranks into top-k. It does not call the reader and does not change DBEC selection.

## Overall

| Metric | Value |
|---|---:|
| Target queries | 62 |
| Target missing gold titles | 86 |
| Prompt rows | 324 |
| Parse ok rate | 100.0% |
| Entity outputs | 338 |
| New Recall@5 | 10.5% |
| New Recall@10 | 18.6% |
| New Recall@20 | 19.8% |
| Mean source rank | 34.4 |
| Mean probe rank | 28.5 |
| Mean rank improvement | 5.9 |
| Decision | weak_future_work |

## Rank Shift

| Bucket | Titles | New @5 | New @10 | New @20 | mean source rank | mean probe rank | mean improvement | positive-score rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 86 | 10.5% | 18.6% | 19.8% | 34.4 | 28.5 | 5.9 | 70.9% |
| source_rank_0_20 | 34 | 17.6% | 23.5% | 2.9% | 10.8 | 18.6 | -7.9 | 73.5% |
| source_rank_51_99 | 24 | 4.2% | 12.5% | 20.8% | 73.8 | 46.2 | 27.6 | 70.8% |
| source_rank_21_50 | 28 | 7.1% | 17.9% | 39.3% | 29.3 | 25.3 | 4.0 | 67.9% |

## Top Rank Improvements

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

- Probe rows: `reports/chain_walking_binding_probe_primary_20260507/probe_rows.csv`
- Entity outputs: `reports/chain_walking_binding_probe_primary_20260507/entity_outputs.jsonl`
- Rank summary: `reports/chain_walking_binding_probe_primary_20260507/rank_shift_summary.csv`
- Full summary: `reports/chain_walking_binding_probe_primary_20260507/summary.json`

