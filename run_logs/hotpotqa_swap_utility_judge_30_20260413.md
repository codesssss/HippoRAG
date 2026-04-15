# Swap-Utility Judge (hotpotqa)

- baseline report: `outputs_step0_general_hotpotqa/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260407.json`
- candidate report: `outputs_step0_general_hotpotqa/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260407.json`
- judge: `gpt-5.4` via `responses`
- judge base_url: `https://api.shenfengwl.fun`
- judge reasoning_effort: `medium`
- replace_bottom_n: `2`

## Job Generation

- aligned queries: `100`
- queries with swap jobs: `38`
- total swap jobs: `98`
- max_queries: `30`
- queries_without_appended_candidates: `62`
- replace_bottom_n: `2`

## Judge Summary

- parse failure count / rate: `5` / `5.102`
- helpful / neutral / harmful: `4` / `18` / `71`
- helpful / neutral / harmful rates: `4.0816` / `18.3673` / `72.449`

## Query Gate

- queries with helpful swap: `2` / `30`
- queries with high-confidence helpful swap: `2` / `30`

## Top Helpful Swaps

- question: `What age was Georgia Middleman when she started singing in the seventh-most populated city in the United States?`
  candidate: `San Antonio` replace `Chris Medina` (slot `4`, CE rank `2`)
  confidence / type: `98.0` / `bridge_relation`
  reason: `The candidate supplies the missing bridge that San Antonio is the seventh-most populous U.S. city, directly linking the location in the Georgia Middleman evidence to the wording of the question, while the incumbent is irrelevant.`
- question: `In what year was the composer of "Anthem" born?`
  candidate: `Tim Rice` replace `Anthony Burgess` (slot `5`, CE rank `3`)
  confidence / type: `90.0` / `bridge_relation`
  reason: `Replacing the irrelevant Anthony Burgess page with Tim Rice adds a directly linked creator of “Anthem” and his birth year, which is much more useful for bridging from the song to a likely answer entity.`
