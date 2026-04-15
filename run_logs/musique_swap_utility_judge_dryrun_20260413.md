# Swap-Utility Judge (musique)

- baseline report: `outputs_step0_general_musique/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260409smoke.json`
- candidate report: `outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260409smoke.json`
- judge: `gpt-5.4` via `responses`
- judge base_url: `https://api.shenfengwl.fun`
- judge reasoning_effort: `medium`
- replace_bottom_n: `2`

## Job Generation

- aligned queries: `100`
- queries with swap jobs: `69`
- total swap jobs: `2`
- max_queries: `1`
- queries_without_appended_candidates: `31`
- replace_bottom_n: `2`

## Judge Summary

- parse failure count / rate: `0` / `0.0`
- helpful / neutral / harmful: `0` / `1` / `1`
- helpful / neutral / harmful rates: `0.0` / `50.0` / `50.0`

## Query Gate

- queries with helpful swap: `0` / `1`
- queries with high-confidence helpful swap: `0` / `1`

## Oracle Alignment

- matched jobs: `1`
- helpful precision EM / F1: `None` / `None`
- helpful recall EM / F1: `None` / `None`
- helpful mean oracle delta EM / F1: `None` / `None`
- judge blocks high-CE nonpositive swaps: `0` / `0`

## Top Helpful Swaps

- none
