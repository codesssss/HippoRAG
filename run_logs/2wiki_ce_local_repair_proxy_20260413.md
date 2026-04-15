# CE Local Repair Proxy Analysis (2wikimultihopqa)

- baseline report: `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260407.json`
- repair report: `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_ce_local_repair_qatopk5_20260413smoke.json`
- swap report: `run_logs/2wiki_swap_value_smoke_20260411.json`
- qa_top_k: `5`
- aligned queries: `100`

## Proxy Sanity

- queries with positive req swap: `5`
- positive req swap rate: `5.0`
- mean missing anchor count: `0.34`
- mean scaffold component count: `86.79`
- queries with positive connector gain: `5`
- queries with positive anchor gain: `0`
- queries with positive both: `0`
- anchor-only rate: `0.0`
- connector-only rate: `5.0`

## Repair Effect

- repair applied query rate: `5.0`
- applied swap positive EM rate: `0.0`
- applied swap positive F1 rate: `20.0`
- applied swap avg CE drop vs scaffold: `-1.0876`
- applied swap mean ΔEM / ΔF1: `-0.2` / `-0.2309`

## Swap Overlap

- oracle positive queries: `7`
- proxy positive and oracle positive: `0`
- overlap rate: `0.0`

## Top Repair Cases

- question: `Which country the performer of song I Like Control is from?`
  repair: `Alicia Keys` -> replace `Nathan Sykes` (connector `1`, anchor `0`, CE drop `-0.8271`)
  delta EM/F1: `0.0` / `0.05`
- question: `What nationality is the performer of song When The Stars Go Blue?`
  repair: `Alicia Keys` -> replace `Astrid North` (connector `3`, anchor `0`, CE drop `-1.8834`)
  delta EM/F1: `0.0` / `0.0`
- question: `Which country Aleksander Koniecpolski (1620–1659)'s father is from?`
  repair: `Roman Polanski` -> replace `Aleksander Ford` (connector `1`, anchor `0`, CE drop `-0.4453`)
  delta EM/F1: `0.0` / `0.0`
- question: `What is the place of birth of the performer of song Changed It?`
  repair: `Alicia Keys` -> replace `Alex da Kid` (connector `4`, anchor `0`, CE drop `-0.4033`)
  delta EM/F1: `0.0` / `-0.2045`
- question: `Are both movies, Naked Tango and Algiers (Film), from the same country?`
  repair: `Alain Corneau` -> replace `A Night at the Moulin Rouge` (connector `1`, anchor `0`, CE drop `-1.8789`)
  delta EM/F1: `-1.0` / `-1.0`
