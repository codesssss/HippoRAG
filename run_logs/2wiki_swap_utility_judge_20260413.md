# Swap-Utility Judge (2wikimultihopqa)

- baseline report: `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260407.json`
- candidate report: `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260407.json`
- judge: `gpt-5.4` via `responses`
- judge base_url: `https://api.shenfengwl.fun`
- judge reasoning_effort: `medium`
- replace_bottom_n: `2`

## Job Generation

- aligned queries: `100`
- queries with swap jobs: `38`
- total swap jobs: `122`
- max_queries: `100`
- queries_without_appended_candidates: `62`
- replace_bottom_n: `2`

## Judge Summary

- parse failure count / rate: `0` / `0.0`
- helpful / neutral / harmful: `10` / `66` / `46`
- helpful / neutral / harmful rates: `8.1967` / `54.0984` / `37.7049`

## Query Gate

- queries with helpful swap: `5` / `38`
- queries with high-confidence helpful swap: `5` / `38`

## Oracle Alignment

- matched jobs: `61`
- helpful precision EM / F1: `40.0` / `80.0`
- helpful recall EM / F1: `100.0` / `57.1429`
- helpful mean oracle delta EM / F1: `0.4` / `0.5905`
- judge blocks high-CE nonpositive swaps: `28` / `29`

## Top Helpful Swaps

- question: `Which film has the director who was born later, The First Day Of Freedom or Malabimba – The Malicious Whore?`
  candidate: `Aleksander Ford` replace `Lina Wertmüller` (slot `4`, CE rank `5`)
  confidence / type: `98.0` / `bridge_relation`
  reason: `The candidate directly provides Aleksander Ford’s birth date for one of the two queried films’ directors, which is needed to compare directors’ ages, while the incumbent is irrelevant to either film or director in the question.`
- question: `Which film has the director born later, Romance On The Run or The Palace Of Angels?`
  candidate: `Walter Hugo Khouri` replace `Run the Race` (slot `4`, CE rank `6`)
  confidence / type: `98.0` / `bridge_relation`
  reason: `The candidate adds the missing birth date for Walter Hugo Khouri, directly supporting comparison of the two films’ directors, while the incumbent is irrelevant to the question.`
- question: `Who is the father-in-law of John Ernest, Duke Of Saxe-Eisenach?`
  candidate: `Christine of Hesse-Kassel (1578–1658)` replace `Fredericka Elisabeth of Saxe-Eisenach` (slot `5`, CE rank `9`)
  confidence / type: `97.0` / `bridge_relation`
  reason: `The candidate directly supplies the likely spouse of John Ernest, Duke of Saxe-Eisenach and names her father, creating the needed bridge to identify his father-in-law, while the incumbent is an unrelated later family member.`
- question: `Which film has the director born later, Christ Walking On The Water or 45 Fathers?`
  candidate: `James Tinling` replace `Benjamin Christensen` (slot `4`, CE rank `5`)
  confidence / type: `96.0` / `bridge_relation`
  reason: `The candidate directly provides the missing birth year for James Tinling, the director of 45 Fathers, whereas the incumbent is unrelated to either film and does not help compare the directors’ birth dates.`
- question: `Who is the spouse of the director of film My Three Merry Widows?`
  candidate: `Fernando Cortés` replace `My Wife's Best Friend` (slot `4`, CE rank `2`)
  confidence / type: `79.0` / `entity_grounding`
  reason: `The candidate is more query-local because it grounds the director named in the film page (“Fernando Cortés”), which is a needed bridge toward finding his spouse, while the incumbent is an unrelated film entry.`
