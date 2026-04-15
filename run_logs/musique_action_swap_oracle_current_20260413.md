# Action-Swap Oracle Ceiling (musique)

- candidate report: `outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260409smoke.json`
- legality_mode: `current`
- reader: `qwen3-8b-train` @ `http://localhost:8043/v1`
- qa_top_k: `5`
- replace_bottom_n: `2`

## Job Generation

- total_queries: `100`
- processed_queries: `100`
- queries_with_appended_candidates: `69`
- queries_without_appended_candidates: `31`
- queries_with_legal_swaps: `16`
- total_keep_jobs: `100`
- total_swap_jobs: `35`
- filtered_nonpositive_score_delta: `293`

## Summary

| View | Count | +EM (%) | +F1 (%) | Mean ΔEM | Mean ΔF1 |
|---|---:|---:|---:|---:|---:|
| legal swaps | 35 | 8.5714 | 20.0 | 0.4286 | 0.5219 |
| oracle best | 100 | 4.0 | 4.0 | 0.02 | 0.0211 |

## Oracle Aggregate

- baseline EM / F1: `0.32` / `0.3727`
- oracle EM / F1: `0.34` / `0.3938`
- oracle delta EM / F1: `0.02` / `0.0211`

## Policy Alignment

| Policy | Covered | Swaps | Keep | Oracle Hit (%) | Oracle+ Precision (%) | Oracle+ Recall (%) |
|---|---:|---:|---:|---:|---:|---:|
| dryrun | 100 | 16 | 84 | 85.0 | 18.75 | 75.0 |
| judge | 100 | 3 | 97 | 94.0 | 33.3333 | 25.0 |

## Top Oracle-Positive Queries

- question: `What city is the star of Sous les pieds des femmes from?`
  keep EM / F1: `0.0` / `0.0`
  oracle action: `swap` `Claudia Cardinale` (replace `Les Bonnes Femmes`)
  oracle delta EM / F1: `1.0` / `1.0`
- question: `When did the 1979-80 European Cup winner win the FA Cup?`
  keep EM / F1: `0.0` / `0.0184`
  oracle action: `swap` `Everton F.C.` (replace `History of Chelsea F.C.`)
  oracle delta EM / F1: `1.0` / `0.9816`
- question: `What is the direction of flow of the body of water by the city where Write This Down was formed?`
  keep EM / F1: `0.0` / `0.0`
  oracle action: `swap` `Ohio River` (replace `Darling Mills Creek`)
  oracle delta EM / F1: `0.0` / `0.1053`
- question: `How did did the people fare during the reign of the abolisher of sati partha in India?`
  keep EM / F1: `0.0` / `0.1538`
  oracle action: `swap` `British Empire` (replace `Indian Rebellion of 1857`)
  oracle delta EM / F1: `0.0` / `0.028`
