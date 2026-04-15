# Action-Swap Oracle Ceiling (musique)

- candidate report: `outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260409smoke.json`
- legality_mode: `relaxed`
- reader: `qwen3-8b-train` @ `http://localhost:8043/v1`
- qa_top_k: `5`
- replace_bottom_n: `2`

## Job Generation

- total_queries: `100`
- processed_queries: `100`
- queries_with_appended_candidates: `69`
- queries_without_appended_candidates: `31`
- queries_with_legal_swaps: `69`
- total_keep_jobs: `100`
- total_swap_jobs: `328`

## Summary

| View | Count | +EM (%) | +F1 (%) | Mean ΔEM | Mean ΔF1 |
|---|---:|---:|---:|---:|---:|
| legal swaps | 328 | 5.7927 | 10.6707 | 0.5429 | 0.6576 |
| oracle best | 100 | 13.0 | 13.0 | 0.08 | 0.0961 |

## Oracle Aggregate

- baseline EM / F1: `0.32` / `0.3727`
- oracle EM / F1: `0.4` / `0.4688`
- oracle delta EM / F1: `0.08` / `0.0961`

## Policy Alignment

| Policy | Covered | Swaps | Keep | Oracle Hit (%) | Oracle+ Precision (%) | Oracle+ Recall (%) |
|---|---:|---:|---:|---:|---:|---:|
| dryrun | 100 | 16 | 84 | 76.0 | 18.75 | 23.0769 |
| judge | 100 | 3 | 97 | 85.0 | 33.3333 | 7.6923 |

## Top Oracle-Positive Queries

- question: `How many times did the plague occur in the city where the painter of The Bacchanal of the Andrians died?`
  keep EM / F1: `0.0` / `0.0`
  oracle action: `swap` `Huns` (replace `Black Death`)
  oracle delta EM / F1: `1.0` / `1.0`
- question: `Where did the arguer that the country Directive 10/2 called for actions against had become an imperialist power declare he would intervene in the Korean conflict?`
  keep EM / F1: `0.0` / `0.0`
  oracle action: `swap` `Cold War` (replace `Korean War`)
  oracle delta EM / F1: `1.0` / `1.0`
- question: `Who is played by the director of The Good Shepherd in The Godfather?`
  keep EM / F1: `0.0` / `0.0`
  oracle action: `swap` `The Bourne Legacy (film)` (replace `Tom Hagen`)
  oracle delta EM / F1: `1.0` / `1.0`
- question: `What city is the star of Sous les pieds des femmes from?`
  keep EM / F1: `0.0` / `0.0`
  oracle action: `swap` `Claudia Cardinale` (replace `Les Bonnes Femmes`)
  oracle delta EM / F1: `1.0` / `1.0`
- question: `When was the region immediately north of the region where the country in which Aluf can be found is located and the Persian Gulf established?`
  keep EM / F1: `0.0` / `0.0`
  oracle action: `swap` `Abdul Rahman bin Faisal` (replace `Arabian Peninsula`)
  oracle delta EM / F1: `1.0` / `1.0`
- question: `When did the 1979-80 European Cup winner win the FA Cup?`
  keep EM / F1: `0.0` / `0.0184`
  oracle action: `swap` `Everton F.C.` (replace `History of Chelsea F.C.`)
  oracle delta EM / F1: `1.0` / `0.9816`
- question: `What is the position of the 1st governor general of India?`
  keep EM / F1: `0.0` / `0.2`
  oracle action: `swap` `Nawabs of Bengal and Murshidabad` (replace `Governor-General of India`)
  oracle delta EM / F1: `1.0` / `0.8`
- question: `Who stars in the video "One Last Time" by the performer of Baby I?`
  keep EM / F1: `0.0` / `0.5714`
  oracle action: `swap` `Make It Last Forever (song)` (replace `Cry Me a River (Justin Timberlake song)`)
  oracle delta EM / F1: `1.0` / `0.4286`
- question: `Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`
  keep EM / F1: `0.0` / `0.0`
  oracle action: `swap` `Riverside Plaza` (replace `Gulf of Mexico`)
  oracle delta EM / F1: `0.0` / `0.8`
- question: `Who had the lowest batting average in the league where the team with the most games in the series after which the MLB MVP is awarded played?`
  keep EM / F1: `0.0` / `0.0`
  oracle action: `swap` `Opening Day` (replace `World Series Most Valuable Player Award`)
  oracle delta EM / F1: `0.0` / `0.8`
