# Swap-Value Smoke Test (musique)

- baseline report: `outputs_step0_general_musique/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260409smoke.json`
- candidate report: `outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260409smoke.json`
- reader: `qwen3-8b-train` @ `http://localhost:8043/v1`
- qa_top_k: `5`

## Job Generation

- aligned queries: `100`
- queries with swap jobs: `69`
- total swap jobs: `164`
- queries_without_appended_candidates: `31`

## Summary

| View | Count | +EM (%) | +F1 (%) | Mean ΔEM | Mean ΔF1 |
|---|---:|---:|---:|---:|---:|
| all swaps | 164 | 9.1463 | 14.6341 | 0.0671 | 0.0675 |
| best-swap oracle | 69 | 14.4928 | 20.2899 | 0.1305 | 0.1457 |

## Oracle Aggregate

- baseline EM / F1: `0.3043` / `0.3673`
- oracle EM / F1: `0.4348` / `0.513`
- oracle delta EM / F1: `0.1305` / `0.1457`

## Top Positive Best-Swap Queries

- question: `When was the Palau de la Generalitat constructed in the city where Martin from the region where Perdiguera is located died?`
  baseline -> swap: `not available.` -> `15th century.`
  candidate: `Utrecht` (CE rank `6`, replace `Josep Domènech i Estapà`)
  delta EM / F1: `1.0` / `1.0`
- question: `How many times did the plague occur in the city where the painter of The Bacchanal of the Andrians died?`
  baseline -> swap: `Cannot be determined.` -> `22.`
  candidate: `Portrait of Lavinia Vecellio` (CE rank `12`, replace `Black Death`)
  delta EM / F1: `1.0` / `1.0`
- question: `Where did the arguer that the country Directive 10/2 called for actions against had become an imperialist power declare he would intervene in the Korean conflict?`
  baseline -> swap: `Korean War.` -> `Politburo.`
  candidate: `Cold War` (CE rank `9`, replace `Korean War`)
  delta EM / F1: `1.0` / `1.0`
- question: `What city is the star of Sous les pieds des femmes from?`
  baseline -> swap: `not present.` -> `Tunis.`
  candidate: `Claudia Cardinale` (CE rank `2`, replace `C'est les vacances`)
  delta EM / F1: `1.0` / `1.0`
- question: `When did the country the top-ranking Warsaw Pact operatives came from, despite it being headquartered in the country where A Generation is set, agree to a unified Germany inside NATO?`
  baseline -> swap: `1955.` -> `1990.`
  candidate: `Dissolution of the Soviet Union` (CE rank `12`, replace `Warsaw Pact`)
  delta EM / F1: `1.0` / `1.0`
- question: `When did the 1979-80 European Cup winner win the FA Cup?`
  baseline -> swap: `<short answer>". The user might expect the most recent one before the European Cup win, but the question isn't specifying which one. However, the answer should list both years. Wait, but the answer format is a single line. Let me check the original entries again.` -> `1898 and 1959.`
  candidate: `List of Chelsea F.C. managers` (CE rank `4`, replace `History of Chelsea F.C.`)
  delta EM / F1: `1.0` / `1.0`
- question: `Who played the girlfriend of who plays marty mcfly's daughter in back to the future 2?`
  baseline -> swap: `Not specified in the provided text.` -> `Claudia Wells.`
  candidate: `Ghosts of Girlfriends Past` (CE rank `11`, replace `List of Back to the Future characters`)
  delta EM / F1: `1.0` / `1.0`
- question: `What is the meaning of the name of the city where the Yongle emperor greeted the person to whom the edict was addressed?`
  baseline -> swap: `The text does not provide the meaning of the city's name. But the user might be expecting "Southern Capital." However, strictly following the instructions, the answer is not present. But the user's question might be designed to have the answer "Southern Capital.` -> `Southern Capital.`
  candidate: `Meiji (era)` (CE rank `10`, replace `Sino-Tibetan relations during the Ming dynasty`)
  delta EM / F1: `1.0` / `0.8889`
- question: `What is the position of the 1st governor general of India?`
  baseline -> swap: `Governor of the Presidency of Fort William (Bengal).` -> `Governor-General of India.`
  candidate: `Nawabs of Bengal and Murshidabad` (CE rank `7`, replace `Governor-General of India`)
  delta EM / F1: `1.0` / `0.8`
- question: `Who stars in the video "One Last Time" by the performer of Baby I?`
  baseline -> swap: `Matt Bennett and Elizabeth Gillies.` -> `Matt Bennett.`
  candidate: `Make It Last Forever (song)` (CE rank `11`, replace `Cry Me a River (Justin Timberlake song)`)
  delta EM / F1: `1.0` / `0.4286`
