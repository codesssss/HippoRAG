# Action-Swap Oracle Ceiling (2wikimultihopqa)

- candidate report: `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260407.json`
- legality_mode: `relaxed`
- reader: `qwen3-8b-train` @ `http://localhost:8043/v1`
- qa_top_k: `5`
- replace_bottom_n: `2`

## Job Generation

- total_queries: `100`
- processed_queries: `100`
- queries_with_appended_candidates: `38`
- queries_without_appended_candidates: `62`
- queries_with_legal_swaps: `38`
- total_keep_jobs: `100`
- total_swap_jobs: `122`

## Summary

| View | Count | +EM (%) | +F1 (%) | Mean ΔEM | Mean ΔF1 |
|---|---:|---:|---:|---:|---:|
| legal swaps | 122 | 5.7377 | 9.0164 | 0.6364 | 0.6991 |
| oracle best | 100 | 7.0 | 7.0 | 0.05 | 0.0489 |

## Oracle Aggregate

- baseline EM / F1: `0.41` / `0.4647`
- oracle EM / F1: `0.46` / `0.5136`
- oracle delta EM / F1: `0.05` / `0.0489`

## Policy Alignment

| Policy | Covered | Swaps | Keep | Oracle Hit (%) | Oracle+ Precision (%) | Oracle+ Recall (%) |
|---|---:|---:|---:|---:|---:|---:|
| dryrun | 100 | 23 | 77 | 77.0 | 13.0435 | 42.8571 |
| judge | 100 | 4 | 96 | 93.0 | 25.0 | 14.2857 |

## Top Oracle-Positive Queries

- question: `Who is the spouse of the director of film My Three Merry Widows?`
  keep EM / F1: `0.0` / `0.0`
  oracle action: `swap` `Fernando Cortés` (replace `My Wife's Best Friend`)
  oracle delta EM / F1: `1.0` / `1.0`
- question: `Which film has the director who was born later, The First Day Of Freedom or Malabimba – The Malicious Whore?`
  keep EM / F1: `0.0` / `0.0`
  oracle action: `swap` `Roman Polanski` (replace `Lina Wertmüller`)
  oracle delta EM / F1: `1.0` / `1.0`
- question: `Which film has the director born later, Romance On The Run or The Palace Of Angels?`
  keep EM / F1: `0.0` / `0.0`
  oracle action: `swap` `Last Tango in Paris` (replace `A Moment of Romance`)
  oracle delta EM / F1: `1.0` / `1.0`
- question: `Who is the father-in-law of John Ernest, Duke Of Saxe-Eisenach?`
  keep EM / F1: `0.0` / `0.1818`
  oracle action: `swap` `Christine of Hesse-Kassel (1578–1658)` (replace `Wilhelm Heinrich, Duke of Saxe-Eisenach`)
  oracle delta EM / F1: `1.0` / `0.8182`
- question: `Which film has the director born later, Christ Walking On The Water or 45 Fathers?`
  keep EM / F1: `0.0` / `0.8`
  oracle action: `swap` `James Tinling` (replace `Benjamin Christensen`)
  oracle delta EM / F1: `1.0` / `0.2`
- question: `Where was the director of film The Private Life Of Cinema born?`
  keep EM / F1: `0.0` / `0.0`
  oracle action: `swap` `Kim Ki-young` (replace `Roger Corman`)
  oracle delta EM / F1: `0.0` / `0.6667`
- question: `What is the place of birth of the performer of song Changed It?`
  keep EM / F1: `0.0` / `0.5455`
  oracle action: `swap` `Jay-Z` (replace `Alex da Kid`)
  oracle delta EM / F1: `0.0` / `0.2045`
