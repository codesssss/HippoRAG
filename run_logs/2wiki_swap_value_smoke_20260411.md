# Swap-Value Smoke Test (2wikimultihopqa)

- baseline report: `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260407.json`
- candidate report: `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260407.json`
- reader: `qwen3-8b-train` @ `http://localhost:8043/v1`
- qa_top_k: `5`

## Job Generation

- aligned queries: `100`
- queries with swap jobs: `38`
- total swap jobs: `61`
- queries_without_appended_candidates: `62`

## Summary

| View | Count | +EM (%) | +F1 (%) | Mean ΔEM | Mean ΔF1 |
|---|---:|---:|---:|---:|---:|
| all swaps | 61 | 3.2787 | 11.4754 | -0.0328 | -0.0011 |
| best-swap oracle | 38 | 5.2632 | 18.4211 | -0.0264 | 0.0353 |

## Oracle Aggregate

- baseline EM / F1: `0.2632` / `0.3167`
- oracle EM / F1: `0.2368` / `0.352`
- oracle delta EM / F1: `-0.0264` / `0.0353`

## Top Positive Best-Swap Queries

- question: `Who is the father-in-law of John Ernest, Duke Of Saxe-Eisenach?`
  baseline -> swap: `that the father-in-law is not` -> `William IV, Landgrave of Hesse-Kassel.`
  candidate: `Christine of Hesse-Kassel (1578–1658)` (CE rank `9`, replace `Fredericka Elisabeth of Saxe-Eisenach`)
  delta EM / F1: `1.0` / `1.0`
- question: `Who is the spouse of the director of film My Three Merry Widows?`
  baseline -> swap: `Not mentioned.` -> `Mapy Cortés.`
  candidate: `Fernando Cortés` (CE rank `2`, replace `The Very Merry Widows`)
  delta EM / F1: `1.0` / `1.0`
- question: `Where was the director of film The Private Life Of Cinema born?`
  baseline -> swap: `that the information is not provided in the given text.` -> `Quebec.`
  candidate: `Kim Ki-young` (CE rank `4`, replace `Rudolph Maté`)
  delta EM / F1: `0.0` / `0.6667`
- question: `Which film has the director born later, Christ Walking On The Water or 45 Fathers?`
  baseline -> swap: `Cannot be determined.` -> `45 Fathers" because the director's`
  candidate: `James Tinling` (CE rank `5`, replace `The Brand New Testament`)
  delta EM / F1: `0.0` / `0.6667`
- question: `Which film has the director who is older than the other, Airheads or Return To Cabin By The Lake? `
  baseline -> swap: `based on the given data.` -> `that the director of "Return to Cabin by the Lake" is older, but I can't confirm.`
  candidate: `Jingle All the Way` (CE rank `4`, replace `The Return of Swamp Thing`)
  delta EM / F1: `0.0` / `0.5263`
- question: `Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Bråk?`
  baseline -> swap: `Aldri Annet Enn Bråk.` -> `that the director of "God's Gift to Women" is older, but again, without data, this is uncertain.`
  candidate: `Ingmar Bergman` (CE rank `5`, replace `Lasse Hallström`)
  delta EM / F1: `0.0` / `0.4`
- question: `Which film has the director born later, Romance On The Run or The Palace Of Angels?`
  baseline -> swap: `that there's not enough information.` -> `The Palace of Angels" because the director's birth year is 1929, and the other director's birth year is not given, so`
  candidate: `Walter Hugo Khouri` (CE rank `6`, replace `A Moment of Romance`)
  delta EM / F1: `0.0` / `0.2857`
