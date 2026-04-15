# Swap-Value Smoke Test (musique)

- baseline report: `outputs_step0_general_musique/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260409smoke.json`
- candidate report: `outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260409smoke.json`
- reader: `qwen3-8b-train` @ `http://localhost:8043/v1`
- qa_top_k: `5`

## Job Generation

- aligned queries: `100`
- queries with swap jobs: `69`
- total swap jobs: `1`
- queries_without_appended_candidates: `31`

## Summary

| View | Count | +EM (%) | +F1 (%) | Mean ΔEM | Mean ΔF1 |
|---|---:|---:|---:|---:|---:|
| all swaps | 1 | 0.0 | 0.0 | 0.0 | 0.0 |
| best-swap oracle | 1 | 0.0 | 0.0 | 0.0 | 0.0 |

## Oracle Aggregate

- baseline EM / F1: `1.0` / `1.0`
- oracle EM / F1: `1.0` / `1.0`
- oracle delta EM / F1: `0.0` / `0.0`

## Top Positive Best-Swap Queries

- none
