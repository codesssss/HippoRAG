# Repair-Gated DBEC Reader Evaluation

Reader-only evaluation on repair-gated external pools. Baselines are the frozen SetR-faithful and DBEC-selective per-query results for the same slices.

## Summary

### 2Wiki

| variant | F1 | dF1 vs SetR | dF1 vs DBEC | dF1 vs rank_fill5 | EM | accepted repairs | offline dGold |
|---|---:|---:|---:|---:|---:|---:|---:|
| rq1_mr1_g010_branchstrict | 0.9344 | +0.3015 | +0.0164 | +0.0956 | 0.9344 | 0.943 | +0.164 |
| rq1_mr2_g010_branchstrict | 0.9344 | +0.3015 | +0.0164 | +0.0956 | 0.9344 | 1.279 | +0.221 |
| rq1_mr3_g010_branchstrict | 0.9344 | +0.3015 | +0.0164 | +0.0956 | 0.9344 | 1.279 | +0.221 |

### HotpotQA

| variant | F1 | dF1 vs SetR | dF1 vs DBEC | dF1 vs rank_fill5 | EM | accepted repairs | offline dGold |
|---|---:|---:|---:|---:|---:|---:|---:|
| rq1_mr1_g010_branchstrict | 0.6270 | +0.1091 | +0.0000 | n/a | 0.4688 | 0.750 | +0.062 |
| rq1_mr2_g010_branchstrict | 0.6270 | +0.1091 | +0.0000 | n/a | 0.4688 | 0.844 | +0.062 |
| rq1_mr3_g010_branchstrict | 0.6270 | +0.1091 | +0.0000 | n/a | 0.4688 | 0.844 | +0.062 |

### MuSiQue

| variant | F1 | dF1 vs SetR | dF1 vs DBEC | dF1 vs rank_fill5 | EM | accepted repairs | offline dGold |
|---|---:|---:|---:|---:|---:|---:|---:|
| rq1_mr1_g010_branchstrict | 0.2922 | +0.0648 | -0.0171 | -0.0470 | 0.2264 | 0.755 | +0.019 |
| rq1_mr2_g010_branchstrict | 0.2922 | +0.0648 | -0.0171 | -0.0470 | 0.2264 | 0.868 | +0.019 |
| rq1_mr3_g010_branchstrict | 0.2922 | +0.0648 | -0.0171 | -0.0470 | 0.2264 | 0.868 | +0.019 |

## Paired CI

- 2Wiki `rq1_mr1_g010_branchstrict` vs SetR F1: delta=+0.3015, 95% CI=[+0.2168, +0.3852], p(delta>0)=1.000
- 2Wiki `rq1_mr2_g010_branchstrict` vs SetR F1: delta=+0.3015, 95% CI=[+0.2140, +0.3898], p(delta>0)=1.000
- 2Wiki `rq1_mr3_g010_branchstrict` vs SetR F1: delta=+0.3015, 95% CI=[+0.2158, +0.3907], p(delta>0)=1.000
- HotpotQA `rq1_mr1_g010_branchstrict` vs SetR F1: delta=+0.1091, 95% CI=[+0.0140, +0.2260], p(delta>0)=0.991
- HotpotQA `rq1_mr2_g010_branchstrict` vs SetR F1: delta=+0.1091, 95% CI=[+0.0140, +0.2217], p(delta>0)=0.992
- HotpotQA `rq1_mr3_g010_branchstrict` vs SetR F1: delta=+0.1091, 95% CI=[+0.0143, +0.2245], p(delta>0)=0.992
- MuSiQue `rq1_mr1_g010_branchstrict` vs SetR F1: delta=+0.0648, 95% CI=[+0.0062, +0.1248], p(delta>0)=0.986
- MuSiQue `rq1_mr2_g010_branchstrict` vs SetR F1: delta=+0.0648, 95% CI=[+0.0069, +0.1283], p(delta>0)=0.987
- MuSiQue `rq1_mr3_g010_branchstrict` vs SetR F1: delta=+0.0648, 95% CI=[+0.0066, +0.1283], p(delta>0)=0.986

## Outputs

- Per-query rows: `reports/repair_gated_arbitration_reader_20260507/reader_rows.csv`
- Method summary: `reports/repair_gated_arbitration_reader_20260507/method_summary.csv`
- Paired CI: `reports/repair_gated_arbitration_reader_20260507/paired_ci.csv`
- Full JSON: `reports/repair_gated_arbitration_reader_20260507/summary.json`

