# Repair-Gated DBEC Offline Arbitration

This is an offline support/replacement diagnostic. It does not report reader EM/F1 for simulated final-5 pools.

## Setup

- Top-k budget: `5`
- Seed: SetR-faithful selected positions.
- Repair proposer: DBEC selective `selection_steps` candidates.
- Rank fallback: original PropRAG pool order excluding already selected positions.
- Admission rule: accept DBEC candidates only when the rank quota, max repair count, min gain, duplicate-title filter, and optional branch filter allow it.

## Slices

- 2Wiki: N=122, `gold_doc_count>=4 and SetR count-underselected`
- MuSiQue: N=106, `gold_doc_count>=3 and SetR count-underselected`
- HotpotQA: N=32, `SetR count-underselected negative/control slice`

## Best Variants By Dataset

### 2Wiki

| variant | dComplete vs rank | dRecall vs rank | harmful | repairs | min_gain | branch |
|---|---:|---:|---:|---:|---:|---|
| rq1_mr2_g010_branchstrict | +0.1885 | +0.0553 | 0.0656 | 1.279 | 0.10 | strict |
| rq1_mr3_g010_branchstrict | +0.1885 | +0.0553 | 0.0656 | 1.279 | 0.10 | strict |
| rq1_mr2_g010_branchoff | +0.1803 | +0.0533 | 0.0738 | 1.287 | 0.10 | off |
| rq1_mr3_g010_branchoff | +0.1803 | +0.0533 | 0.0738 | 1.287 | 0.10 | off |
| rq1_mr2_g005_branchstrict | +0.1803 | +0.0533 | 0.0738 | 1.336 | 0.05 | strict |
| rq1_mr3_g005_branchstrict | +0.1803 | +0.0533 | 0.0738 | 1.336 | 0.05 | strict |
| rq1_mr2_g005_branchoff | +0.1721 | +0.0512 | 0.0820 | 1.352 | 0.05 | off |
| rq1_mr3_g005_branchoff | +0.1721 | +0.0512 | 0.0820 | 1.352 | 0.05 | off |

### HotpotQA

| variant | dComplete vs rank | dRecall vs rank | harmful | repairs | min_gain | branch |
|---|---:|---:|---:|---:|---:|---|
| rq1_mr1_g010_branchoff | +0.0625 | +0.0312 | 0.0000 | 0.750 | 0.10 | off |
| rq1_mr1_g010_branchstrict | +0.0625 | +0.0312 | 0.0000 | 0.750 | 0.10 | strict |
| rq2_mr1_g010_branchoff | +0.0625 | +0.0312 | 0.0000 | 0.750 | 0.10 | off |
| rq2_mr1_g010_branchstrict | +0.0625 | +0.0312 | 0.0000 | 0.750 | 0.10 | strict |
| rq1_mr1_g005_branchoff | +0.0625 | +0.0312 | 0.0000 | 0.781 | 0.05 | off |
| rq1_mr1_g005_branchstrict | +0.0625 | +0.0312 | 0.0000 | 0.781 | 0.05 | strict |
| rq2_mr1_g005_branchoff | +0.0625 | +0.0312 | 0.0000 | 0.781 | 0.05 | off |
| rq2_mr1_g005_branchstrict | +0.0625 | +0.0312 | 0.0000 | 0.781 | 0.05 | strict |

### MuSiQue

| variant | dComplete vs rank | dRecall vs rank | harmful | repairs | min_gain | branch |
|---|---:|---:|---:|---:|---:|---|
| rq1_mr1_g010_branchstrict | -0.0094 | +0.0047 | 0.0377 | 0.755 | 0.10 | strict |
| rq1_mr1_g010_branchoff | -0.0094 | +0.0047 | 0.0377 | 0.764 | 0.10 | off |
| rq1_mr1_g005_branchstrict | -0.0094 | +0.0047 | 0.0377 | 0.764 | 0.05 | strict |
| rq1_mr1_g003_branchstrict | -0.0094 | +0.0047 | 0.0377 | 0.774 | 0.03 | strict |
| rq1_mr1_g005_branchoff | -0.0094 | +0.0047 | 0.0377 | 0.783 | 0.05 | off |
| rq1_mr1_g003_branchoff | -0.0094 | +0.0047 | 0.0377 | 0.792 | 0.03 | off |
| rq1_mr1_g000_branchstrict | -0.0094 | +0.0047 | 0.0377 | 0.792 | 0.00 | strict |
| rq1_mr1_g000_branchoff | -0.0094 | +0.0047 | 0.0377 | 0.811 | 0.00 | off |

## CI Highlights

- 2Wiki best `rq1_mr2_g010_branchstrict` vs rank_fill5:
  - support_recall: delta=+0.0553, 95% CI=[+0.0287, +0.0820], p(delta>0)=1.000
  - support_complete: delta=+0.1885, 95% CI=[+0.0984, +0.2869], p(delta>0)=1.000
- HotpotQA best `rq1_mr1_g010_branchoff` vs rank_fill5:
  - support_recall: delta=+0.0312, 95% CI=[+0.0000, +0.0781], p(delta>0)=0.873
  - support_complete: delta=+0.0625, 95% CI=[+0.0000, +0.1562], p(delta>0)=0.867
- MuSiQue best `rq1_mr1_g010_branchstrict` vs rank_fill5:
  - support_recall: delta=+0.0047, 95% CI=[-0.0118, +0.0220], p(delta>0)=0.689
  - support_complete: delta=-0.0094, 95% CI=[-0.0472, +0.0189], p(delta>0)=0.185

## Outputs

- Per-query audit: `reports/repair_gated_arbitration_offline_20260507/replacement_audit.csv`
- Variant summary: `reports/repair_gated_arbitration_offline_20260507/variant_support_summary.csv`
- Paired support CI: `reports/repair_gated_arbitration_offline_20260507/paired_support_ci.csv`
- Full JSON: `reports/repair_gated_arbitration_offline_20260507/summary.json`

