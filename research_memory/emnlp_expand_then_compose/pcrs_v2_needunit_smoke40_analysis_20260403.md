# Requirement Beam Diagnostics

## Raw Data Table

| Label | EM | ΔEM | F1 | ΔF1 | R@5 | Avg Support | Avg Leakage | Avg Finalist Leak Range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke40_v2_legacyfact_pool100_ann50_reserve3.json | 0.3500 | -0.0250 | 0.3667 | -0.0349 | 0.5521 | 0.4941 | 0.0999 | 0.0187 |

## Key Findings

1. `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke40_v2_legacyfact_pool100_ann50_reserve3.json`: avg support=0.4941, avg leakage=0.0999, finalist leak range=0.0187, positive-vs-negative separation AUC=0.5161.

## Suggested Next Experiments

- If finalist leak range stays tiny while cache-level positive/negative separation is healthy, change the leakage aggregation before changing counterfactual generation.
- If cache-level positive/negative separation is already poor, prioritize counterfactual/query-requirement construction or coverage scoring before more beam tuning.
- If alternative aggregator rankings recover higher gold recall from the same finalists, test that aggregator in retrieval-only mode before paying reader cost.
