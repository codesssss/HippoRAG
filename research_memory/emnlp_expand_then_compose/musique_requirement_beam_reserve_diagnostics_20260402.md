# Requirement Beam Diagnostics

## Raw Data Table

| Label | EM | ΔEM | F1 | ΔF1 | R@5 | Avg Support | Avg Leakage | Avg Finalist Leak Range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| outputs_step0_general_musique/eval_reports/requirement_beam_oracle_smoke40_v2_legacyfact_pool100_ann50_reserve3.json | 0.3250 | -0.0500 | 0.3753 | -0.0263 | 0.5208 | 0.5531 | 0.9511 | 0.0065 |
| outputs_step0_general_musique/eval_reports/requirement_beam_oracle_smoke40_v2_legacyfact_pool100_ann50_reserve1.json | 0.1750 | -0.2000 | 0.2079 | -0.1937 | 0.3812 | 0.6065 | 0.9503 | 0.0046 |

## Key Findings

1. `outputs_step0_general_musique/eval_reports/requirement_beam_oracle_smoke40_v2_legacyfact_pool100_ann50_reserve3.json`: avg support=0.5531, avg leakage=0.9511, finalist leak range=0.0065, positive-vs-negative separation AUC=0.4988.
2. `outputs_step0_general_musique/eval_reports/requirement_beam_oracle_smoke40_v2_legacyfact_pool100_ann50_reserve1.json`: avg support=0.6065, avg leakage=0.9503, finalist leak range=0.0046, positive-vs-negative separation AUC=0.4988.
3. Pairwise `outputs_step0_general_musique/eval_reports/requirement_beam_oracle_smoke40_v2_legacyfact_pool100_ann50_reserve3.json` -> `outputs_step0_general_musique/eval_reports/requirement_beam_oracle_smoke40_v2_legacyfact_pool100_ann50_reserve1.json`: support mean delta=+0.0547, F1 mean delta=-0.1674, gold-recall mean delta=-0.1396, corr(supportΔ,F1Δ)=+0.0304.

## Suggested Next Experiments

- If finalist leak range stays tiny while cache-level positive/negative separation is healthy, change the leakage aggregation before changing counterfactual generation.
- If cache-level positive/negative separation is already poor, prioritize counterfactual/query-requirement construction or coverage scoring before more beam tuning.
- If alternative aggregator rankings recover higher gold recall from the same finalists, test that aggregator in retrieval-only mode before paying reader cost.
