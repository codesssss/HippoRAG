# Counterfactual Fill Mechanism

This analysis isolates whether SetR-faithful under-selection failures are caused by missing support documents rather than only by using fewer reader-context slots.

- `rank_fill5`: SetR selected docs plus rank-order filler to five docs.
- `dbec_repair_fill5`: SetR selected docs plus DBEC-selected repair docs, then rank fill.
- `oracle_fill5`: SetR selected docs plus SetR-missing gold supports, then rank fill; diagnostic upper bound only.

## Method Means

| Dataset | Method | N | EM | F1 | Support recall | Support complete | Selected count | Empty answer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | SetR-faithful | 122 | 0.6230 | 0.6330 | 0.6270 | 0.0% | 2.57 |  |
| 2Wiki | DBEC-selective | 122 | 0.9180 | 0.9180 | 0.9242 | 70.5% | 5.00 |  |
| 2Wiki | rank_fill5 | 122 | 0.8361 | 0.8388 | 0.8770 | 54.9% | 5.00 | 0.0% |
| 2Wiki | oracle_fill5 | 122 | 0.9508 | 0.9508 | 0.9795 | 91.8% | 5.00 | 0.0% |
| 2Wiki | dbec_repair_fill5 | 122 | 0.9098 | 0.9098 | 0.9344 | 74.6% | 5.00 | 0.0% |
| MuSiQue | SetR-faithful | 106 | 0.1698 | 0.2274 | 0.4481 | 0.0% | 2.36 |  |
| MuSiQue | DBEC-selective | 106 | 0.2547 | 0.3093 | 0.6179 | 19.8% | 5.00 |  |
| MuSiQue | rank_fill5 | 106 | 0.2830 | 0.3392 | 0.6368 | 20.8% | 5.00 | 0.0% |
| MuSiQue | oracle_fill5 | 106 | 0.4717 | 0.5405 | 0.9175 | 73.6% | 5.00 | 0.0% |
| MuSiQue | dbec_repair_fill5 | 106 | 0.2264 | 0.2869 | 0.6124 | 15.1% | 5.00 | 0.0% |

## Paired F1 Deltas

| Dataset | Comparison | dF1 | 95% CI | p(delta>0) |
| --- | --- | ---: | ---: | ---: |
| 2Wiki | rank_fill5 - SetR-faithful | +0.2058 | [+0.1193, +0.2960] | 1.000 |
| 2Wiki | oracle_fill5 - SetR-faithful | +0.3179 | [+0.2304, +0.4080] | 1.000 |
| 2Wiki | dbec_repair_fill5 - SetR-faithful | +0.2769 | [+0.1940, +0.3625] | 1.000 |
| 2Wiki | DBEC-selective - SetR-faithful | +0.2851 | [+0.1994, +0.3698] | 1.000 |
| 2Wiki | oracle_fill5 - rank_fill5 | +0.1120 | [+0.0574, +0.1721] | 1.000 |
| 2Wiki | dbec_repair_fill5 - rank_fill5 | +0.0710 | [+0.0082, +0.1366] | 0.984 |
| 2Wiki | oracle_fill5 - DBEC-selective | +0.0328 | [-0.0082, +0.0819] | 0.898 |
| 2Wiki | dbec_repair_fill5 - DBEC-selective | -0.0082 | [-0.0574, +0.0410] | 0.310 |
| MuSiQue | rank_fill5 - SetR-faithful | +0.1117 | [+0.0446, +0.1851] | 0.999 |
| MuSiQue | oracle_fill5 - SetR-faithful | +0.3131 | [+0.2322, +0.3962] | 1.000 |
| MuSiQue | dbec_repair_fill5 - SetR-faithful | +0.0594 | [-0.0074, +0.1305] | 0.958 |
| MuSiQue | DBEC-selective - SetR-faithful | +0.0819 | [+0.0039, +0.1642] | 0.980 |
| MuSiQue | oracle_fill5 - rank_fill5 | +0.2013 | [+0.1260, +0.2819] | 1.000 |
| MuSiQue | dbec_repair_fill5 - rank_fill5 | -0.0523 | [-0.1175, +0.0094] | 0.049 |
| MuSiQue | oracle_fill5 - DBEC-selective | +0.2312 | [+0.1393, +0.3240] | 1.000 |
| MuSiQue | dbec_repair_fill5 - DBEC-selective | -0.0225 | [-0.0837, +0.0355] | 0.230 |

## Support-Repair Gap Accounting

| Dataset | Method | dF1 vs SetR | DBEC dF1 vs SetR | Share of DBEC gap |
| --- | --- | ---: | ---: | ---: |
| 2Wiki | rank_fill5 | +0.2058 | +0.2851 | 0.7220 |
| 2Wiki | dbec_repair_fill5 | +0.2769 | +0.2851 | 0.9712 |
| 2Wiki | oracle_fill5 | +0.3179 | +0.2851 | 1.1150 |
| MuSiQue | rank_fill5 | +0.1117 | +0.0819 | 1.3640 |
| MuSiQue | dbec_repair_fill5 | +0.0594 | +0.0819 | 0.7256 |
| MuSiQue | oracle_fill5 | +0.3131 | +0.0819 | 3.8221 |

## Interpretation Guide

- If `rank_fill5` is close to SetR but `dbec_repair_fill5` improves, the failure is not just fewer documents; it is which missing documents are inserted.
- If `oracle_fill5` is much higher than SetR, missing gold support is a real reader bottleneck on this slice.
- If `dbec_repair_fill5` recovers a large share of the DBEC-selective gain, the DBEC advantage is largely explained by support-chain repair.
- If `rank_fill5` also recovers most of the gain, the mechanism should be framed as budget under-fill rather than dependency-aware support repair.

## Files

- Per-query rows: `reports/counterfactual_fill_mechanism_20260507/counterfactual_rows.csv`
- Method means: `reports/counterfactual_fill_mechanism_20260507/method_summary.csv`
- Paired CIs: `reports/counterfactual_fill_mechanism_20260507/paired_ci.csv`
- Gap accounting: `reports/counterfactual_fill_mechanism_20260507/gap_decomposition.csv`
