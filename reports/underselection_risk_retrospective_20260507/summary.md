# Under-Selection Risk Retrospective Sanity

Date: 2026-05-07

This is an M0 retrospective check for a later preregistered predictive-validity experiment. It uses existing PropRAG full1000 DBEC-selective traces and SetR-faithful selected-only outputs; it makes no new LLM or reader calls.

Frozen risk split: **high-risk** if DBEC decomposition has `requirement_count >= 3`, otherwise **low-risk**. This signal is computed before SetR selection and before answer evaluation.

`under-select` is count-based: `SetR-faithful selected passage count < gold_doc_count`. It does not assert that selected passages are the correct supports.

## Raw Risk-Slice Table

| Dataset | Risk | N | Share | Demand count | Gold docs | SetR passages | SetR under-select | DBEC F1 | SetR F1 | dF1 | 95% CI | dR5_TITLE | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | low | 458 | 45.8% | 1.97 | 2.00 | 2.27 | 7.4% | 0.5215 | 0.4987 | +0.0227 | [-0.0056, +0.0514] | +0.0600 | [+0.0426, +0.0775] |
| 2Wiki | high | 542 | 54.2% | 3.45 | 2.87 | 2.84 | 22.7% | 0.8727 | 0.8232 | +0.0494 | [+0.0167, +0.0824] | +0.0544 | [+0.0346, +0.0733] |
| HotpotQA | low | 630 | 63.0% | 1.82 | 2.00 | 2.67 | 3.7% | 0.7301 | 0.7176 | +0.0126 | [-0.0112, +0.0366] | +0.0492 | [+0.0317, +0.0667] |
| HotpotQA | high | 370 | 37.0% | 3.20 | 2.00 | 2.85 | 2.4% | 0.7766 | 0.7877 | -0.0111 | [-0.0399, +0.0167] | +0.0189 | [+0.0014, +0.0378] |
| MuSiQue | low | 615 | 61.5% | 1.95 | 2.29 | 3.40 | 9.4% | 0.5156 | 0.5050 | +0.0106 | [-0.0228, +0.0434] | +0.0828 | [+0.0589, +0.1053] |
| MuSiQue | high | 385 | 38.5% | 3.37 | 3.22 | 3.98 | 18.7% | 0.3577 | 0.3534 | +0.0043 | [-0.0393, +0.0475] | +0.0848 | [+0.0584, +0.1113] |

## High-Low Interactions

Bootstrap CIs resample high-risk and low-risk rows independently. Positive values mean the high-risk slice has more under-selection or a larger DBEC-vs-SetR gap.

| Dataset | Metric | N high | N low | High-low delta | 95% CI | P(delta > 0) | Excludes 0 |
|---|---|---:|---:|---:|---:|---:|---|
| 2Wiki | SetR under-select rate: high - low | 542 | 458 | +0.1527 | [+0.1108, +0.1958] | 1.000 | yes |
| 2Wiki | DBEC-SetR dF1 interaction: high - low | 542 | 458 | +0.0267 | [-0.0166, +0.0706] | 0.887 | no |
| 2Wiki | DBEC-SetR dR5_TITLE interaction: high - low | 542 | 458 | -0.0056 | [-0.0320, +0.0202] | 0.341 | no |
| HotpotQA | SetR under-select rate: high - low | 370 | 630 | -0.0122 | [-0.0332, +0.0094] | 0.135 | no |
| HotpotQA | DBEC-SetR dF1 interaction: high - low | 370 | 630 | -0.0237 | [-0.0611, +0.0134] | 0.108 | no |
| HotpotQA | DBEC-SetR dR5_TITLE interaction: high - low | 370 | 630 | -0.0303 | [-0.0547, -0.0050] | 0.008 | yes |
| MuSiQue | SetR under-select rate: high - low | 385 | 615 | +0.0927 | [+0.0488, +0.1375] | 1.000 | yes |
| MuSiQue | DBEC-SetR dF1 interaction: high - low | 385 | 615 | -0.0063 | [-0.0627, +0.0481] | 0.405 | no |
| MuSiQue | DBEC-SetR dR5_TITLE interaction: high - low | 385 | 615 | +0.0021 | [-0.0330, +0.0370] | 0.541 | no |
| 2Wiki+MuSiQue | SetR under-select rate: high - low | 927 | 1073 | +0.1246 | [+0.0940, +0.1564] | 1.000 | yes |
| 2Wiki+MuSiQue | DBEC-SetR dF1 interaction: high - low | 927 | 1073 | +0.0149 | [-0.0196, +0.0500] | 0.802 | no |
| 2Wiki+MuSiQue | DBEC-SetR dR5_TITLE interaction: high - low | 927 | 1073 | -0.0060 | [-0.0277, +0.0165] | 0.298 | no |

## Gold-Controlled Diagnostic

This table checks whether `requirement_count >= 3` still separates risk after conditioning on the number of annotated gold support documents. These gold counts are not available at inference time; this is only a retrospective confound diagnostic.

| Dataset | Gold slice | Risk | N | Share in slice | Demand count | SetR passages | Under-select | dF1 | 95% CI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | gold_doc_count=2 | low | 458 | 59.9% | 1.97 | 2.27 | 7.4% | +0.0227 | [-0.0054, +0.0512] |
| 2Wiki | gold_doc_count=2 | high | 307 | 40.1% | 3.03 | 2.41 | 0.3% | -0.0227 | [-0.0599, +0.0144] |
| 2Wiki | gold_doc_count>=3 | high | 235 | 100.0% | 3.99 | 3.40 | 51.9% | +0.1437 | [+0.0879, +0.1995] |
| 2Wiki | gold_doc_count>=4 | high | 235 | 100.0% | 3.99 | 3.40 | 51.9% | +0.1437 | [+0.0875, +0.1991] |
| HotpotQA | gold_doc_count=2 | low | 630 | 63.0% | 1.82 | 2.67 | 3.7% | +0.0126 | [-0.0114, +0.0374] |
| HotpotQA | gold_doc_count=2 | high | 370 | 37.0% | 3.20 | 2.85 | 2.4% | -0.0111 | [-0.0396, +0.0178] |
| MuSiQue | gold_doc_count=2 | low | 466 | 90.0% | 1.94 | 3.18 | 4.9% | +0.0014 | [-0.0348, +0.0385] |
| MuSiQue | gold_doc_count=2 | high | 52 | 10.0% | 3.12 | 3.37 | 1.9% | -0.0317 | [-0.1538, +0.0908] |
| MuSiQue | gold_doc_count>=3 | low | 149 | 30.9% | 1.99 | 4.08 | 23.5% | +0.0395 | [-0.0295, +0.1082] |
| MuSiQue | gold_doc_count>=3 | high | 333 | 69.1% | 3.41 | 4.08 | 21.3% | +0.0099 | [-0.0389, +0.0563] |
| MuSiQue | gold_doc_count>=4 | low | 31 | 18.7% | 1.97 | 4.13 | 35.5% | +0.1950 | [+0.0158, +0.3763] |
| MuSiQue | gold_doc_count>=4 | high | 135 | 81.3% | 3.58 | 4.56 | 26.7% | -0.0215 | [-0.0905, +0.0471] |

| Dataset | Gold slice | N high | N low | Under-select high-low | dF1 high-low |
|---|---|---:|---:|---:|---:|
| 2Wiki | gold_doc_count=2 | 307 | 458 | -0.0710 | -0.0455 |
| HotpotQA | gold_doc_count=2 | 370 | 630 | -0.0122 | -0.0237 |
| MuSiQue | gold_doc_count=2 | 52 | 466 | -0.0301 | -0.0331 |
| MuSiQue | gold_doc_count>=3 | 333 | 149 | -0.0217 | -0.0296 |
| MuSiQue | gold_doc_count>=4 | 135 | 31 | -0.0882 | -0.2165 |

## Exploratory Feature Scout

The following rules are exploratory and were checked after the frozen M0 split. They should not be presented as preregistered evidence. Their purpose is to decide whether a better selection-independent risk rule is worth preregistering for a held-out suffix.

| Rule | Dataset | N high | Share high | Gold high/low | Under high-low | 95% CI | dF1 high | dF1 low | dF1 high-low | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| requirement_count>=3 | 2Wiki | 542 | 54.2% | 2.87/2.00 | +0.1527 | [+0.1102, +0.1962] | +0.0494 | +0.0227 | +0.0267 | [-0.0162, +0.0703] |
| requirement_count>=3 | HotpotQA | 370 | 37.0% | 2.00/2.00 | -0.0122 | [-0.0330, +0.0097] | -0.0111 | +0.0126 | -0.0237 | [-0.0614, +0.0135] |
| requirement_count>=3 | MuSiQue | 385 | 38.5% | 3.22/2.29 | +0.0927 | [+0.0482, +0.1385] | +0.0043 | +0.0106 | -0.0063 | [-0.0607, +0.0488] |
| requirement_count>=3 | 2Wiki+MuSiQue | 927 | 46.4% | 3.01/2.17 | +0.1246 | [+0.0937, +0.1556] | +0.0307 | +0.0158 | +0.0149 | [-0.0204, +0.0497] |
| requirement_count>=3_and_mean_binding_candidates>=2 | 2Wiki | 76 | 7.6% | 2.95/2.43 | +0.1434 | [+0.0417, +0.2506] | +0.0945 | +0.0325 | +0.0620 | [-0.0608, +0.1824] |
| requirement_count>=3_and_mean_binding_candidates>=2 | HotpotQA | 108 | 10.8% | 2.00/2.00 | -0.0047 | [-0.0336, +0.0312] | -0.0328 | +0.0082 | -0.0410 | [-0.1038, +0.0181] |
| requirement_count>=3_and_mean_binding_candidates>=2 | MuSiQue | 32 | 3.2% | 2.91/2.64 | +0.0917 | [-0.0426, +0.2479] | -0.0291 | +0.0094 | -0.0385 | [-0.2237, +0.1488] |
| requirement_count>=3_and_mean_binding_candidates>=2 | 2Wiki+MuSiQue | 108 | 5.4% | 2.94/2.54 | +0.1322 | [+0.0504, +0.2184] | +0.0579 | +0.0207 | +0.0372 | [-0.0616, +0.1377] |
| requirement_count>=3_and_max_binding_candidates>=2 | 2Wiki | 167 | 16.7% | 3.37/2.29 | +0.2356 | [+0.1590, +0.3111] | +0.0693 | +0.0308 | +0.0385 | [-0.0317, +0.1115] |
| requirement_count>=3_and_max_binding_candidates>=2 | HotpotQA | 151 | 15.1% | 2.00/2.00 | +0.0091 | [-0.0214, +0.0446] | -0.0285 | +0.0096 | -0.0381 | [-0.0864, +0.0098] |
| requirement_count>=3_and_max_binding_candidates>=2 | MuSiQue | 74 | 7.4% | 3.11/2.61 | +0.0639 | [-0.0258, +0.1612] | +0.0162 | +0.0075 | +0.0087 | [-0.1015, +0.1182] |
| requirement_count>=3_and_max_binding_candidates>=2 | 2Wiki+MuSiQue | 241 | 12.0% | 3.29/2.46 | +0.1812 | [+0.1216, +0.2406] | +0.0530 | +0.0185 | +0.0344 | [-0.0267, +0.0958] |
| requirement_count>=3_and_dependent_req_count>=2 | 2Wiki | 305 | 30.5% | 3.52/2.01 | +0.3449 | [+0.2884, +0.4019] | +0.1239 | -0.0008 | +0.1248 | [+0.0695, +0.1811] |
| requirement_count>=3_and_dependent_req_count>=2 | HotpotQA | 188 | 18.8% | 2.00/2.00 | +0.0195 | [-0.0111, +0.0543] | -0.0095 | +0.0069 | -0.0164 | [-0.0605, +0.0285] |
| requirement_count>=3_and_dependent_req_count>=2 | MuSiQue | 345 | 34.5% | 3.24/2.34 | +0.1024 | [+0.0545, +0.1498] | +0.0172 | +0.0034 | +0.0137 | [-0.0437, +0.0715] |
| requirement_count>=3_and_dependent_req_count>=2 | 2Wiki+MuSiQue | 650 | 32.5% | 3.37/2.17 | +0.2182 | [+0.1814, +0.2565] | +0.0673 | +0.0012 | +0.0660 | [+0.0261, +0.1058] |
| requirement_count>=4 | 2Wiki | 242 | 24.2% | 3.93/2.01 | +0.4525 | [+0.3871, +0.5156] | +0.1428 | +0.0035 | +0.1393 | [+0.0810, +0.1997] |
| requirement_count>=4 | HotpotQA | 74 | 7.4% | 2.00/2.00 | -0.0054 | [-0.0378, +0.0390] | -0.0223 | +0.0059 | -0.0282 | [-0.0969, +0.0401] |
| requirement_count>=4 | MuSiQue | 143 | 14.3% | 3.50/2.51 | +0.0931 | [+0.0243, +0.1642] | +0.0074 | +0.0083 | -0.0009 | [-0.0747, +0.0741] |
| requirement_count>=4 | 2Wiki+MuSiQue | 385 | 19.2% | 3.77/2.27 | +0.3080 | [+0.2562, +0.3583] | +0.0925 | +0.0060 | +0.0865 | [+0.0386, +0.1342] |

Gold-controlled spot check for exploratory rules:

| Rule | Dataset | Gold slice | N high | N low | Under high-low | dF1 high-low | dF1 high | dF1 low |
|---|---|---|---:|---:|---:|---:|---:|---:|
| requirement_count>=3_and_max_binding_candidates>=2 | 2Wiki | gold_doc_count=2 | 53 | 712 | -0.0492 | +0.0044 | +0.0086 | +0.0042 |
| requirement_count>=3_and_max_binding_candidates>=2 | 2Wiki | gold_doc_count>=3 | 114 | 121 | -0.0031 | -0.0899 | +0.0975 | +0.1873 |
| requirement_count>=3_and_max_binding_candidates>=2 | 2Wiki | gold_doc_count>=4 | 114 | 121 | -0.0031 | -0.0899 | +0.0975 | +0.1873 |
| requirement_count>=3_and_max_binding_candidates>=2 | HotpotQA | gold_doc_count=2 | 151 | 849 | +0.0091 | -0.0381 | -0.0285 | +0.0096 |
| requirement_count>=3_and_max_binding_candidates>=2 | MuSiQue | gold_doc_count=2 | 18 | 500 | -0.0480 | -0.1211 | -0.1188 | +0.0023 |
| requirement_count>=3_and_max_binding_candidates>=2 | MuSiQue | gold_doc_count>=3 | 56 | 426 | +0.0340 | +0.0459 | +0.0596 | +0.0137 |
| requirement_count>=3_and_max_binding_candidates>=2 | MuSiQue | gold_doc_count>=4 | 26 | 140 | +0.1203 | -0.1820 | -0.1346 | +0.0474 |
| requirement_count>=3_and_dependent_req_count>=2 | 2Wiki | gold_doc_count=2 | 73 | 692 | -0.0354 | +0.0560 | +0.0551 | -0.0009 |
| requirement_count>=3_and_dependent_req_count>=2 | 2Wiki | gold_doc_count>=3 | 232 | 3 | -0.1494 | +0.1456 | +0.1456 | +0.0000 |
| requirement_count>=3_and_dependent_req_count>=2 | 2Wiki | gold_doc_count>=4 | 232 | 3 | -0.1494 | +0.1456 | +0.1456 | +0.0000 |
| requirement_count>=3_and_dependent_req_count>=2 | HotpotQA | gold_doc_count=2 | 188 | 812 | +0.0195 | -0.0164 | -0.0095 | +0.0069 |
| requirement_count>=3_and_dependent_req_count>=2 | MuSiQue | gold_doc_count=2 | 46 | 472 | -0.0270 | +0.0026 | +0.0004 | -0.0022 |
| requirement_count>=3_and_dependent_req_count>=2 | MuSiQue | gold_doc_count>=3 | 299 | 183 | +0.0110 | +0.0019 | +0.0197 | +0.0179 |
| requirement_count>=3_and_dependent_req_count>=2 | MuSiQue | gold_doc_count>=4 | 128 | 38 | -0.0765 | -0.1932 | -0.0253 | +0.1678 |
| requirement_count>=4 | 2Wiki | gold_doc_count=2 | 9 | 756 | -0.0463 | +0.0829 | +0.0864 | +0.0035 |
| requirement_count>=4 | 2Wiki | gold_doc_count>=3 | 233 | 2 | +0.0193 | +0.1450 | +0.1450 | +0.0000 |
| requirement_count>=4 | 2Wiki | gold_doc_count>=4 | 233 | 2 | +0.0193 | +0.1450 | +0.1450 | +0.0000 |
| requirement_count>=4 | HotpotQA | gold_doc_count=2 | 74 | 926 | -0.0054 | -0.0282 | -0.0223 | +0.0059 |
| requirement_count>=4 | MuSiQue | gold_doc_count=2 | 6 | 512 | +0.1217 | -0.1104 | -0.1111 | -0.0007 |
| requirement_count>=4 | MuSiQue | gold_doc_count>=3 | 137 | 345 | -0.0115 | -0.0090 | +0.0126 | +0.0216 |
| requirement_count>=4 | MuSiQue | gold_doc_count>=4 | 78 | 88 | -0.0988 | -0.1063 | -0.0374 | +0.0688 |

## Key Findings

1. 2Wiki: high-risk share is 54.2%; SetR under-selection is 22.7% high vs 7.4% low, and DBEC-SetR dF1 is +0.0494 high vs +0.0227 low.
2. MuSiQue: high-risk share is 38.5%; SetR under-selection is 18.7% high vs 9.4% low, and DBEC-SetR dF1 is +0.0043 high vs +0.0106 low.
3. HotpotQA: high-risk share is 37.0%; SetR under-selection is 2.4% high vs 3.7% low, and DBEC-SetR dF1 is -0.0111 high vs +0.0126 low.
4. 2Wiki+MuSiQue combined interaction: SetR under-selection high-low delta is +0.1246 with CI [+0.0940, +0.1564]; DBEC-SetR dF1 high-low delta is +0.0149 with CI [-0.0196, +0.0500].
5. Gold-controlled check: 2Wiki gold_doc_count=2: under high-low -0.0710, dF1 high-low -0.0455; HotpotQA gold_doc_count=2: under high-low -0.0122, dF1 high-low -0.0237; MuSiQue gold_doc_count=2: under high-low -0.0301, dF1 high-low -0.0331; MuSiQue gold_doc_count>=3: under high-low -0.0217, dF1 high-low -0.0296. This suggests the unconditioned risk signal is largely a support-depth proxy, not yet a clean answer-gain predictor.
6. Exploratory rules remain mixed: `requirement_count>=3_and_max_binding_candidates>=2` gives combined dF1 high-low +0.0344, while `requirement_count>=3_and_dependent_req_count>=2` gives +0.0660; neither resolves the gold-controlled confound consistently across datasets.

## M0 Decision

M0 verdict: **partial retrospective pass**.
Gold-controlled diagnostics weaken the interpretation: the frozen risk split partly proxies latent support depth, and DBEC F1-gain concentration is not stable within fixed gold-doc-count slices.
Proceed to held-out preregistration only if we accept point-estimate under-selection prediction as enough for a low-cost follow-up. Do not frame this M0 result as strong predictive evidence for answer gain without a better risk feature or a gold-controlled success criterion.

## Suggested Next Experiments

1. Do not preregister `requirement_count >= 3` as a strong DBEC-gain predictor. At most, preregister it as an under-selection-risk predictor and use answer gain as a secondary endpoint.
2. Do not switch directly to the candidate-multiplicity conjunction either: the exploratory scout is mixed and gold-controlled checks still look dataset-dependent.
3. Keep HotpotQA as a retrospective negative-control diagnostic only unless a nontrivial high-risk slice exists in the held-out suffix.

## Output Files

- Query rows: `reports/underselection_risk_retrospective_20260507/risk_rows.csv`
- Slice summary: `reports/underselection_risk_retrospective_20260507/risk_slice_summary.csv`
- Interaction statistics: `reports/underselection_risk_retrospective_20260507/risk_interactions.csv`
- Gold-controlled diagnostics: `reports/underselection_risk_retrospective_20260507/gold_controlled_risk_summary.csv`
- Exploratory feature scout: `reports/underselection_risk_retrospective_20260507/exploratory_feature_scout.csv`
- Exploratory gold-controlled scout: `reports/underselection_risk_retrospective_20260507/exploratory_gold_controlled_scout.csv`
- JSON payload: `reports/underselection_risk_retrospective_20260507/summary.json`
