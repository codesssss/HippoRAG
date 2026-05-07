# Support-Repair Mechanism Audit

Date: 2026-05-07

This diagnostic uses existing PropRAG full1000 DBEC-selective and SetR-faithful selected-only outputs. It makes no new LLM or reader calls. Gold support titles are used only to audit mechanism behavior.

Core question: when SetR-faithful omits annotated support titles, does DBEC recover those supports, and is the answer gain concentrated in those repair transitions?

## Dataset / Slice Summary

| Dataset | Slice | N | Count under-select | SetR complete | DBEC complete | SetR R | DBEC R | dR | DBEC recovers any | DBEC F1 | SetR F1 | dF1 | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | all | 1000 | 15.7% | 73.4% | 86.3% | 0.884 | 0.941 | +0.057 | 21.3% | 0.7118 | 0.6746 | +0.0372 | [+0.0154, +0.0595] |
| 2Wiki | gold_doc_count=2 | 765 | 4.6% | 83.8% | 89.9% | 0.915 | 0.944 | +0.029 | 10.8% | 0.6494 | 0.6450 | +0.0045 | [-0.0185, +0.0266] |
| 2Wiki | gold_doc_count>=4 | 235 | 51.9% | 39.6% | 74.5% | 0.783 | 0.930 | +0.147 | 55.3% | 0.9149 | 0.7712 | +0.1437 | [+0.0884, +0.1995] |
| 2Wiki | SetR count-underselected | 157 | 100.0% | 0.0% | 68.8% | 0.599 | 0.900 | +0.301 | 86.6% | 0.7946 | 0.5344 | +0.2603 | [+0.1861, +0.3355] |
| 2Wiki | gold_doc_count>=4 and SetR count-underselected | 122 | 100.0% | 0.0% | 70.5% | 0.627 | 0.924 | +0.297 | 91.8% | 0.9180 | 0.6330 | +0.2851 | [+0.2004, +0.3716] |
| HotpotQA | all | 1000 | 3.2% | 85.7% | 93.0% | 0.926 | 0.963 | +0.036 | 12.5% | 0.7473 | 0.7435 | +0.0038 | [-0.0149, +0.0227] |
| HotpotQA | SetR count-underselected | 32 | 100.0% | 0.0% | 81.2% | 0.484 | 0.906 | +0.422 | 84.4% | 0.6270 | 0.5179 | +0.1091 | [-0.0226, +0.2500] |
| MuSiQue | all | 1000 | 13.0% | 40.7% | 52.7% | 0.698 | 0.774 | +0.077 | 34.8% | 0.4548 | 0.4467 | +0.0082 | [-0.0179, +0.0344] |
| MuSiQue | gold_doc_count=2 | 518 | 4.6% | 63.5% | 77.2% | 0.802 | 0.877 | +0.075 | 25.5% | 0.5525 | 0.5544 | -0.0020 | [-0.0392, +0.0339] |
| MuSiQue | gold_doc_count>=3 | 482 | 22.0% | 16.2% | 26.3% | 0.586 | 0.664 | +0.078 | 44.8% | 0.3499 | 0.3308 | +0.0190 | [-0.0206, +0.0586] |
| MuSiQue | SetR count-underselected | 130 | 100.0% | 0.0% | 26.9% | 0.454 | 0.650 | +0.196 | 57.7% | 0.3174 | 0.2354 | +0.0820 | [+0.0132, +0.1548] |
| MuSiQue | gold_doc_count>=3 and SetR count-underselected | 106 | 100.0% | 0.0% | 19.8% | 0.448 | 0.618 | +0.170 | 56.6% | 0.3093 | 0.2274 | +0.0819 | [+0.0041, +0.1635] |

## Support Completeness Transitions

| Dataset | Slice | Transition | N | SetR R | DBEC R | dR | DBEC F1 | SetR F1 | dF1 | 95% CI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | all | SetR complete -> DBEC complete | 685 | 1.000 | 1.000 | +0.000 | 0.7409 | 0.7577 | -0.0167 | [-0.0343, +0.0008] |
| 2Wiki | all | SetR complete -> DBEC incomplete | 49 | 1.000 | 0.515 | -0.485 | 0.4683 | 0.8049 | -0.3366 | [-0.4876, -0.1868] |
| 2Wiki | all | SetR incomplete -> DBEC complete | 178 | 0.586 | 1.000 | +0.414 | 0.8007 | 0.4868 | +0.3139 | [+0.2494, +0.3806] |
| 2Wiki | all | SetR incomplete -> DBEC incomplete | 88 | 0.520 | 0.599 | +0.080 | 0.4412 | 0.3357 | +0.1055 | [+0.0185, +0.1951] |
| 2Wiki | gold_doc_count>=4 | SetR complete -> DBEC complete | 73 | 1.000 | 1.000 | +0.000 | 0.9315 | 0.9315 | +0.0000 | [-0.0548, +0.0548] |
| 2Wiki | gold_doc_count>=4 | SetR complete -> DBEC incomplete | 20 | 1.000 | 0.688 | -0.312 | 0.8000 | 0.9500 | -0.1500 | [-0.3000, +0.0000] |
| 2Wiki | gold_doc_count>=4 | SetR incomplete -> DBEC complete | 102 | 0.659 | 1.000 | +0.341 | 0.9706 | 0.6993 | +0.2712 | [+0.1895, +0.3595] |
| 2Wiki | gold_doc_count>=4 | SetR incomplete -> DBEC incomplete | 40 | 0.594 | 0.744 | +0.150 | 0.8000 | 0.5722 | +0.2278 | [+0.0528, +0.4028] |
| 2Wiki | SetR count-underselected | SetR incomplete -> DBEC complete | 108 | 0.616 | 1.000 | +0.384 | 0.8774 | 0.5904 | +0.2870 | [+0.2037, +0.3735] |
| 2Wiki | SetR count-underselected | SetR incomplete -> DBEC incomplete | 49 | 0.561 | 0.679 | +0.117 | 0.6123 | 0.4110 | +0.2013 | [+0.0635, +0.3441] |
| 2Wiki | gold_doc_count>=4 and SetR count-underselected | SetR incomplete -> DBEC complete | 86 | 0.645 | 1.000 | +0.355 | 0.9651 | 0.6783 | +0.2868 | [+0.1938, +0.3837] |
| 2Wiki | gold_doc_count>=4 and SetR count-underselected | SetR incomplete -> DBEC incomplete | 36 | 0.583 | 0.743 | +0.160 | 0.8056 | 0.5247 | +0.2809 | [+0.0896, +0.4722] |
| HotpotQA | all | SetR complete -> DBEC complete | 813 | 1.000 | 1.000 | +0.000 | 0.7860 | 0.7885 | -0.0025 | [-0.0192, +0.0143] |
| HotpotQA | all | SetR complete -> DBEC incomplete | 44 | 1.000 | 0.477 | -0.523 | 0.3581 | 0.7456 | -0.3875 | [-0.5220, -0.2580] |
| HotpotQA | all | SetR incomplete -> DBEC complete | 117 | 0.496 | 1.000 | +0.504 | 0.7304 | 0.5254 | +0.2050 | [+0.1309, +0.2805] |
| HotpotQA | all | SetR incomplete -> DBEC incomplete | 26 | 0.442 | 0.442 | +0.000 | 0.2718 | 0.3148 | -0.0430 | [-0.1795, +0.0897] |
| HotpotQA | SetR count-underselected | SetR incomplete -> DBEC complete | 26 | 0.481 | 1.000 | +0.519 | 0.6922 | 0.5662 | +0.1260 | [+0.0106, +0.2674] |
| HotpotQA | SetR count-underselected | SetR incomplete -> DBEC incomplete | 6 | 0.500 | 0.500 | +0.000 | 0.3444 | 0.3085 | +0.0359 | [-0.4641, +0.5000] |
| MuSiQue | all | SetR complete -> DBEC complete | 337 | 1.000 | 1.000 | +0.000 | 0.7063 | 0.7282 | -0.0219 | [-0.0548, +0.0108] |
| MuSiQue | all | SetR complete -> DBEC incomplete | 70 | 1.000 | 0.535 | -0.465 | 0.2609 | 0.6890 | -0.4281 | [-0.5452, -0.3146] |
| MuSiQue | all | SetR incomplete -> DBEC complete | 190 | 0.500 | 1.000 | +0.500 | 0.5245 | 0.3067 | +0.2178 | [+0.1523, +0.2855] |
| MuSiQue | all | SetR incomplete -> DBEC incomplete | 403 | 0.486 | 0.521 | +0.036 | 0.2453 | 0.2351 | +0.0102 | [-0.0302, +0.0497] |
| MuSiQue | gold_doc_count>=3 | SetR complete -> DBEC complete | 51 | 1.000 | 1.000 | +0.000 | 0.7380 | 0.7250 | +0.0131 | [-0.0693, +0.1007] |
| MuSiQue | gold_doc_count>=3 | SetR complete -> DBEC incomplete | 27 | 1.000 | 0.664 | -0.336 | 0.3519 | 0.6975 | -0.3457 | [-0.5432, -0.1481] |
| MuSiQue | gold_doc_count>=3 | SetR incomplete -> DBEC complete | 76 | 0.534 | 1.000 | +0.466 | 0.4406 | 0.2709 | +0.1697 | [+0.0761, +0.2662] |
| MuSiQue | gold_doc_count>=3 | SetR incomplete -> DBEC incomplete | 328 | 0.499 | 0.534 | +0.035 | 0.2683 | 0.2532 | +0.0151 | [-0.0317, +0.0629] |
| MuSiQue | SetR count-underselected | SetR incomplete -> DBEC complete | 35 | 0.495 | 1.000 | +0.505 | 0.5046 | 0.3002 | +0.2044 | [+0.0493, +0.3616] |
| MuSiQue | SetR count-underselected | SetR incomplete -> DBEC incomplete | 95 | 0.439 | 0.521 | +0.082 | 0.2484 | 0.2116 | +0.0368 | [-0.0389, +0.1140] |
| MuSiQue | gold_doc_count>=3 and SetR count-underselected | SetR incomplete -> DBEC complete | 21 | 0.492 | 1.000 | +0.508 | 0.5757 | 0.2622 | +0.3135 | [+0.1429, +0.5238] |
| MuSiQue | gold_doc_count>=3 and SetR count-underselected | SetR incomplete -> DBEC incomplete | 85 | 0.437 | 0.524 | +0.086 | 0.2435 | 0.2188 | +0.0247 | [-0.0573, +0.1075] |

## Recovery Among SetR Support-Incomplete Cases

| Dataset | Slice | Recovery label | N | Mean SetR missing | Mean DBEC recovered | DBEC complete | DBEC F1 | SetR F1 | dF1 | 95% CI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | all | dbec_recovers_none | 53 | 1.04 | 0.00 | 0.0% | 0.2860 | 0.2402 | +0.0458 | [-0.0249, +0.1206] |
| 2Wiki | all | dbec_recovers_some_missing | 24 | 2.00 | 1.00 | 0.0% | 0.8194 | 0.4444 | +0.3750 | [+0.1667, +0.5833] |
| 2Wiki | all | dbec_recovers_all_missing | 189 | 1.22 | 1.22 | 94.2% | 0.7752 | 0.4909 | +0.2843 | [+0.2179, +0.3530] |
| 2Wiki | gold_doc_count>=4 | dbec_recovers_none | 12 | 1.08 | 0.00 | 0.0% | 0.7501 | 0.5833 | +0.1667 | [+0.0000, +0.4167] |
| 2Wiki | gold_doc_count>=4 | dbec_recovers_some_missing | 21 | 2.00 | 1.00 | 0.0% | 0.9048 | 0.4762 | +0.4286 | [+0.2381, +0.6190] |
| 2Wiki | gold_doc_count>=4 | dbec_recovers_all_missing | 109 | 1.37 | 1.37 | 93.6% | 0.9450 | 0.7085 | +0.2365 | [+0.1478, +0.3272] |
| 2Wiki | SetR count-underselected | dbec_recovers_none | 21 | 1.05 | 0.00 | 0.0% | 0.3810 | 0.2857 | +0.0953 | [+0.0000, +0.2381] |
| 2Wiki | SetR count-underselected | dbec_recovers_some_missing | 20 | 2.00 | 1.00 | 0.0% | 0.9000 | 0.4500 | +0.4500 | [+0.2500, +0.6500] |
| 2Wiki | SetR count-underselected | dbec_recovers_all_missing | 116 | 1.34 | 1.34 | 93.1% | 0.8514 | 0.5939 | +0.2574 | [+0.1731, +0.3436] |
| HotpotQA | all | dbec_recovers_none | 18 | 1.06 | 0.00 | 0.0% | 0.3370 | 0.2695 | +0.0675 | [-0.0131, +0.1972] |
| HotpotQA | all | dbec_recovers_some_missing | 2 | 2.00 | 1.00 | 0.0% | 0.0000 | 0.1667 | -0.1667 | [-0.3333, +0.0000] |
| HotpotQA | all | dbec_recovers_all_missing | 123 | 1.00 | 1.00 | 95.1% | 0.7029 | 0.5242 | +0.1787 | [+0.1042, +0.2566] |
| HotpotQA | SetR count-underselected | dbec_recovers_none | 5 | 1.00 | 0.00 | 0.0% | 0.4133 | 0.1702 | +0.2431 | [-0.0471, +0.6431] |
| HotpotQA | SetR count-underselected | dbec_recovers_all_missing | 27 | 1.00 | 1.00 | 96.3% | 0.6665 | 0.5822 | +0.0843 | [-0.0564, +0.2325] |
| MuSiQue | all | dbec_recovers_none | 245 | 1.42 | 0.00 | 0.0% | 0.2527 | 0.2669 | -0.0143 | [-0.0582, +0.0293] |
| MuSiQue | all | dbec_recovers_some_missing | 115 | 2.27 | 1.10 | 0.0% | 0.2280 | 0.1211 | +0.1070 | [+0.0365, +0.1799] |
| MuSiQue | all | dbec_recovers_all_missing | 233 | 1.20 | 1.20 | 81.5% | 0.4738 | 0.3163 | +0.1575 | [+0.0898, +0.2236] |
| MuSiQue | gold_doc_count>=3 | dbec_recovers_none | 188 | 1.55 | 0.00 | 0.0% | 0.2768 | 0.3110 | -0.0341 | [-0.0893, +0.0200] |
| MuSiQue | gold_doc_count>=3 | dbec_recovers_some_missing | 104 | 2.30 | 1.12 | 0.0% | 0.2425 | 0.1291 | +0.1135 | [+0.0327, +0.1970] |
| MuSiQue | gold_doc_count>=3 | dbec_recovers_all_missing | 112 | 1.38 | 1.38 | 67.9% | 0.3948 | 0.2836 | +0.1112 | [+0.0173, +0.2067] |
| MuSiQue | SetR count-underselected | dbec_recovers_none | 55 | 1.71 | 0.00 | 0.0% | 0.2655 | 0.2533 | +0.0121 | [-0.0818, +0.1073] |
| MuSiQue | SetR count-underselected | dbec_recovers_some_missing | 35 | 2.31 | 1.09 | 0.0% | 0.2571 | 0.1476 | +0.1095 | [-0.0286, +0.2524] |
| MuSiQue | SetR count-underselected | dbec_recovers_all_missing | 40 | 1.35 | 1.35 | 87.5% | 0.4415 | 0.2877 | +0.1539 | [+0.0039, +0.3056] |

## Count-Underselected Repair Categories

| Dataset | Recovery label | N | Mean SetR missing | Mean DBEC recovered | DBEC complete | DBEC F1 | SetR F1 | dF1 | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | dbec_recovers_none | 21 | 1.05 | 0.00 | 0.0% | 0.3810 | 0.2857 | +0.0953 | [+0.0000, +0.2381] |
| 2Wiki | dbec_recovers_some_missing | 20 | 2.00 | 1.00 | 0.0% | 0.9000 | 0.4500 | +0.4500 | [+0.2500, +0.6500] |
| 2Wiki | dbec_recovers_all_missing | 116 | 1.34 | 1.34 | 93.1% | 0.8514 | 0.5939 | +0.2574 | [+0.1705, +0.3436] |
| HotpotQA | dbec_recovers_none | 5 | 1.00 | 0.00 | 0.0% | 0.4133 | 0.1702 | +0.2431 | [-0.0471, +0.6431] |
| HotpotQA | dbec_recovers_all_missing | 27 | 1.00 | 1.00 | 96.3% | 0.6665 | 0.5822 | +0.0843 | [-0.0564, +0.2381] |
| MuSiQue | dbec_recovers_none | 55 | 1.71 | 0.00 | 0.0% | 0.2655 | 0.2533 | +0.0121 | [-0.0788, +0.1048] |
| MuSiQue | dbec_recovers_some_missing | 35 | 2.31 | 1.09 | 0.0% | 0.2571 | 0.1476 | +0.1095 | [-0.0143, +0.2524] |
| MuSiQue | dbec_recovers_all_missing | 40 | 1.35 | 1.35 | 87.5% | 0.4415 | 0.2877 | +0.1539 | [+0.0041, +0.3039] |

## Key Findings

2Wiki gold_doc_count>=4: SetR support-complete 39.6%, DBEC support-complete 74.5%, dR +0.147, dF1 +0.1437 with CI [+0.0884, +0.1995].
MuSiQue gold_doc_count>=3: SetR support-complete 16.2%, DBEC support-complete 26.3%, dR +0.078, dF1 +0.0190 with CI [-0.0206, +0.0586].
HotpotQA all: SetR support-complete 85.7%, DBEC support-complete 93.0%, dR +0.036, dF1 +0.0038 with CI [-0.0149, +0.0227].
2Wiki gold_doc_count>=4: the support-repair transition `SetR incomplete -> DBEC complete` has N=102 and dF1 +0.2712 with CI [+0.1895, +0.3595].
MuSiQue gold_doc_count>=3: the support-repair transition `SetR incomplete -> DBEC complete` has N=76 and dF1 +0.1697 with CI [+0.0761, +0.2662].
2Wiki count-underselected: DBEC recovers at least one SetR-missing support in N=136 support-incomplete cases, with weighted dF1 +0.2857.
MuSiQue count-underselected: DBEC recovers at least one SetR-missing support in N=75 support-incomplete cases, with weighted dF1 +0.1332.
This is stronger than the earlier under-selection slice because it verifies the concrete object being repaired: annotated support coverage. It is still not a causal intervention; use it to motivate counterfactual fill, not to claim proof.

## Paper-Facing Interpretation

DBEC's gains are better described as support-chain repair than as universal reranking superiority: the strongest gains occur when SetR-faithful omits annotated supports and DBEC recovers them. However, this audit is still diagnostic rather than causal because it does not intervene on the reader context. The next stronger test is a counterfactual fill experiment: add oracle missing supports or DBEC repair documents to SetR's context and compare against rank-fill controls.

## Example Repaired Cases

| Dataset | Query | dF1 | Transition | Recovered titles |
|---|---:|---:|---|---|
| 2Wiki | 616 | +1.0000 | SetR incomplete -> DBEC complete | The Man in the Funny Suit, Ralph Nelson, The Devil and Miss Jones |
| 2Wiki | 199 | +1.0000 | SetR incomplete -> DBEC complete | Harold D. Schuster, William Friedkin |
| 2Wiki | 217 | +1.0000 | SetR incomplete -> DBEC complete | Ann Hui, Karl Maka |
| 2Wiki | 564 | +1.0000 | SetR incomplete -> DBEC complete | Laughing at Death, Five Red Tulips |
| 2Wiki | 581 | +1.0000 | SetR incomplete -> DBEC incomplete | Chad Hanna, The Avenging Shadow |
| 2Wiki | 708 | +1.0000 | SetR incomplete -> DBEC complete | Rob Margolies, Robert G. Vignola |
| 2Wiki | 781 | +1.0000 | SetR incomplete -> DBEC complete | Jesse Hibbs, Michał Waszyński |
| 2Wiki | 792 | +1.0000 | SetR incomplete -> DBEC complete | K. Raghavendra Rao, The Laughing Woman |
| 2Wiki | 803 | +1.0000 | SetR incomplete -> DBEC complete | Claude Autant-Lara, Howard Bretherton |
| 2Wiki | 838 | +1.0000 | SetR incomplete -> DBEC complete | Manuel Romero, Hugo Fregonese |
| 2Wiki | 865 | +1.0000 | SetR incomplete -> DBEC complete | Scott Shaw, Christine Pascal |
| 2Wiki | 901 | +1.0000 | SetR incomplete -> DBEC complete | Charles Crichton, Howard Bretherton |

## Output Files

- Query rows: `reports/support_repair_mechanism_20260507/support_repair_rows.csv`
- Slice summary: `reports/support_repair_mechanism_20260507/slice_summary.csv`
- Transition summary: `reports/support_repair_mechanism_20260507/transition_summary.csv`
- Recovery summary: `reports/support_repair_mechanism_20260507/recovery_summary.csv`
- Count-underselected recovery summary: `reports/support_repair_mechanism_20260507/count_underselected_recovery_summary.csv`
- Case examples: `reports/support_repair_mechanism_20260507/case_examples.csv`
- JSON payload: `reports/support_repair_mechanism_20260507/summary.json`
