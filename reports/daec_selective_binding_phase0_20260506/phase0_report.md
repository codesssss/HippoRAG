# DAEC Selective Binding Phase-0

- Verdict: `go_phase1`
- Strong flip label: `|dF1| >= 0.5` or answer EM flip.
- Split: stable hash split, approximately 20% dev / 80% test.
- Primary router scores exclude dep-score margin; margin appears only in diagnostic rows.

## Overall Metrics

| Dataset | Split | DAEC F1 | Nobind F1 | dF1 DAEC-Nobind | DAEC EM | Nobind EM |
|---|---|---:|---:|---:|---:|---:|
| 2Wiki | all | 0.7118 | 0.6120 | 0.0998 | 0.6420 | 0.5480 |
| HotpotQA | all | 0.7473 | 0.7387 | 0.0086 | 0.6200 | 0.6160 |
| MuSiQue | all | 0.4359 | 0.4458 | -0.0099 | 0.3370 | 0.3430 |
| 2Wiki | dev | 0.7298 | 0.6027 | 0.1271 | 0.6569 | 0.5294 |
| HotpotQA | dev | 0.7424 | 0.7281 | 0.0143 | 0.6099 | 0.6099 |
| MuSiQue | dev | 0.4362 | 0.4326 | 0.0037 | 0.3165 | 0.3249 |
| 2Wiki | test | 0.7072 | 0.6144 | 0.0928 | 0.6382 | 0.5528 |
| HotpotQA | test | 0.7487 | 0.7418 | 0.0069 | 0.6229 | 0.6178 |
| MuSiQue | test | 0.4358 | 0.4499 | -0.0141 | 0.3434 | 0.3486 |

## Strong Flip Counts

| Dataset | Strong Cases | Abstain Helpful | Bind Helpful | Ignored/Noisy |
|---|---:|---:|---:|---:|
| 2Wiki | 159 | 23 | 136 | 841 |
| HotpotQA | 57 | 25 | 32 | 943 |
| MuSiQue | 130 | 69 | 61 | 870 |

## Top AUC Features

AUC predicts `abstain_helpful` over strong flip cases. AUC below 0.5 means high feature values favor binding.

| Dataset | Feature | AUC | Direction | Abstain Mean | Bind Mean | n+ / n- |
|---|---|---:|---|---:|---:|---:|
| ALL | `bind_conf_composite_primary` | 0.3426 | high=>bind | 0.6745 | 0.7870 | 117 / 229 |
| ALL | `unmatched_entity_count` | 0.6496 | high=>abstain | 8.7778 | 4.1135 | 117 / 229 |
| ALL | `bind_conf_composite_with_margin_diagnostic` | 0.3538 | high=>bind | 0.6583 | 0.7742 | 117 / 229 |
| ALL | `match_rate` | 0.3546 | high=>bind | 0.4001 | 0.5731 | 117 / 229 |
| ALL | `raw_entity_count` | 0.6434 | high=>abstain | 13.0000 | 6.8297 | 117 / 229 |
| ALL | `bind_conf_match_quality` | 0.3730 | high=>bind | 0.4567 | 0.6084 | 117 / 229 |
| ALL | `avg_candidate_title_occurrences` | 0.6244 | high=>abstain | 1.8009 | 1.1285 | 117 / 229 |
| ALL | `bind_conf_title_unique` | 0.3776 | high=>bind | 0.7947 | 0.9405 | 117 / 229 |
| 2Wiki | `candidate_count_max` | 0.5855 | high=>abstain | 2.1304 | 1.7721 | 23 / 136 |
| 2Wiki | `bind_conf_candidate_simplicity` | 0.4146 | high=>bind | 0.6449 | 0.7485 | 23 / 136 |
| 2Wiki | `low_ambiguity` | 0.4146 | high=>bind | 0.6449 | 0.7485 | 23 / 136 |
| 2Wiki | `binding_count` | 0.5825 | high=>abstain | 2.2609 | 2.4118 | 23 / 136 |
| 2Wiki | `binding_count_unpruned` | 0.5825 | high=>abstain | 2.2609 | 2.4118 | 23 / 136 |
| 2Wiki | `candidate_count_mean` | 0.5694 | high=>abstain | 1.6957 | 1.5968 | 23 / 136 |
| 2Wiki | `duplicate_entity_count` | 0.5684 | high=>abstain | 0.4783 | 0.2132 | 23 / 136 |
| 2Wiki | `multi_candidate_req_count` | 0.5684 | high=>abstain | 0.6087 | 0.4779 | 23 / 136 |
| HotpotQA | `bind_conf_composite_with_margin_diagnostic` | 0.3025 | high=>bind | 0.6368 | 0.7879 | 25 / 32 |
| HotpotQA | `bind_conf_composite_primary` | 0.3175 | high=>bind | 0.6705 | 0.7962 | 25 / 32 |
| HotpotQA | `bind_conf_candidate_simplicity` | 0.3350 | high=>bind | 0.6304 | 0.8370 | 25 / 32 |
| HotpotQA | `low_ambiguity` | 0.3350 | high=>bind | 0.6304 | 0.8370 | 25 / 32 |
| HotpotQA | `multi_candidate_rate` | 0.6581 | high=>abstain | 0.5267 | 0.2344 | 25 / 32 |
| HotpotQA | `bind_conf_single_candidate_rate` | 0.3475 | high=>bind | 0.4267 | 0.7188 | 25 / 32 |
| HotpotQA | `exactish_match_rate` | 0.3475 | high=>bind | 0.6327 | 0.8333 | 25 / 32 |
| HotpotQA | `single_candidate_rate` | 0.3475 | high=>bind | 0.4267 | 0.7188 | 25 / 32 |
| MuSiQue | `duplicate_entity_count` | 0.6467 | high=>abstain | 1.3043 | 0.4262 | 69 / 61 |
| MuSiQue | `avg_candidate_title_occurrences` | 0.6367 | high=>abstain | 2.3247 | 1.4085 | 69 / 61 |
| MuSiQue | `bind_conf_title_unique` | 0.3647 | high=>bind | 0.6757 | 0.8423 | 69 / 61 |
| MuSiQue | `title_unique_rate` | 0.3647 | high=>bind | 0.6757 | 0.8423 | 69 / 61 |
| MuSiQue | `title_nonunique_rate` | 0.6353 | high=>abstain | 0.3243 | 0.1577 | 69 / 61 |
| MuSiQue | `bind_conf_composite_primary` | 0.3767 | high=>bind | 0.6421 | 0.7343 | 69 / 61 |
| MuSiQue | `bind_conf_composite_with_margin_diagnostic` | 0.3837 | high=>bind | 0.6317 | 0.7293 | 69 / 61 |
| MuSiQue | `raw_entity_count` | 0.6025 | high=>abstain | 15.6377 | 10.3115 | 69 / 61 |

## Dev-Optimal Gate

Selected dev score: `bind_conf_match_quality` at threshold `0.18`.

| Dataset | F1 | dF1 vs DAEC | Null Rate |
|---|---:|---:|---:|
| 2Wiki | 0.7345 | 0.0047 | 0.3235 |
| HotpotQA | 0.7459 | 0.0035 | 0.2735 |
| MuSiQue | 0.4550 | 0.0188 | 0.4473 |

## Held-Out Test Result For Dev-Optimal Gate

Selected dev score: `bind_conf_match_quality` at threshold `0.18`.

| Dataset | F1 | dF1 vs DAEC | Null Rate |
|---|---:|---:|---:|
| 2Wiki | 0.7042 | -0.0030 | 0.2977 |
| HotpotQA | 0.7503 | 0.0016 | 0.2587 |
| MuSiQue | 0.4361 | 0.0003 | 0.4024 |

## Robust Exploratory Gate (Passes Dev And Test)

Selected dev score: `bind_conf_title_unique` at threshold `0.88`.

| Dataset | F1 | dF1 vs DAEC | Null Rate |
|---|---:|---:|---:|
| 2Wiki | 0.7090 | -0.0029 | 0.3120 |
| HotpotQA | 0.7493 | 0.0020 | 0.3720 |
| MuSiQue | 0.4553 | 0.0194 | 0.6380 |

## Title-Uniqueness Robustness

The frozen Phase-1 rule uses `bind_conf_title_unique >= 0.88`. The table below reports the full high-uniqueness band `[0.80, 0.95]`, so `0.88` is treated as a representative structural cutoff near `0.9`, not an isolated optimum.

| Split | Dataset | Threshold Band | dF1 vs DAEC Range | Null Rate Range |
|---|---|---:|---:|---:|
| all | 2Wiki | [0.80, 0.95] | [-0.0029, -0.0029] | [0.3120, 0.3120] |
| all | HotpotQA | [0.80, 0.95] | [0.0020, 0.0020] | [0.3660, 0.3720] |
| all | MuSiQue | [0.80, 0.95] | [0.0156, 0.0194] | [0.6050, 0.6390] |
| dev | 2Wiki | [0.80, 0.95] | [-0.0074, -0.0074] | [0.3284, 0.3284] |
| dev | HotpotQA | [0.80, 0.95] | [0.0000, 0.0000] | [0.3812, 0.3857] |
| dev | MuSiQue | [0.80, 0.95] | [0.0130, 0.0147] | [0.6118, 0.6371] |
| test | 2Wiki | [0.80, 0.95] | [-0.0017, -0.0017] | [0.3078, 0.3078] |
| test | HotpotQA | [0.80, 0.95] | [0.0026, 0.0026] | [0.3616, 0.3681] |
| test | MuSiQue | [0.80, 0.95] | [0.0159, 0.0214] | [0.6029, 0.6396] |

Representative `0.88` all-split point:

| Dataset | F1 | dF1 vs DAEC | Null Rate |
|---|---:|---:|---:|
| 2Wiki | 0.7090 | -0.0029 | 0.3120 |
| HotpotQA | 0.7493 | 0.0020 | 0.3720 |
| MuSiQue | 0.4553 | 0.0194 | 0.6380 |

Interpretation: the high-uniqueness band consistently preserves 2Wiki within about 0.003 F1 of DAEC while improving MuSiQue; the selected `0.88` threshold is frozen before Phase-1 fresh reader runs.

## Gate Interpretation

- A primary, selection-independent confidence score passes the gate on both dev and test.
- Treat the robust gate as exploratory evidence of separability, not as a tuned held-out result.
- If implemented, freeze the rule before any new reader run and validate on a fresh rerun or separate split.

## Artifacts

- `query_features_csv`: `reports/daec_selective_binding_phase0_20260506/query_features.csv`
- `feature_auc_csv`: `reports/daec_selective_binding_phase0_20260506/feature_auc.csv`
- `threshold_curve_csv`: `reports/daec_selective_binding_phase0_20260506/threshold_curve.csv`
- `title_unique_robustness_csv`: `reports/daec_selective_binding_phase0_20260506/title_unique_robustness.csv`
- `threshold_curve_svg`: `reports/daec_selective_binding_phase0_20260506/threshold_curve.svg`
- `report_json`: `reports/daec_selective_binding_phase0_20260506/phase0_report.json`
- `report_md`: `reports/daec_selective_binding_phase0_20260506/phase0_report.md`
