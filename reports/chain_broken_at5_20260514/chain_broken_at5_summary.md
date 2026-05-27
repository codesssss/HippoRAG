# Chain-Broken@5 Offline Summary

Definition: `chain_broken@5 = mean_q[gold_titles(q) not subset of top5_titles(q)]`.
This is computed from retrieval outputs only; no LLM is called.

## Across Method-Dataset Correlation

- Rows with QA metrics: 36
- corr(chain_broken@5, F1): -0.9837061781363838
- corr(chain_broken@5, EM): -0.9602488910134842
- corr(all_gold@5, F1): 0.9837061781363842

## Summary

| Method | Dataset | chain-broken@5 | all-gold@5 | R@5 title | EM | F1 |
|---|---:|---:|---:|---:|---:|---:|
| ETV4+PCEC_all32 | 2wikimultihopqa | 0.1050 | 0.8950 | 0.9623 | 0.6590 | 0.7468 |
| ETV4+PCEC_old8b_utility | 2wikimultihopqa | 0.0950 | 0.9050 | 0.9637 | 0.6610 | 0.7472 |
| ETv3+PCEC_all32 | 2wikimultihopqa | 0.1580 | 0.8420 | 0.9490 | 0.6520 | 0.7379 |
| ETv3+PCEC_no_fact_witness_old8b_utility | 2wikimultihopqa | 0.1970 | 0.8030 | 0.9277 | 0.6400 | 0.7232 |
| ETv3+PCEC_old8b_utility | 2wikimultihopqa | 0.1320 | 0.8680 | 0.9547 | 0.6590 | 0.7445 |
| HippoRAG_valid_qwen32b_top200 | 2wikimultihopqa | 0.4000 | 0.6000 | 0.8275 | 0.5670 | 0.6343 |
| NeocorRAG_k3_old | 2wikimultihopqa | 0.4980 | 0.5020 | 0.7482 | 0.4560 | 0.5127 |
| PCEC_position_slot1_old8b | 2wikimultihopqa | 0.1450 | 0.8550 | 0.9500 | 0.6150 | 0.6960 |
| PCEC_position_slot2_old8b | 2wikimultihopqa | 0.1450 | 0.8550 | 0.9500 | 0.6290 | 0.7055 |
| PCEC_position_slot3_old8b | 2wikimultihopqa | 0.1450 | 0.8550 | 0.9500 | 0.6350 | 0.7091 |
| PCEC_position_slot4_old8b | 2wikimultihopqa | 0.1450 | 0.8550 | 0.9500 | 0.6290 | 0.7062 |
| PropRAG_qwen32b_top200 | 2wikimultihopqa | 0.2200 | 0.7800 | 0.9090 | 0.6110 | 0.6909 |
| ETV4+PCEC_all32 | hotpotqa | 0.0670 | 0.9330 | 0.9650 | 0.6280 | 0.7557 |
| ETV4+PCEC_old8b_utility | hotpotqa | 0.0700 | 0.9300 | 0.9630 | 0.6230 | 0.7542 |
| ETv3+PCEC_all32 | hotpotqa | 0.0670 | 0.9330 | 0.9650 | 0.6270 | 0.7553 |
| ETv3+PCEC_no_fact_witness_old8b_utility | hotpotqa | 0.0710 | 0.9290 | 0.9625 | 0.6240 | 0.7504 |
| ETv3+PCEC_old8b_utility | hotpotqa | 0.0660 | 0.9340 | 0.9650 | 0.6270 | 0.7550 |
| HippoRAG_valid_qwen32b_top200 | hotpotqa | 0.1410 | 0.8590 | 0.9260 | 0.5930 | 0.7251 |
| NeocorRAG_k3_old | hotpotqa | 0.2020 | 0.7980 | 0.8905 | 0.5740 | 0.6946 |
| PCEC_position_slot1_old8b | hotpotqa | 0.0710 | 0.9290 | 0.9630 | 0.6320 | 0.7503 |
| PCEC_position_slot2_old8b | hotpotqa | 0.0710 | 0.9290 | 0.9630 | 0.6290 | 0.7494 |
| PCEC_position_slot3_old8b | hotpotqa | 0.0710 | 0.9290 | 0.9630 | 0.6330 | 0.7512 |
| PCEC_position_slot4_old8b | hotpotqa | 0.0710 | 0.9290 | 0.9630 | 0.6320 | 0.7503 |
| PropRAG_qwen32b_top200 | hotpotqa | 0.0930 | 0.9070 | 0.9510 | 0.6180 | 0.7510 |
| ETV4+PCEC_all32 | musique | 0.4900 | 0.5100 | 0.7646 | 0.3720 | 0.4891 |
| ETV4+PCEC_old8b_utility | musique | 0.4970 | 0.5030 | 0.7618 | 0.3590 | 0.4764 |
| ETv3+PCEC_all32 | musique | 0.4910 | 0.5090 | 0.7636 | 0.3720 | 0.4902 |
| ETv3+PCEC_no_fact_witness_old8b_utility | musique | 0.4960 | 0.5040 | 0.7612 | 0.3630 | 0.4782 |
| ETv3+PCEC_old8b_utility | musique | 0.5010 | 0.4990 | 0.7598 | 0.3610 | 0.4766 |
| HippoRAG_valid_qwen32b_top200 | musique | 0.5770 | 0.4230 | 0.7096 | 0.3490 | 0.4654 |
| NeocorRAG_k3_old | musique | 0.6370 | 0.3630 | 0.6427 | 0.2910 | 0.3878 |
| PCEC_position_slot1_old8b | musique | 0.5120 | 0.4880 | 0.7560 | 0.3440 | 0.4411 |
| PCEC_position_slot2_old8b | musique | 0.5120 | 0.4880 | 0.7560 | 0.3440 | 0.4413 |
| PCEC_position_slot3_old8b | musique | 0.5120 | 0.4880 | 0.7560 | 0.3390 | 0.4424 |
| PCEC_position_slot4_old8b | musique | 0.5120 | 0.4880 | 0.7560 | 0.3400 | 0.4405 |
| PropRAG_qwen32b_top200 | musique | 0.5240 | 0.4760 | 0.7424 | 0.3650 | 0.4760 |
