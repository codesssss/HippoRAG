# Reviewer Baseline Paired CI - 2026-05-06

Query-paired percentile bootstrap over answer EM/F1 and unified title-multiset support R@5, `10000` resamples. Delta is left method minus right method.

Support R@5 here is recomputed uniformly from each method's final top-5 titles against DAEC's reference gold_titles using a title-multiset match. These numbers may differ from pipeline-stored aggregate R@5 fields.

## Main F1 Rows

| Dataset | Comparison | Left F1 | Right F1 | dF1 | 95% CI | P(delta > 0) | Excludes 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | DAEC - Top5 | 0.7118 | 0.6457 | 0.0661 | [0.0442, 0.0878] | 1.000 | True |
| 2Wiki | DAEC-selective - Top5 | 0.7118 | 0.6457 | 0.0661 | [0.0439, 0.0879] | 1.000 | True |
| 2Wiki | DAEC-selective - DAEC | 0.7118 | 0.7118 | 0.0000 | [0.0000, 0.0000] | 0.000 | False |
| 2Wiki | DAEC - SetR-style k20 | 0.7118 | 0.6936 | 0.0182 | [-0.0015, 0.0383] | 0.964 | False |
| 2Wiki | DAEC-selective - SetR-style k20 | 0.7118 | 0.6936 | 0.0182 | [-0.0019, 0.0381] | 0.963 | False |
| 2Wiki | DAEC-selective - IRCoT-style local | 0.7118 | 0.6293 | 0.0825 | [0.0574, 0.1085] | 1.000 | True |
| 2Wiki | DAEC-selective - LLM-direct title | 0.7118 | 0.5697 | 0.1421 | [0.1155, 0.1681] | 1.000 | True |
| 2Wiki | DAEC-selective - LLM-direct snippet128 | 0.7118 | 0.6535 | 0.0583 | [0.0351, 0.0822] | 1.000 | True |
| HotpotQA | DAEC - Top5 | 0.7473 | 0.7227 | 0.0246 | [0.0110, 0.0383] | 1.000 | True |
| HotpotQA | DAEC-selective - Top5 | 0.7473 | 0.7227 | 0.0246 | [0.0114, 0.0377] | 1.000 | True |
| HotpotQA | DAEC-selective - DAEC | 0.7473 | 0.7473 | 0.0000 | [0.0000, 0.0000] | 0.000 | False |
| HotpotQA | DAEC - SetR-style k20 | 0.7473 | 0.7552 | -0.0079 | [-0.0225, 0.0064] | 0.140 | False |
| HotpotQA | DAEC-selective - SetR-style k20 | 0.7473 | 0.7552 | -0.0079 | [-0.0226, 0.0064] | 0.136 | False |
| HotpotQA | DAEC-selective - IRCoT-style local | 0.7473 | 0.7079 | 0.0394 | [0.0203, 0.0584] | 1.000 | True |
| HotpotQA | DAEC-selective - LLM-direct title | 0.7473 | 0.6478 | 0.0995 | [0.0764, 0.1216] | 1.000 | True |
| HotpotQA | DAEC-selective - LLM-direct snippet128 | 0.7473 | 0.6544 | 0.0929 | [0.0705, 0.1156] | 1.000 | True |
| MuSiQue | DAEC - Top5 | 0.4359 | 0.4266 | 0.0093 | [-0.0114, 0.0295] | 0.812 | False |
| MuSiQue | DAEC-selective - Top5 | 0.4548 | 0.4266 | 0.0282 | [0.0077, 0.0492] | 0.997 | True |
| MuSiQue | DAEC-selective - DAEC | 0.4548 | 0.4359 | 0.0189 | [0.0079, 0.0306] | 1.000 | True |
| MuSiQue | DAEC - SetR-style k20 | 0.4359 | 0.4761 | -0.0402 | [-0.0632, -0.0174] | 0.000 | True |
| MuSiQue | DAEC-selective - SetR-style k20 | 0.4548 | 0.4761 | -0.0212 | [-0.0445, 0.0011] | 0.032 | False |
| MuSiQue | DAEC-selective - IRCoT-style local | 0.4548 | 0.4254 | 0.0294 | [0.0038, 0.0555] | 0.986 | True |
| MuSiQue | DAEC-selective - LLM-direct title | 0.4548 | 0.3582 | 0.0966 | [0.0692, 0.1245] | 1.000 | True |
| MuSiQue | DAEC-selective - LLM-direct snippet128 | 0.4548 | 0.3850 | 0.0698 | [0.0428, 0.0971] | 1.000 | True |

## Main Unified Support R@5 Rows

| Dataset | Comparison | Left R@5 | Right R@5 | dR@5 | 95% CI | P(delta > 0) | Excludes 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | DAEC - Top5 | 0.9410 | 0.9028 | 0.0382 | [0.0265, 0.0498] | 1.000 | True |
| 2Wiki | DAEC-selective - Top5 | 0.9410 | 0.9028 | 0.0382 | [0.0265, 0.0498] | 1.000 | True |
| 2Wiki | DAEC-selective - DAEC | 0.9410 | 0.9410 | 0.0000 | [0.0000, 0.0000] | 0.000 | False |
| 2Wiki | DAEC - SetR-style k20 | 0.9410 | 0.9425 | -0.0015 | [-0.0120, 0.0088] | 0.388 | False |
| 2Wiki | DAEC-selective - SetR-style k20 | 0.9410 | 0.9425 | -0.0015 | [-0.0118, 0.0088] | 0.384 | False |
| 2Wiki | DAEC-selective - IRCoT-style local | 0.9410 | 0.8780 | 0.0630 | [0.0483, 0.0783] | 1.000 | True |
| 2Wiki | DAEC-selective - LLM-direct title | 0.9410 | 0.8167 | 0.1242 | [0.1077, 0.1405] | 1.000 | True |
| 2Wiki | DAEC-selective - LLM-direct snippet128 | 0.9410 | 0.8888 | 0.0522 | [0.0375, 0.0673] | 1.000 | True |
| HotpotQA | DAEC - Top5 | 0.9625 | 0.9520 | 0.0105 | [0.0040, 0.0175] | 0.998 | True |
| HotpotQA | DAEC-selective - Top5 | 0.9625 | 0.9520 | 0.0105 | [0.0040, 0.0175] | 0.999 | True |
| HotpotQA | DAEC-selective - DAEC | 0.9625 | 0.9625 | 0.0000 | [0.0000, 0.0000] | 0.000 | False |
| HotpotQA | DAEC - SetR-style k20 | 0.9625 | 0.9735 | -0.0110 | [-0.0200, -0.0020] | 0.006 | True |
| HotpotQA | DAEC-selective - SetR-style k20 | 0.9625 | 0.9735 | -0.0110 | [-0.0200, -0.0020] | 0.006 | True |
| HotpotQA | DAEC-selective - IRCoT-style local | 0.9625 | 0.9200 | 0.0425 | [0.0300, 0.0555] | 1.000 | True |
| HotpotQA | DAEC-selective - LLM-direct title | 0.9625 | 0.8280 | 0.1345 | [0.1185, 0.1510] | 1.000 | True |
| HotpotQA | DAEC-selective - LLM-direct snippet128 | 0.9625 | 0.8145 | 0.1480 | [0.1300, 0.1655] | 1.000 | True |
| MuSiQue | DAEC - Top5 | 0.7612 | 0.7378 | 0.0234 | [0.0103, 0.0363] | 1.000 | True |
| MuSiQue | DAEC-selective - Top5 | 0.7745 | 0.7378 | 0.0367 | [0.0250, 0.0486] | 1.000 | True |
| MuSiQue | DAEC-selective - DAEC | 0.7745 | 0.7612 | 0.0133 | [0.0053, 0.0216] | 0.999 | True |
| MuSiQue | DAEC - SetR-style k20 | 0.7612 | 0.7837 | -0.0225 | [-0.0375, -0.0075] | 0.001 | True |
| MuSiQue | DAEC-selective - SetR-style k20 | 0.7745 | 0.7837 | -0.0092 | [-0.0230, 0.0044] | 0.097 | False |
| MuSiQue | DAEC-selective - IRCoT-style local | 0.7745 | 0.6969 | 0.0776 | [0.0614, 0.0938] | 1.000 | True |
| MuSiQue | DAEC-selective - LLM-direct title | 0.7745 | 0.6119 | 0.1626 | [0.1438, 0.1809] | 1.000 | True |
| MuSiQue | DAEC-selective - LLM-direct snippet128 | 0.7745 | 0.6305 | 0.1440 | [0.1240, 0.1638] | 1.000 | True |

## All EM/F1/R@5 Rows

| Dataset | Comparison | Metric | Left | Right | Delta | 95% CI | Excludes 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | DAEC - Top5 | EM | 0.6420 | 0.5750 | 0.0670 | [0.0440, 0.0900] | True |
| 2Wiki | DAEC - Top5 | F1 | 0.7118 | 0.6457 | 0.0661 | [0.0442, 0.0878] | True |
| 2Wiki | DAEC - Top5 | R5_TITLE | 0.9410 | 0.9028 | 0.0382 | [0.0265, 0.0498] | True |
| 2Wiki | DAEC-selective - Top5 | EM | 0.6420 | 0.5750 | 0.0670 | [0.0440, 0.0900] | True |
| 2Wiki | DAEC-selective - Top5 | F1 | 0.7118 | 0.6457 | 0.0661 | [0.0439, 0.0879] | True |
| 2Wiki | DAEC-selective - Top5 | R5_TITLE | 0.9410 | 0.9028 | 0.0382 | [0.0265, 0.0498] | True |
| 2Wiki | DAEC-selective - DAEC | EM | 0.6420 | 0.6420 | 0.0000 | [0.0000, 0.0000] | False |
| 2Wiki | DAEC-selective - DAEC | F1 | 0.7118 | 0.7118 | 0.0000 | [0.0000, 0.0000] | False |
| 2Wiki | DAEC-selective - DAEC | R5_TITLE | 0.9410 | 0.9410 | 0.0000 | [0.0000, 0.0000] | False |
| 2Wiki | DAEC - SetR-style k20 | EM | 0.6420 | 0.6240 | 0.0180 | [-0.0040, 0.0390] | False |
| 2Wiki | DAEC - SetR-style k20 | F1 | 0.7118 | 0.6936 | 0.0182 | [-0.0015, 0.0383] | False |
| 2Wiki | DAEC - SetR-style k20 | R5_TITLE | 0.9410 | 0.9425 | -0.0015 | [-0.0120, 0.0088] | False |
| 2Wiki | DAEC-selective - SetR-style k20 | EM | 0.6420 | 0.6240 | 0.0180 | [-0.0030, 0.0400] | False |
| 2Wiki | DAEC-selective - SetR-style k20 | F1 | 0.7118 | 0.6936 | 0.0182 | [-0.0019, 0.0381] | False |
| 2Wiki | DAEC-selective - SetR-style k20 | R5_TITLE | 0.9410 | 0.9425 | -0.0015 | [-0.0118, 0.0088] | False |
| 2Wiki | DAEC - IRCoT-style local | EM | 0.6420 | 0.5640 | 0.0780 | [0.0520, 0.1040] | True |
| 2Wiki | DAEC - IRCoT-style local | F1 | 0.7118 | 0.6293 | 0.0825 | [0.0575, 0.1078] | True |
| 2Wiki | DAEC - IRCoT-style local | R5_TITLE | 0.9410 | 0.8780 | 0.0630 | [0.0483, 0.0780] | True |
| 2Wiki | DAEC-selective - IRCoT-style local | EM | 0.6420 | 0.5640 | 0.0780 | [0.0520, 0.1050] | True |
| 2Wiki | DAEC-selective - IRCoT-style local | F1 | 0.7118 | 0.6293 | 0.0825 | [0.0574, 0.1085] | True |
| 2Wiki | DAEC-selective - IRCoT-style local | R5_TITLE | 0.9410 | 0.8780 | 0.0630 | [0.0483, 0.0783] | True |
| 2Wiki | DAEC - LLM-direct title | EM | 0.6420 | 0.5190 | 0.1230 | [0.0960, 0.1500] | True |
| 2Wiki | DAEC - LLM-direct title | F1 | 0.7118 | 0.5697 | 0.1421 | [0.1161, 0.1688] | True |
| 2Wiki | DAEC - LLM-direct title | R5_TITLE | 0.9410 | 0.8167 | 0.1242 | [0.1075, 0.1410] | True |
| 2Wiki | DAEC - LLM-direct snippet128 | EM | 0.6420 | 0.5880 | 0.0540 | [0.0300, 0.0790] | True |
| 2Wiki | DAEC - LLM-direct snippet128 | F1 | 0.7118 | 0.6535 | 0.0583 | [0.0349, 0.0822] | True |
| 2Wiki | DAEC - LLM-direct snippet128 | R5_TITLE | 0.9410 | 0.8888 | 0.0522 | [0.0375, 0.0673] | True |
| 2Wiki | DAEC-selective - LLM-direct title | EM | 0.6420 | 0.5190 | 0.1230 | [0.0960, 0.1500] | True |
| 2Wiki | DAEC-selective - LLM-direct title | F1 | 0.7118 | 0.5697 | 0.1421 | [0.1155, 0.1681] | True |
| 2Wiki | DAEC-selective - LLM-direct title | R5_TITLE | 0.9410 | 0.8167 | 0.1242 | [0.1077, 0.1405] | True |
| 2Wiki | DAEC-selective - LLM-direct snippet128 | EM | 0.6420 | 0.5880 | 0.0540 | [0.0290, 0.0790] | True |
| 2Wiki | DAEC-selective - LLM-direct snippet128 | F1 | 0.7118 | 0.6535 | 0.0583 | [0.0351, 0.0822] | True |
| 2Wiki | DAEC-selective - LLM-direct snippet128 | R5_TITLE | 0.9410 | 0.8888 | 0.0522 | [0.0375, 0.0673] | True |
| HotpotQA | DAEC - Top5 | EM | 0.6200 | 0.5950 | 0.0250 | [0.0110, 0.0400] | True |
| HotpotQA | DAEC - Top5 | F1 | 0.7473 | 0.7227 | 0.0246 | [0.0110, 0.0383] | True |
| HotpotQA | DAEC - Top5 | R5_TITLE | 0.9625 | 0.9520 | 0.0105 | [0.0040, 0.0175] | True |
| HotpotQA | DAEC-selective - Top5 | EM | 0.6200 | 0.5950 | 0.0250 | [0.0100, 0.0390] | True |
| HotpotQA | DAEC-selective - Top5 | F1 | 0.7473 | 0.7227 | 0.0246 | [0.0114, 0.0377] | True |
| HotpotQA | DAEC-selective - Top5 | R5_TITLE | 0.9625 | 0.9520 | 0.0105 | [0.0040, 0.0175] | True |
| HotpotQA | DAEC-selective - DAEC | EM | 0.6200 | 0.6200 | 0.0000 | [0.0000, 0.0000] | False |
| HotpotQA | DAEC-selective - DAEC | F1 | 0.7473 | 0.7473 | 0.0000 | [0.0000, 0.0000] | False |
| HotpotQA | DAEC-selective - DAEC | R5_TITLE | 0.9625 | 0.9625 | 0.0000 | [0.0000, 0.0000] | False |
| HotpotQA | DAEC - SetR-style k20 | EM | 0.6200 | 0.6290 | -0.0090 | [-0.0250, 0.0070] | False |
| HotpotQA | DAEC - SetR-style k20 | F1 | 0.7473 | 0.7552 | -0.0079 | [-0.0225, 0.0064] | False |
| HotpotQA | DAEC - SetR-style k20 | R5_TITLE | 0.9625 | 0.9735 | -0.0110 | [-0.0200, -0.0020] | True |
| HotpotQA | DAEC-selective - SetR-style k20 | EM | 0.6200 | 0.6290 | -0.0090 | [-0.0250, 0.0070] | False |
| HotpotQA | DAEC-selective - SetR-style k20 | F1 | 0.7473 | 0.7552 | -0.0079 | [-0.0226, 0.0064] | False |
| HotpotQA | DAEC-selective - SetR-style k20 | R5_TITLE | 0.9625 | 0.9735 | -0.0110 | [-0.0200, -0.0020] | True |
| HotpotQA | DAEC - IRCoT-style local | EM | 0.6200 | 0.5900 | 0.0300 | [0.0100, 0.0500] | True |
| HotpotQA | DAEC - IRCoT-style local | F1 | 0.7473 | 0.7079 | 0.0394 | [0.0209, 0.0584] | True |
| HotpotQA | DAEC - IRCoT-style local | R5_TITLE | 0.9625 | 0.9200 | 0.0425 | [0.0300, 0.0550] | True |
| HotpotQA | DAEC-selective - IRCoT-style local | EM | 0.6200 | 0.5900 | 0.0300 | [0.0100, 0.0510] | True |
| HotpotQA | DAEC-selective - IRCoT-style local | F1 | 0.7473 | 0.7079 | 0.0394 | [0.0203, 0.0584] | True |
| HotpotQA | DAEC-selective - IRCoT-style local | R5_TITLE | 0.9625 | 0.9200 | 0.0425 | [0.0300, 0.0555] | True |
| HotpotQA | DAEC - LLM-direct title | EM | 0.6200 | 0.5370 | 0.0830 | [0.0600, 0.1070] | True |
| HotpotQA | DAEC - LLM-direct title | F1 | 0.7473 | 0.6478 | 0.0995 | [0.0772, 0.1219] | True |
| HotpotQA | DAEC - LLM-direct title | R5_TITLE | 0.9625 | 0.8280 | 0.1345 | [0.1185, 0.1510] | True |
| HotpotQA | DAEC - LLM-direct snippet128 | EM | 0.6200 | 0.5450 | 0.0750 | [0.0520, 0.0990] | True |
| HotpotQA | DAEC - LLM-direct snippet128 | F1 | 0.7473 | 0.6544 | 0.0929 | [0.0709, 0.1156] | True |
| HotpotQA | DAEC - LLM-direct snippet128 | R5_TITLE | 0.9625 | 0.8145 | 0.1480 | [0.1300, 0.1665] | True |
| HotpotQA | DAEC-selective - LLM-direct title | EM | 0.6200 | 0.5370 | 0.0830 | [0.0600, 0.1070] | True |
| HotpotQA | DAEC-selective - LLM-direct title | F1 | 0.7473 | 0.6478 | 0.0995 | [0.0764, 0.1216] | True |
| HotpotQA | DAEC-selective - LLM-direct title | R5_TITLE | 0.9625 | 0.8280 | 0.1345 | [0.1185, 0.1510] | True |
| HotpotQA | DAEC-selective - LLM-direct snippet128 | EM | 0.6200 | 0.5450 | 0.0750 | [0.0520, 0.0990] | True |
| HotpotQA | DAEC-selective - LLM-direct snippet128 | F1 | 0.7473 | 0.6544 | 0.0929 | [0.0705, 0.1156] | True |
| HotpotQA | DAEC-selective - LLM-direct snippet128 | R5_TITLE | 0.9625 | 0.8145 | 0.1480 | [0.1300, 0.1655] | True |
| MuSiQue | DAEC - Top5 | EM | 0.3370 | 0.3300 | 0.0070 | [-0.0140, 0.0280] | False |
| MuSiQue | DAEC - Top5 | F1 | 0.4359 | 0.4266 | 0.0093 | [-0.0114, 0.0295] | False |
| MuSiQue | DAEC - Top5 | R5_TITLE | 0.7612 | 0.7378 | 0.0234 | [0.0103, 0.0363] | True |
| MuSiQue | DAEC-selective - Top5 | EM | 0.3530 | 0.3300 | 0.0230 | [0.0020, 0.0440] | True |
| MuSiQue | DAEC-selective - Top5 | F1 | 0.4548 | 0.4266 | 0.0282 | [0.0077, 0.0492] | True |
| MuSiQue | DAEC-selective - Top5 | R5_TITLE | 0.7745 | 0.7378 | 0.0367 | [0.0250, 0.0486] | True |
| MuSiQue | DAEC-selective - DAEC | EM | 0.3530 | 0.3370 | 0.0160 | [0.0060, 0.0260] | True |
| MuSiQue | DAEC-selective - DAEC | F1 | 0.4548 | 0.4359 | 0.0189 | [0.0079, 0.0306] | True |
| MuSiQue | DAEC-selective - DAEC | R5_TITLE | 0.7745 | 0.7612 | 0.0133 | [0.0053, 0.0216] | True |
| MuSiQue | DAEC - SetR-style k20 | EM | 0.3370 | 0.3770 | -0.0400 | [-0.0640, -0.0170] | True |
| MuSiQue | DAEC - SetR-style k20 | F1 | 0.4359 | 0.4761 | -0.0402 | [-0.0632, -0.0174] | True |
| MuSiQue | DAEC - SetR-style k20 | R5_TITLE | 0.7612 | 0.7837 | -0.0225 | [-0.0375, -0.0075] | True |
| MuSiQue | DAEC-selective - SetR-style k20 | EM | 0.3530 | 0.3770 | -0.0240 | [-0.0470, -0.0010] | True |
| MuSiQue | DAEC-selective - SetR-style k20 | F1 | 0.4548 | 0.4761 | -0.0212 | [-0.0445, 0.0011] | False |
| MuSiQue | DAEC-selective - SetR-style k20 | R5_TITLE | 0.7745 | 0.7837 | -0.0092 | [-0.0230, 0.0044] | False |
| MuSiQue | DAEC - IRCoT-style local | EM | 0.3370 | 0.3270 | 0.0100 | [-0.0180, 0.0370] | False |
| MuSiQue | DAEC - IRCoT-style local | F1 | 0.4359 | 0.4254 | 0.0105 | [-0.0165, 0.0371] | False |
| MuSiQue | DAEC - IRCoT-style local | R5_TITLE | 0.7612 | 0.6969 | 0.0643 | [0.0467, 0.0813] | True |
| MuSiQue | DAEC-selective - IRCoT-style local | EM | 0.3530 | 0.3270 | 0.0260 | [-0.0010, 0.0520] | False |
| MuSiQue | DAEC-selective - IRCoT-style local | F1 | 0.4548 | 0.4254 | 0.0294 | [0.0038, 0.0555] | True |
| MuSiQue | DAEC-selective - IRCoT-style local | R5_TITLE | 0.7745 | 0.6969 | 0.0776 | [0.0614, 0.0938] | True |
| MuSiQue | DAEC - LLM-direct title | EM | 0.3370 | 0.2720 | 0.0650 | [0.0370, 0.0930] | True |
| MuSiQue | DAEC - LLM-direct title | F1 | 0.4359 | 0.3582 | 0.0777 | [0.0505, 0.1053] | True |
| MuSiQue | DAEC - LLM-direct title | R5_TITLE | 0.7612 | 0.6119 | 0.1492 | [0.1306, 0.1686] | True |
| MuSiQue | DAEC - LLM-direct snippet128 | EM | 0.3370 | 0.2940 | 0.0430 | [0.0150, 0.0700] | True |
| MuSiQue | DAEC - LLM-direct snippet128 | F1 | 0.4359 | 0.3850 | 0.0509 | [0.0238, 0.0788] | True |
| MuSiQue | DAEC - LLM-direct snippet128 | R5_TITLE | 0.7612 | 0.6305 | 0.1307 | [0.1100, 0.1510] | True |
| MuSiQue | DAEC-selective - LLM-direct title | EM | 0.3530 | 0.2720 | 0.0810 | [0.0530, 0.1100] | True |
| MuSiQue | DAEC-selective - LLM-direct title | F1 | 0.4548 | 0.3582 | 0.0966 | [0.0692, 0.1245] | True |
| MuSiQue | DAEC-selective - LLM-direct title | R5_TITLE | 0.7745 | 0.6119 | 0.1626 | [0.1438, 0.1809] | True |
| MuSiQue | DAEC-selective - LLM-direct snippet128 | EM | 0.3530 | 0.2940 | 0.0590 | [0.0300, 0.0870] | True |
| MuSiQue | DAEC-selective - LLM-direct snippet128 | F1 | 0.4548 | 0.3850 | 0.0698 | [0.0428, 0.0971] | True |
| MuSiQue | DAEC-selective - LLM-direct snippet128 | R5_TITLE | 0.7745 | 0.6305 | 0.1440 | [0.1240, 0.1638] | True |

## Interpretation

- DAEC/DAEC-selective significantly beat Top5 on 2Wiki and HotpotQA answer F1; on MuSiQue, base DAEC vs Top5 crosses zero but DAEC-selective is significant.
- DAEC-selective significantly beats IRCoT-style and both LLM-direct variants on answer F1 across all three datasets.
- DAEC-selective does not significantly beat SetR-style on answer F1 on any dataset. The SetR comparison must be framed as competitive/mixed, not as a clean win.
- Unified title support shows why answer F1 and evidence coverage should both be reported: on MuSiQue, DAEC-selective improves support over base DAEC and remains close to SetR-style by support even though SetR-style is stronger on answer F1.
