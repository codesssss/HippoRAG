# EvidenceLink FCRG Mechanism Diagnostic

FCRG measures whether a method promotes gold supporting documents whose relevance is licensed by a source-grounded OpenIE fact transition from another gold supporting document. Dense query-only is the reference ranker, so its FCRG is expected to be 0.

## Aggregate

| method | FCRG | avg_delta | rho_f | method_at5 | promoted | demoted | fact_docs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense query-only | 0.0000 | 0.0000 | 40.46 | 54.17 | 0.00 | 0.00 | 2880 |
| EvidenceLink pool | 0.1435 | 0.0753 | 40.46 | 78.58 | 47.74 | 27.15 | 2880 |
| EvidenceLink delivered | 0.1567 | 0.0822 | 40.46 | 83.40 | 49.13 | 26.01 | 2880 |
| w/o evidence-linked transitions pool | 0.1338 | 0.0702 | 40.46 | 73.19 | 45.28 | 24.62 | 2880 |
| w/o evidence-linked transitions delivered | 0.1509 | 0.0792 | 40.46 | 79.62 | 46.49 | 23.82 | 2880 |
| Dense-doc KNN transitions pool | 0.0158 | 0.0083 | 40.46 | 54.17 | 21.04 | 4.48 | 2880 |
| Dense-doc KNN transitions delivered | 0.0625 | 0.0328 | 40.46 | 71.11 | 28.75 | 8.12 | 2880 |
| Degree-matched shuffled transitions pool | -0.0082 | -0.0043 | 40.46 | 54.17 | 0.35 | 12.15 | 2880 |
| Degree-matched shuffled transitions delivered | 0.0038 | 0.0020 | 40.46 | 60.87 | 7.33 | 15.10 | 2880 |
| Edge-count matched dense transitions pool | 0.0101 | 0.0053 | 40.46 | 54.17 | 13.78 | 2.99 | 2880 |
| Edge-count matched dense transitions delivered | 0.0440 | 0.0231 | 40.46 | 67.43 | 21.46 | 7.19 | 2880 |
| HippoRAG2 pool | 0.0313 | 0.0164 | 40.46 | 65.10 | 39.44 | 28.68 | 2880 |
| PropRAG pool | 0.1705 | 0.0895 | 40.46 | 77.74 | 52.29 | 19.72 | 2880 |

## By Dataset

| method | dataset | FCRG | avg_delta | rho_f | method_at5 | baseline_at5 | promoted | demoted | fact_docs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dense query-only | HotpotQA | 0.0000 | 0.0000 | 42.45 | 88.34 | 88.34 | 0.00 | 0.00 | 849 |
| EvidenceLink pool | HotpotQA | -0.0498 | -0.0169 | 42.45 | 92.82 | 88.34 | 28.03 | 33.10 | 849 |
| EvidenceLink delivered | HotpotQA | -0.0416 | -0.0141 | 42.45 | 96.00 | 88.34 | 29.21 | 32.27 | 849 |
| w/o evidence-linked transitions pool | HotpotQA | -0.0020 | -0.0007 | 42.45 | 92.34 | 88.34 | 27.33 | 27.56 | 849 |
| w/o evidence-linked transitions delivered | HotpotQA | 0.0044 | 0.0015 | 42.45 | 94.70 | 88.34 | 28.27 | 27.56 | 849 |
| Dense-doc KNN transitions pool | HotpotQA | 0.0028 | 0.0009 | 42.45 | 88.34 | 88.34 | 2.12 | 0.24 | 849 |
| Dense-doc KNN transitions delivered | HotpotQA | 0.0173 | 0.0059 | 42.45 | 93.29 | 88.34 | 6.24 | 1.77 | 849 |
| Degree-matched shuffled transitions pool | HotpotQA | -0.0015 | -0.0005 | 42.45 | 88.34 | 88.34 | 0.00 | 1.30 | 849 |
| Degree-matched shuffled transitions delivered | HotpotQA | 0.0076 | 0.0026 | 42.45 | 92.23 | 88.34 | 4.00 | 2.36 | 849 |
| Edge-count matched dense transitions pool | HotpotQA | 0.0017 | 0.0006 | 42.45 | 88.34 | 88.34 | 1.53 | 0.12 | 849 |
| Edge-count matched dense transitions delivered | HotpotQA | 0.0152 | 0.0052 | 42.45 | 93.05 | 88.34 | 5.65 | 1.77 | 849 |
| HippoRAG2 pool | HotpotQA | -0.1402 | -0.0476 | 42.45 | 88.22 | 88.34 | 21.67 | 37.81 | 849 |
| PropRAG pool | HotpotQA | 0.0793 | 0.0269 | 42.45 | 94.11 | 88.34 | 30.98 | 22.38 | 849 |
| Dense query-only | 2WikiMultiHopQA | 0.0000 | 0.0000 | 45.26 | 34.53 | 34.53 | 0.00 | 0.00 | 1118 |
| EvidenceLink pool | 2WikiMultiHopQA | 0.3099 | 0.1965 | 45.26 | 85.96 | 34.53 | 68.60 | 14.22 | 1118 |
| EvidenceLink delivered | 2WikiMultiHopQA | 0.3228 | 0.2046 | 45.26 | 91.68 | 34.53 | 70.30 | 12.43 | 1118 |
| w/o evidence-linked transitions pool | 2WikiMultiHopQA | 0.2672 | 0.1694 | 45.26 | 72.63 | 34.53 | 65.47 | 13.77 | 1118 |
| w/o evidence-linked transitions delivered | 2WikiMultiHopQA | 0.2930 | 0.1857 | 45.26 | 84.08 | 34.53 | 67.26 | 11.99 | 1118 |
| Dense-doc KNN transitions pool | 2WikiMultiHopQA | 0.0284 | 0.0180 | 45.26 | 34.53 | 34.53 | 40.43 | 4.38 | 1118 |
| Dense-doc KNN transitions delivered | 2WikiMultiHopQA | 0.1023 | 0.0648 | 45.26 | 65.74 | 34.53 | 51.52 | 5.72 | 1118 |
| Degree-matched shuffled transitions pool | 2WikiMultiHopQA | -0.0090 | -0.0057 | 45.26 | 34.53 | 34.53 | 0.72 | 17.89 | 1118 |
| Degree-matched shuffled transitions delivered | 2WikiMultiHopQA | 0.0070 | 0.0045 | 45.26 | 44.81 | 34.53 | 10.55 | 19.50 | 1118 |
| Edge-count matched dense transitions pool | 2WikiMultiHopQA | 0.0169 | 0.0107 | 45.26 | 34.53 | 34.53 | 24.06 | 2.68 | 1118 |
| Edge-count matched dense transitions delivered | 2WikiMultiHopQA | 0.0694 | 0.0440 | 45.26 | 58.32 | 34.53 | 34.88 | 4.47 | 1118 |
| HippoRAG2 pool | 2WikiMultiHopQA | 0.1262 | 0.0800 | 45.26 | 59.39 | 34.53 | 58.05 | 15.03 | 1118 |
| PropRAG pool | 2WikiMultiHopQA | 0.2719 | 0.1723 | 45.26 | 81.40 | 34.53 | 70.30 | 10.38 | 1118 |
| Dense query-only | MuSiQue | 0.0000 | 0.0000 | 34.48 | 46.44 | 46.44 | 0.00 | 0.00 | 913 |
| EvidenceLink pool | MuSiQue | 0.0225 | 0.0127 | 34.48 | 56.30 | 46.44 | 40.53 | 37.46 | 913 |
| EvidenceLink delivered | MuSiQue | 0.0390 | 0.0219 | 34.48 | 61.56 | 46.44 | 41.73 | 36.80 | 913 |
| w/o evidence-linked transitions pool | MuSiQue | 0.0261 | 0.0147 | 34.48 | 56.08 | 46.44 | 37.24 | 35.16 | 913 |
| w/o evidence-linked transitions delivered | MuSiQue | 0.0371 | 0.0209 | 34.48 | 60.13 | 46.44 | 38.01 | 34.83 | 913 |
| Dense-doc KNN transitions pool | MuSiQue | 0.0057 | 0.0032 | 34.48 | 46.44 | 46.44 | 14.90 | 8.54 | 913 |
| Dense-doc KNN transitions delivered | MuSiQue | 0.0329 | 0.0185 | 34.48 | 57.06 | 46.44 | 21.80 | 16.98 | 913 |
| Degree-matched shuffled transitions pool | MuSiQue | -0.0107 | -0.0060 | 34.48 | 46.44 | 46.44 | 0.22 | 15.22 | 913 |
| Degree-matched shuffled transitions delivered | MuSiQue | -0.0027 | -0.0015 | 34.48 | 51.37 | 46.44 | 6.46 | 21.58 | 913 |
| Edge-count matched dense transitions pool | MuSiQue | 0.0054 | 0.0030 | 34.48 | 46.44 | 46.44 | 12.60 | 6.02 | 913 |
| Edge-count matched dense transitions delivered | MuSiQue | 0.0251 | 0.0141 | 34.48 | 54.76 | 46.44 | 19.72 | 15.55 | 913 |
| HippoRAG2 pool | MuSiQue | -0.0032 | -0.0018 | 34.48 | 50.60 | 46.44 | 33.19 | 36.91 | 913 |
| PropRAG pool | MuSiQue | 0.0819 | 0.0461 | 34.48 | 58.05 | 46.44 | 50.05 | 28.70 | 913 |

