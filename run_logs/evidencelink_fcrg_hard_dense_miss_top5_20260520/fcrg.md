# EvidenceLink FCRG Mechanism Diagnostic

FCRG measures whether a method promotes gold supporting documents whose relevance is licensed by a source-grounded OpenIE fact transition from another gold supporting document. Dense query-only is the reference ranker, so its FCRG is expected to be 0.

## Aggregate

| method | FCRG | avg_delta | rho_f | method_at5 | promoted | demoted | fact_docs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense query-only | 0.0000 | 0.0000 | 18.54 | 0.00 | 0.00 | 0.00 | 1320 |
| EvidenceLink pool | 0.2807 | 0.2218 | 18.54 | 57.12 | 79.85 | 13.41 | 1320 |
| EvidenceLink delivered | 0.2995 | 0.2366 | 18.54 | 67.58 | 82.88 | 11.44 | 1320 |
| w/o evidence-linked transitions pool | 0.2431 | 0.1921 | 18.54 | 44.55 | 73.64 | 14.55 | 1320 |
| w/o evidence-linked transitions delivered | 0.2675 | 0.2113 | 18.54 | 58.56 | 76.29 | 13.11 | 1320 |
| Dense-doc KNN transitions pool | 0.0229 | 0.0181 | 18.54 | 0.00 | 45.91 | 9.77 | 1320 |
| Dense-doc KNN transitions delivered | 0.0911 | 0.0720 | 18.54 | 38.41 | 62.73 | 16.29 | 1320 |
| Degree-matched shuffled transitions pool | -0.0118 | -0.0093 | 18.54 | 0.00 | 0.76 | 26.52 | 1320 |
| Degree-matched shuffled transitions delivered | 0.0059 | 0.0047 | 18.54 | 15.68 | 15.98 | 31.89 | 1320 |
| Edge-count matched dense transitions pool | 0.0146 | 0.0115 | 18.54 | 0.00 | 30.08 | 6.52 | 1320 |
| Edge-count matched dense transitions delivered | 0.0642 | 0.0507 | 18.54 | 30.30 | 46.82 | 14.32 | 1320 |
| HippoRAG2 pool | 0.1303 | 0.1030 | 18.54 | 32.12 | 65.00 | 13.26 | 1320 |
| PropRAG pool | 0.2516 | 0.1988 | 18.54 | 58.18 | 84.70 | 9.47 | 1320 |

## By Dataset

| method | dataset | FCRG | avg_delta | rho_f | method_at5 | baseline_at5 | promoted | demoted | fact_docs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dense query-only | HotpotQA | 0.0000 | 0.0000 | 4.95 | 0.00 | 0.00 | 0.00 | 0.00 | 99 |
| EvidenceLink pool | HotpotQA | 0.2379 | 0.1735 | 4.95 | 47.47 | 0.00 | 76.77 | 12.12 | 99 |
| EvidenceLink delivered | HotpotQA | 0.2671 | 0.1949 | 4.95 | 69.70 | 0.00 | 86.87 | 6.06 | 99 |
| w/o evidence-linked transitions pool | HotpotQA | 0.2218 | 0.1618 | 4.95 | 42.42 | 0.00 | 67.68 | 13.13 | 99 |
| w/o evidence-linked transitions delivered | HotpotQA | 0.2445 | 0.1783 | 4.95 | 59.60 | 0.00 | 75.76 | 14.14 | 99 |
| Dense-doc KNN transitions pool | HotpotQA | 0.0111 | 0.0081 | 4.95 | 0.00 | 0.00 | 18.18 | 2.02 | 99 |
| Dense-doc KNN transitions delivered | HotpotQA | 0.0701 | 0.0511 | 4.95 | 44.44 | 0.00 | 53.54 | 13.13 | 99 |
| Degree-matched shuffled transitions pool | HotpotQA | -0.0059 | -0.0043 | 4.95 | 0.00 | 0.00 | 0.00 | 11.11 | 99 |
| Degree-matched shuffled transitions delivered | HotpotQA | 0.0308 | 0.0225 | 4.95 | 34.34 | 0.00 | 34.34 | 19.19 | 99 |
| Edge-count matched dense transitions pool | HotpotQA | 0.0067 | 0.0049 | 4.95 | 0.00 | 0.00 | 13.13 | 1.01 | 99 |
| Edge-count matched dense transitions delivered | HotpotQA | 0.0614 | 0.0448 | 4.95 | 42.42 | 0.00 | 48.48 | 13.13 | 99 |
| HippoRAG2 pool | HotpotQA | 0.1142 | 0.0833 | 4.95 | 37.37 | 0.00 | 66.67 | 14.14 | 99 |
| PropRAG pool | HotpotQA | 0.2967 | 0.2164 | 4.95 | 69.70 | 0.00 | 83.84 | 10.10 | 99 |
| Dense query-only | 2WikiMultiHopQA | 0.0000 | 0.0000 | 29.64 | 0.00 | 0.00 | 0.00 | 0.00 | 732 |
| EvidenceLink pool | 2WikiMultiHopQA | 0.3930 | 0.3177 | 29.64 | 80.05 | 0.00 | 93.17 | 4.64 | 732 |
| EvidenceLink delivered | 2WikiMultiHopQA | 0.4081 | 0.3298 | 29.64 | 88.66 | 0.00 | 95.77 | 2.60 | 732 |
| w/o evidence-linked transitions pool | 2WikiMultiHopQA | 0.3337 | 0.2697 | 29.64 | 59.29 | 0.00 | 87.02 | 5.74 | 732 |
| w/o evidence-linked transitions delivered | 2WikiMultiHopQA | 0.3639 | 0.2942 | 29.64 | 75.96 | 0.00 | 89.75 | 3.69 | 732 |
| Dense-doc KNN transitions pool | 2WikiMultiHopQA | 0.0340 | 0.0275 | 29.64 | 0.00 | 0.00 | 61.75 | 6.69 | 732 |
| Dense-doc KNN transitions delivered | 2WikiMultiHopQA | 0.1225 | 0.0990 | 29.64 | 47.81 | 0.00 | 78.69 | 8.61 | 732 |
| Degree-matched shuffled transitions pool | 2WikiMultiHopQA | -0.0108 | -0.0087 | 29.64 | 0.00 | 0.00 | 1.09 | 27.32 | 732 |
| Degree-matched shuffled transitions delivered | 2WikiMultiHopQA | 0.0085 | 0.0068 | 29.64 | 15.85 | 0.00 | 16.12 | 29.64 | 732 |
| Edge-count matched dense transitions pool | 2WikiMultiHopQA | 0.0202 | 0.0163 | 29.64 | 0.00 | 0.00 | 36.75 | 4.10 | 732 |
| Edge-count matched dense transitions delivered | 2WikiMultiHopQA | 0.0832 | 0.0673 | 29.64 | 36.61 | 0.00 | 53.28 | 6.56 | 732 |
| HippoRAG2 pool | 2WikiMultiHopQA | 0.1703 | 0.1376 | 29.64 | 40.98 | 0.00 | 76.64 | 4.23 | 732 |
| PropRAG pool | 2WikiMultiHopQA | 0.3148 | 0.2544 | 29.64 | 73.36 | 0.00 | 92.90 | 3.69 | 732 |
| Dense query-only | MuSiQue | 0.0000 | 0.0000 | 18.47 | 0.00 | 0.00 | 0.00 | 0.00 | 489 |
| EvidenceLink pool | MuSiQue | 0.1134 | 0.0879 | 18.47 | 24.74 | 0.00 | 60.53 | 26.79 | 489 |
| EvidenceLink delivered | MuSiQue | 0.1362 | 0.1056 | 18.47 | 35.58 | 0.00 | 62.78 | 25.77 | 489 |
| w/o evidence-linked transitions pool | MuSiQue | 0.1057 | 0.0820 | 18.47 | 22.90 | 0.00 | 54.81 | 28.02 | 489 |
| w/o evidence-linked transitions delivered | MuSiQue | 0.1213 | 0.0940 | 18.47 | 32.31 | 0.00 | 56.24 | 26.99 | 489 |
| Dense-doc KNN transitions pool | MuSiQue | 0.0077 | 0.0060 | 18.47 | 0.00 | 0.00 | 27.81 | 15.95 | 489 |
| Dense-doc KNN transitions delivered | MuSiQue | 0.0460 | 0.0356 | 18.47 | 23.11 | 0.00 | 40.70 | 28.43 | 489 |
| Degree-matched shuffled transitions pool | MuSiQue | -0.0145 | -0.0112 | 18.47 | 0.00 | 0.00 | 0.41 | 28.43 | 489 |
| Degree-matched shuffled transitions delivered | MuSiQue | -0.0027 | -0.0021 | 18.47 | 11.66 | 0.00 | 12.07 | 37.83 | 489 |
| Edge-count matched dense transitions pool | MuSiQue | 0.0073 | 0.0056 | 18.47 | 0.00 | 0.00 | 23.52 | 11.25 | 489 |
| Edge-count matched dense transitions delivered | MuSiQue | 0.0351 | 0.0272 | 18.47 | 18.40 | 0.00 | 36.81 | 26.18 | 489 |
| HippoRAG2 pool | MuSiQue | 0.0709 | 0.0550 | 18.47 | 17.79 | 0.00 | 47.24 | 26.58 | 489 |
| PropRAG pool | MuSiQue | 0.1445 | 0.1120 | 18.47 | 33.13 | 0.00 | 72.60 | 18.00 | 489 |

