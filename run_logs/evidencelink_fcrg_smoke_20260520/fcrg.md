# EvidenceLink FCRG Mechanism Diagnostic

FCRG measures whether a method promotes gold supporting documents whose relevance is licensed by a source-grounded OpenIE fact transition from another gold supporting document. Dense query-only is the reference ranker, so its FCRG is expected to be 0.

## Aggregate

| method | FCRG | avg_delta | rho_f | method_at5 | promoted | demoted | fact_docs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense query-only | 0.0000 | 0.0000 | 42.45 | 88.34 | 0.00 | 0.00 | 849 |
| EvidenceLink pool | -0.0498 | -0.0169 | 42.45 | 92.82 | 28.03 | 33.10 | 849 |
| EvidenceLink delivered | -0.0416 | -0.0141 | 42.45 | 96.00 | 29.21 | 32.27 | 849 |
| w/o evidence-linked transitions pool | -0.0020 | -0.0007 | 42.45 | 92.34 | 27.33 | 27.56 | 849 |
| w/o evidence-linked transitions delivered | 0.0044 | 0.0015 | 42.45 | 94.70 | 28.27 | 27.56 | 849 |
| Dense-doc KNN transitions pool | 0.0028 | 0.0009 | 42.45 | 88.34 | 2.12 | 0.24 | 849 |
| Dense-doc KNN transitions delivered | 0.0173 | 0.0059 | 42.45 | 93.29 | 6.24 | 1.77 | 849 |
| Degree-matched shuffled transitions pool | -0.0015 | -0.0005 | 42.45 | 88.34 | 0.00 | 1.30 | 849 |
| Degree-matched shuffled transitions delivered | 0.0076 | 0.0026 | 42.45 | 92.23 | 4.00 | 2.36 | 849 |
| Edge-count matched dense transitions pool | 0.0017 | 0.0006 | 42.45 | 88.34 | 1.53 | 0.12 | 849 |
| HippoRAG2 pool | -0.1402 | -0.0476 | 42.45 | 88.22 | 21.67 | 37.81 | 849 |
| PropRAG pool | 0.0793 | 0.0269 | 42.45 | 94.11 | 30.98 | 22.38 | 849 |

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
| HippoRAG2 pool | HotpotQA | -0.1402 | -0.0476 | 42.45 | 88.22 | 88.34 | 21.67 | 37.81 | 849 |
| PropRAG pool | HotpotQA | 0.0793 | 0.0269 | 42.45 | 94.11 | 88.34 | 30.98 | 22.38 | 849 |

