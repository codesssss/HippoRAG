# EvidenceLink Mechanism Audits

## Pool-to-Top5 Conversion

| method | dataset | delivered_top5_r_pct | delivered_top5_all_pct | pool_recall_pct | pool_all_pct | pool_to_delivered_r_gap_pct | pool_to_delivered_all_gap_pct | changed_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EvidenceLink | HotpotQA | 96.55 | 93.4 | 99.75 | 99.6 | 3.2 | 6.2 | 149 |
| w/o evidence-linked transitions | HotpotQA | 96.25 | 92.9 | 99.5 | 99.1 | 3.25 | 6.2 | 149 |
| HippoRAG2 | HotpotQA | 92.6 | 85.9 | 99.75 | 99.5 | 7.15 | 13.6 | 0 |
| PropRAG | HotpotQA | 95.1 | 90.7 | 99.95 | 99.9 | 4.85 | 9.2 | 0 |
| EvidenceLink | 2WikiMultiHopQA | 96.3 | 89.7 | 99.25 | 98.2 | 2.95 | 8.5 | 334 |
| w/o evidence-linked transitions | 2WikiMultiHopQA | 93.87 | 82.1 | 98.28 | 95.2 | 4.41 | 13.1 | 403 |
| HippoRAG2 | 2WikiMultiHopQA | 82.75 | 60.0 | 97.3 | 91.8 | 14.55 | 31.8 | 0 |
| PropRAG | 2WikiMultiHopQA | 90.9 | 78.0 | 99.6 | 98.9 | 8.7 | 20.9 | 0 |
| EvidenceLink | MuSiQue | 76.42 | 51.0 | 94.32 | 85.2 | 17.9 | 34.2 | 543 |
| w/o evidence-linked transitions | MuSiQue | 75.02 | 48.1 | 90.07 | 76.2 | 15.05 | 28.1 | 520 |
| HippoRAG2 | MuSiQue | 70.96 | 42.3 | 95.66 | 88.3 | 24.7 | 46.0 | 0 |
| PropRAG | MuSiQue | 74.24 | 47.6 | 98.41 | 95.3 | 24.17 | 47.7 | 0 |

## Transition-Witness Win Audit

| dataset | admit_decisions | added_docs | added_from_graph_tail | added_from_graph_tail_pct | added_gold | added_gold_from_graph_tail | all_gold_rescues | all_gold_rescues_from_graph_tail | all_gold_regressions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HotpotQA | 149 | 149 | 128 | 85.91 | 30 | 23 | 28 | 22 | 0 |
| 2WikiMultiHopQA | 334 | 334 | 295 | 88.32 | 83 | 75 | 78 | 71 | 9 |
| MuSiQue | 561 | 561 | 512 | 91.27 | 103 | 86 | 60 | 47 | 22 |

