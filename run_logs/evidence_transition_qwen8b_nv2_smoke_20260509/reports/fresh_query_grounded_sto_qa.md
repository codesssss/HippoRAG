# Transition Top5 QA

This report evaluates fixed reader-facing top5 evidence. R@10/R@200 are not optimization targets here.

## Metrics

| dataset | method | R@5 | all-gold@5 | answer-string@5 | EM | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | evidence_transition_graphrag | 1.0000 | 1.0000 | 1.0000 | 0.8000 | 0.9500 |

## Delta Vs SFB

| dataset | method | delta R@5 | delta all-gold@5 | delta answer-string@5 | delta EM | delta F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |

## QA Failure Buckets

### 2wikimultihopqa

| method | bucket | count |
| --- | --- | ---: |
| evidence_transition_graphrag | all_gold_top5_qa_wrong | 1 |
| evidence_transition_graphrag | qa_exact_all_gold_top5 | 4 |
