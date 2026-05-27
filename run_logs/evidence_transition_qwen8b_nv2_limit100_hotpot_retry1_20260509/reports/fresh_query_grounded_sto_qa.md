# Transition Top5 QA

This report evaluates fixed reader-facing top5 evidence. R@10/R@200 are not optimization targets here.

## Metrics

| dataset | method | R@5 | all-gold@5 | answer-string@5 | EM | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hotpotqa | evidence_transition_graphrag | 0.9500 | 0.9000 | 0.9100 | 0.6000 | 0.7029 |

## Delta Vs SFB

| dataset | method | delta R@5 | delta all-gold@5 | delta answer-string@5 | delta EM | delta F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |

## QA Failure Buckets

### hotpotqa

| method | bucket | count |
| --- | --- | ---: |
| evidence_transition_graphrag | all_gold_top5_qa_wrong | 32 |
| evidence_transition_graphrag | partial_gold_top5_qa_wrong | 8 |
| evidence_transition_graphrag | qa_exact_all_gold_top5 | 58 |
| evidence_transition_graphrag | qa_exact_partial_gold_top5 | 2 |
