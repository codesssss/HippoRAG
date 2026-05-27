# Transition Top5 QA

This report evaluates fixed reader-facing top5 evidence. R@10/R@200 are not optimization targets here.

## Metrics

| dataset | method | R@5 | all-gold@5 | answer-string@5 | EM | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| musique | evidence_transition_graphrag | 0.7033 | 0.3800 | 0.5900 | 0.3400 | 0.4082 |

## Delta Vs SFB

| dataset | method | delta R@5 | delta all-gold@5 | delta answer-string@5 | delta EM | delta F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |

## QA Failure Buckets

### musique

| method | bucket | count |
| --- | --- | ---: |
| evidence_transition_graphrag | all_gold_top5_qa_wrong | 17 |
| evidence_transition_graphrag | no_gold_top5_qa_wrong | 1 |
| evidence_transition_graphrag | partial_gold_top5_qa_wrong | 48 |
| evidence_transition_graphrag | qa_exact_all_gold_top5 | 21 |
| evidence_transition_graphrag | qa_exact_partial_gold_top5 | 13 |
