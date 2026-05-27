# ETv3 Reader-Only QA

This report evaluates fixed ETv3 top-5 evidence. It does not load SFB variants, run causal extraction, or re-index the corpus.

## Metrics

| dataset | method | R@5 | all-gold@5 | answer-string@5 | EM | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| musique | evidence_transition_graphragv3_variable_flow | 0.7092 | 0.3900 | 0.6000 | 0.3600 | 0.4282 |

## QA Failure Buckets

### musique

| method | bucket | count |
| --- | --- | ---: |
| evidence_transition_graphragv3_variable_flow | all_gold_top5_qa_wrong | 16 |
| evidence_transition_graphragv3_variable_flow | no_gold_top5_qa_wrong | 1 |
| evidence_transition_graphragv3_variable_flow | partial_gold_top5_qa_wrong | 47 |
| evidence_transition_graphragv3_variable_flow | qa_exact_all_gold_top5 | 23 |
| evidence_transition_graphragv3_variable_flow | qa_exact_partial_gold_top5 | 13 |
