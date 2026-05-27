# Transition Top5 QA

This report evaluates fixed reader-facing top5 evidence. R@10/R@200 are not optimization targets here.

## Metrics

| dataset | method | R@5 | all-gold@5 | answer-string@5 | EM | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| musique | obligation_closed_sto_local_ppr | 0.6775 | 0.3900 | 0.5800 | 0.3900 | 0.4474 |

## Delta Vs SFB

| dataset | method | delta R@5 | delta all-gold@5 | delta answer-string@5 | delta EM | delta F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |

## QA Failure Buckets

### musique

| method | bucket | count |
| --- | --- | ---: |
| obligation_closed_sto_local_ppr | all_gold_top5_qa_wrong | 13 |
| obligation_closed_sto_local_ppr | no_gold_top5_qa_wrong | 5 |
| obligation_closed_sto_local_ppr | partial_gold_top5_qa_wrong | 43 |
| obligation_closed_sto_local_ppr | qa_exact_all_gold_top5 | 26 |
| obligation_closed_sto_local_ppr | qa_exact_partial_gold_top5 | 13 |
