# AG-STO v12 Cached Evaluation

Paper-facing cached-proposal reproduction using the standalone `agsto_v12` package.

## Config

| parameter                  |           value |
| -------------------------- | --------------: |
| `policy`                   |           graph |
| `completion_policy`        | graph_obligated |
| `proposal_source`          |          cached |
| `max_queries`              |               0 |
| `retrieval_top_k`          |              20 |
| `evidence_set_size`        |               5 |
| `stable_anchor_k`          |               2 |
| `proposal_candidate_depth` |              10 |
| `support_proposal_depth`   |               6 |
| `candidate_limit`          |             120 |
| `beam_size`                |              12 |
| `set_search_policy`        |            beam |
| `max_endpoint_degree`      |              30 |

## Metrics

| dataset         | rows |    R@5 |   R@10 | all-gold@5 | all-gold@10 | mean candidates |
| --------------- | ---: | -----: | -----: | ---------: | ----------: | --------------: |
| 2wikimultihopqa | 1000 | 0.9155 | 0.9473 |     0.7840 |      0.8630 |            43.4 |
| musique         | 1000 | 0.6722 | 0.7485 |     0.3980 |      0.4970 |            41.3 |
| hotpotqa        | 1000 | 0.9390 | 0.9675 |     0.8820 |      0.9380 |            36.8 |
