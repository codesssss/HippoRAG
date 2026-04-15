# Bridge Failure Diagnosis: NQ + PopQA (limit=100)

- control date tag: `20260410nqpopqa100dual`
- diagnosis date tag: `20260410bridge_diag`
- `baseline_top10_plus_ce` is the no-append width-matched CE control.
- `bridge_append_plus_ce_default` is the original bridge setting from the previous 100-query run.
- `append_query_rate` = share of queries with at least one appended bridge doc.
- `final_appended_rate` = share of queries whose final top-5 contains at least one appended doc.

| Dataset | Run | q_source | thr | EM | F1 | R@5 | R@20 | R@100 | append_q_rate | avg_append | final_app_q_rate | changed_vs_base | query_ent_rate | proposal_ent_rate | covered_pos_rate | top_stop_reasons |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| nq | baseline_top10_plus_ce | — | 0.3500 | 0.5300 | 0.6444 | 0.6882 | 0.9872 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9000 | 0.0500 | 0.0500 | 0.0000 | append_cap_zero:100 |
| nq | bridge_append_plus_ce_default | — | 0.3500 | 0.5300 | 0.6444 | 0.6882 | 0.9872 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9000 | 0.0500 | 0.0500 | 0.0000 | structure_below_threshold:100 |
| nq | bridge_diag_nq_question_thr000 | — | 0.0000 | 0.5300 | 0.6444 | 0.6922 | 0.9872 | 1.0000 | 1.0000 | 3.0000 | 0.1300 | 0.9000 | 0.0500 | 0.0500 | 0.0000 | append_cap_reached:100 |
| nq | bridge_diag_nq_question_thr035 | — | 0.3500 | 0.5300 | 0.6444 | 0.6882 | 0.9872 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9000 | 0.0500 | 0.0500 | 0.0000 | structure_below_threshold:100 |
| nq | bridge_diag_nq_seed_thr000 | — | 0.0000 | 0.5300 | 0.6444 | 0.6922 | 0.9872 | 1.0000 | 1.0000 | 3.0000 | 0.1300 | 0.9000 | 0.0500 | 0.0500 | 0.0000 | append_cap_reached:100 |
| popqa | baseline_top10_plus_ce | — | 0.3500 | 0.2000 | 0.4819 | 0.5000 | 0.5300 | 0.5750 | 0.0000 | 0.0000 | 0.0000 | 0.9800 | 0.9900 | 0.9900 | 0.0000 | append_cap_zero:100 |
| popqa | bridge_append_plus_ce_default | — | 0.3500 | 0.2000 | 0.4819 | 0.5000 | 0.5300 | 0.5750 | 0.0000 | 0.0000 | 0.0000 | 0.9800 | 0.9900 | 0.9900 | 0.0000 | structure_below_threshold:100 |
| popqa | bridge_diag_popqa_question_thr000 | — | 0.0000 | 0.2100 | 0.4853 | 0.5000 | 0.5300 | 0.5750 | 1.0000 | 3.0000 | 0.2100 | 0.9900 | 0.9900 | 0.9900 | 0.0000 | append_cap_reached:100 |
| popqa | bridge_diag_popqa_question_thr035 | — | 0.3500 | 0.2000 | 0.4819 | 0.5000 | 0.5300 | 0.5750 | 0.0000 | 0.0000 | 0.0000 | 0.9800 | 0.9900 | 0.9900 | 0.0000 | structure_below_threshold:100 |
| popqa | bridge_diag_popqa_seed_thr000 | — | 0.0000 | 0.2100 | 0.4853 | 0.5000 | 0.5300 | 0.5750 | 1.0000 | 3.0000 | 0.2100 | 0.9900 | 0.9900 | 0.9900 | 0.0000 | append_cap_reached:100 |
