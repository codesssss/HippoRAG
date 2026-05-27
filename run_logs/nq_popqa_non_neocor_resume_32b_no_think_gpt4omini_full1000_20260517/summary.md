# NQ/PopQA 32B no_think + GPT-4o-mini Summary

| Dataset | Method group | Method | Count | R@5 | R@20 | R@100 | R@200 | EM | F1 | Reader docs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nq | dense | dense_entry | 1000 | 0.7716 | 0.9907 | 0.9988 |  | 0.4940 | 0.6329 | 5.0000 |
| popqa | dense | dense_entry | 1000 | 0.4920 | 0.5635 | 0.6800 |  | 0.4790 | 0.6042 | 5.0000 |
| nq | hipporag | hipporag_qwen32b_no_think_top200 | 1000 | 0.7984 | 0.9917 | 0.9993 | 0.9995 | 0.5030 | 0.6336 | 5.0000 |
| popqa | hipporag | hipporag_qwen32b_no_think_top200 | 1000 | 0.4960 | 0.5830 | 0.7020 | 0.7585 | 0.4770 | 0.6033 | 5.0000 |
| nq | proprag | proprag_qwen32b_no_think_top200 | 1000 | 0.8105 | 0.9926 | 0.9988 | 0.9993 | 0.5010 | 0.6373 | 5.0000 |
| popqa | proprag | proprag_qwen32b_no_think_top200 | 1000 | 0.5645 | 0.7070 | 0.8040 | 0.8550 | 0.4940 | 0.6155 | 5.0000 |
| nq | evidenceflow | evidence_transition_graphragv3_variable_flow | 1000 | 0.7163 |  |  |  | 0.5190 | 0.6504 |  |
| popqa | evidenceflow | evidence_transition_graphragv3_variable_flow | 1000 | 0.6495 |  |  |  | 0.4940 | 0.6223 |  |
