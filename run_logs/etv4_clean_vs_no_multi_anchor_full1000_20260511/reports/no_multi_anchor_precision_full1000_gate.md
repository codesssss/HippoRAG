# ETv4 No Multi-Anchor Precision Full1000 Retrieval Gate

| Dataset | Rows | Clean R@5 | No-rule R@5 | Delta R@5 | Clean all-gold@5 | No-rule all-gold@5 | Delta all-gold@5 | Top5 changed | Gold gains | Gold losses | Net gold losses | Decision signal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2wikimultihopqa | 1000 | 0.9200 | 0.9200 | +0.0000 | 0.7970 | 0.7970 | +0.0000 | 3 | 0 | 0 | 0 | retrieval_tie_delete_candidate |
| musique | 1000 | 0.7249 | 0.7238 | -0.0012 | 0.4320 | 0.4300 | -0.0020 | 39 | 1 | 5 | 4 | retrieval_loss_keep_candidate |
| hotpotqa | 1000 | 0.9505 | 0.9505 | +0.0000 | 0.9060 | 0.9060 | +0.0000 | 52 | 0 | 0 | 0 | retrieval_tie_delete_candidate |
