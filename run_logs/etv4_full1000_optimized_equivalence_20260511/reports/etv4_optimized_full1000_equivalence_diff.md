# ETv4 Optimized Full1000 Equivalence Diff

| Dataset         | Rows | Old R@5  | New R@5  | Old all-gold@5 | New all-gold@5 | Top5 mismatch | Elapsed  |
| --------------- | ---- | -------- | -------- | -------------- | -------------- | ------------- | -------- |
| 2wikimultihopqa | 1000 | 0.920000 | 0.920000 | 0.797000       | 0.797000       | 0             | 14:37.74 |
| musique         | 1000 | 0.724917 | 0.724917 | 0.432000       | 0.432000       | 0             | 26:21.79 |
| hotpotqa        | 1000 | 0.950500 | 0.950500 | 0.906000       | 0.906000       | 0             | 21:45.84 |

Old root: `run_logs/etv4_clean_mainline_full1000_reuse_gpt4omini_20260511`
New root: `run_logs/etv4_full1000_optimized_equivalence_20260511`

Decision: full1000 retrieval top5 is identical for all compared datasets.
