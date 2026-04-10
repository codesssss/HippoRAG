# Baseline+CE Control

Naming note:
- In this older control file, `baseline_plus_ce` means baseline-only CE rerank with `expand_base_k = qa_top_k` and `append_max_docs = 0`.
- In this older control file, `expand_plus_ce` means the same-pool pure CE rerank condition over `top-qa_top_k baseline prefix + 3 bridge-appended docs`.
- This file is useful for baseline-vs-expand attribution, but the later width-matched summaries are the cleaner place to compare `baseline_top10_plus_ce`, `random3_deep_plus_ce`, and `bridge_append_plus_ce`.

| Dataset | Run | qa_top_k | EM | F1 | ΔEM vs baseline |
|---|---|---:|---:|---:|---:|
| musique | baseline | 5 | 0.2700 | 0.3348 | — |
| musique | baseline_plus_ce | 5 | 0.3100 | 0.3577 | 0.0400 |
| musique | expand_plus_ce | 5 | 0.3200 | 0.3810 | 0.0500 |
| musique | baseline | 7 | 0.3200 | 0.3924 | — |
| musique | baseline_plus_ce | 7 | 0.3400 | 0.4007 | 0.0200 |
| musique | expand_plus_ce | 7 | 0.3800 | 0.4385 | 0.0600 |
| musique | baseline | 10 | 0.3400 | 0.4235 | — |
| musique | baseline_plus_ce | 10 | 0.3900 | 0.4412 | 0.0500 |
| musique | expand_plus_ce | 10 | 0.3700 | 0.4422 | 0.0300 |
| hotpotqa | baseline | 5 | 0.5800 | 0.6967 | — |
| hotpotqa | baseline_plus_ce | 5 | 0.5600 | 0.6746 | -0.0200 |
| hotpotqa | expand_plus_ce | 5 | 0.6200 | 0.7173 | 0.0400 |
| hotpotqa | baseline | 7 | 0.5800 | 0.6953 | — |
| hotpotqa | baseline_plus_ce | 7 | 0.6000 | 0.7222 | 0.0200 |
| hotpotqa | expand_plus_ce | 7 | 0.6000 | 0.7136 | 0.0200 |
| hotpotqa | baseline | 10 | 0.6000 | 0.7295 | — |
| hotpotqa | baseline_plus_ce | 10 | 0.5900 | 0.6981 | -0.0100 |
| hotpotqa | expand_plus_ce | 10 | 0.5800 | 0.6881 | -0.0200 |
| 2wikimultihopqa | baseline | 5 | 0.3600 | 0.4008 | — |
| 2wikimultihopqa | baseline_plus_ce | 5 | 0.3700 | 0.4083 | 0.0100 |
| 2wikimultihopqa | expand_plus_ce | 5 | 0.4300 | 0.4728 | 0.0700 |
| 2wikimultihopqa | baseline | 7 | 0.4200 | 0.4642 | — |
| 2wikimultihopqa | baseline_plus_ce | 7 | 0.4300 | 0.4860 | 0.0100 |
| 2wikimultihopqa | expand_plus_ce | 7 | 0.4400 | 0.5192 | 0.0200 |
| 2wikimultihopqa | baseline | 10 | 0.4500 | 0.5118 | — |
| 2wikimultihopqa | baseline_plus_ce | 10 | 0.4600 | 0.4992 | 0.0100 |
| 2wikimultihopqa | expand_plus_ce | 10 | 0.5000 | 0.5403 | 0.0600 |
