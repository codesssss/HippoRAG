# Width-Matched Control: top-5

Naming note:
- `baseline_top5_plus_ce` = baseline-only CE rerank, not the same-pool append control.
- `baseline_top10_plus_ce` = width-matched no-append CE control over the top-10 baseline prefix.
- `bridge_append_plus_ce` = same-pool pure CE rerank over `top-10 baseline prefix + 3 bridge-appended docs`.
- `random3_deep_plus_ce` = matched random-append CE control.
- This `limit=100` summary is method-matched to the later valid `20260409fullfix` full-scale control family.

| Dataset | Run | EM | F1 | ΔEM vs baseline top-5 | ΔEM vs baseline top-10+CE |
|---|---|---:|---:|---:|---:|
| musique | baseline_top5 | 0.2700 | 0.3348 | — | — |
| musique | baseline_top5_plus_ce | 0.3100 | 0.3577 | +0.0400 | — |
| musique | baseline_top10_plus_ce | 0.3100 | 0.3698 | +0.0400 | +0.0000 |
| musique | next3_deep_plus_ce | 0.3100 | 0.3706 | +0.0400 | +0.0000 |
| musique | random3_deep_plus_ce | 0.2900 | 0.3524 | +0.0200 | -0.0200 |
| musique | bridge_append_plus_ce | 0.3200 | 0.3810 | +0.0500 | +0.0100 |
| hotpotqa | baseline_top5 | 0.5800 | 0.6967 | — | — |
| hotpotqa | baseline_top5_plus_ce | 0.5600 | 0.6746 | -0.0200 | — |
| hotpotqa | baseline_top10_plus_ce | 0.6200 | 0.7173 | +0.0400 | +0.0000 |
| hotpotqa | next3_deep_plus_ce | 0.6100 | 0.7186 | +0.0300 | -0.0100 |
| hotpotqa | random3_deep_plus_ce | 0.6100 | 0.7073 | +0.0300 | -0.0100 |
| hotpotqa | bridge_append_plus_ce | 0.6200 | 0.7173 | +0.0400 | +0.0000 |
| 2wikimultihopqa | baseline_top5 | 0.3600 | 0.4008 | — | — |
| 2wikimultihopqa | baseline_top5_plus_ce | 0.3700 | 0.4083 | +0.0100 | — |
| 2wikimultihopqa | baseline_top10_plus_ce | 0.4200 | 0.4553 | +0.0600 | +0.0000 |
| 2wikimultihopqa | next3_deep_plus_ce | 0.3800 | 0.4601 | +0.0200 | -0.0400 |
| 2wikimultihopqa | random3_deep_plus_ce | 0.3900 | 0.4348 | +0.0300 | -0.0300 |
| 2wikimultihopqa | bridge_append_plus_ce | 0.4300 | 0.4728 | +0.0700 | +0.0100 |
