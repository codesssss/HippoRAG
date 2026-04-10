# Full-Scale Width-Matched Control

Naming note:
- `baseline_top10_plus_ce` = width-matched no-append CE control over the top-10 baseline prefix.
- `bridge_append_plus_ce` = same-pool pure CE rerank over `top-10 baseline prefix + 3 bridge-appended docs`.
- `random3_deep_plus_ce` = matched random-append CE control.
- This `20260409fullfix` family is the first valid full-scale version of the width-matched CE controls and is method-matched to the earlier `limit=100` runs.

| Phase | Dataset | Run | EM | F1 | R@5 | R@20 | R@100 | ΔEM vs baseline | ΔEM vs baseline top10+CE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| top5_all | musique | baseline_top5 | 0.2650 | 0.3464 | 0.6196 | 0.7922 | 0.9033 | — | — |
| top5_all | musique | baseline_top10_plus_ce | 0.3070 | 0.3882 | 0.6603 | 0.7923 | 0.9033 | +0.0410 | +0.0000 |
| top5_all | musique | random3_deep_plus_ce | 0.3040 | 0.3848 | 0.6548 | 0.7950 | 0.9033 | +0.0390 | -0.0030 |
| top5_all | musique | bridge_append_plus_ce | 0.3110 | 0.3917 | 0.6603 | 0.7953 | 0.9033 | +0.0460 | +0.0040 |
