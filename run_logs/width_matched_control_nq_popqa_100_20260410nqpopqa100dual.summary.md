# Width-Matched Control: NQ + PopQA top-5 (limit=100)

Naming note:
- `nq` resolves to the packaged `nq_rear` files already present in `reproduce/dataset/`.
- `nq_rear` and `popqa` here are the repo/HippoRAG-bundle `1000`-query evaluation subsets.
- `baseline_top10_plus_ce` = width-matched no-append CE control over the top-10 baseline prefix.
- `bridge_append_plus_ce` = same-pool pure CE rerank over `top-10 baseline prefix + 3 bridge-appended docs`.
- `random3_deep_plus_ce` = matched random-append CE control.

| Dataset | Run | EM | F1 | R@5 | R@20 | R@100 | num_queries | ΔEM vs baseline top-5 | ΔEM vs baseline top-10+CE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nq | baseline_top5 | 0.5200 | 0.6500 | 0.7105 | 0.9872 | 1.0000 | 100 | — | — |
| nq | baseline_top10_plus_ce | 0.5300 | 0.6444 | 0.6882 | 0.9872 | 1.0000 | 100 | +0.0100 | +0.0000 |
| nq | random3_deep_plus_ce | 0.5200 | 0.6327 | 0.6849 | 0.9852 | 1.0000 | 100 | +0.0000 | -0.0100 |
| nq | bridge_append_plus_ce | 0.5300 | 0.6444 | 0.6882 | 0.9872 | 1.0000 | 100 | +0.0100 | +0.0000 |
| popqa | baseline_top5 | 0.1800 | 0.4811 | 0.5000 | 0.5300 | 0.5750 | 100 | — | — |
| popqa | baseline_top10_plus_ce | 0.2000 | 0.4819 | 0.5000 | 0.5300 | 0.5750 | 100 | +0.0200 | +0.0000 |
| popqa | random3_deep_plus_ce | 0.2000 | 0.4829 | 0.5000 | 0.5300 | 0.5750 | 100 | +0.0200 | +0.0000 |
| popqa | bridge_append_plus_ce | 0.2000 | 0.4819 | 0.5000 | 0.5300 | 0.5750 | 100 | +0.0200 | +0.0000 |
