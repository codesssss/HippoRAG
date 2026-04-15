# Width-Matched Control: popqa top-5 (limit=100)

Naming note:
- `nq` resolves to the packaged `nq_rear` files already present in `reproduce/dataset/`.
- `nq_rear` and `popqa` here are the repo/HippoRAG-bundle `1000`-query evaluation subsets.
- `baseline_top10_plus_ce` = width-matched no-append CE control over the top-10 baseline prefix.
- `bridge_append_plus_ce` = same-pool pure CE rerank over `top-10 baseline prefix + 3 bridge-appended docs`.
- `random3_deep_plus_ce` = matched random-append CE control.

| Dataset | Run | EM | F1 | R@5 | R@20 | R@100 | num_queries | ΔEM vs baseline top-5 | ΔEM vs baseline top-10+CE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
