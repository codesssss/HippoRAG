# Smoke Width-Matched Control: NQ + PopQA top-5

Naming note:
- `nq` resolves to the packaged `nq_rear` files already present in `reproduce/dataset/`.
- `nq_rear` and `popqa` here are the repo/HippoRAG-bundle `1000`-query evaluation subsets.
- This light smoke uses `limit=5` and only runs the width-matched no-append CE control.

| Dataset | Run | EM | F1 |
|---|---|---:|---:|
| nq | width_match_baseline_top10_plus_ce_qatopk5 | 0.6000 | 0.6800 |
| popqa | width_match_baseline_top10_plus_ce_qatopk5 | 0.0000 | 0.4152 |
