# Pool Support Depth Analysis

- Pool: `/mnt/nvme/code/HippoRAG/run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json`
- Dataset: `2wikimultihopqa`
- Records: `1000`
- Pool K: `100`

## Overall Support Rank

- count: `2470`
- present_count: `2438`
- missing_count: `32`
- present_fraction: `0.9870445344129555`
- median: `2.0`
- p75: `3.0`
- p90: `5.0`
- mean: `3.9036095159967186`
- beyond_top20_fraction: `0.041700404858299595`
- within_top20_fraction: `0.9582995951417004`
- within_top50_fraction: `0.9728744939271255`
- within_top100_fraction: `0.9870445344129555`
- within_pool_fraction: `0.9870445344129555`

## Query Coverage

| K | All Support Within K | Any Support Beyond/Missing | No Support Within K |
|---:|---:|---:|---:|
| 20 | `0.9000` | `0.1000` | `0.0010` |
| 50 | `0.9340` | `0.0660` | `0.0010` |
| 100 | `0.9680` | `0.0320` | `0.0000` |
