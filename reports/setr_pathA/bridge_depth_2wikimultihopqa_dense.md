# Pool Support Depth Analysis

- Pool: `/mnt/nvme/code/HippoRAG/run_logs/dense_pool_exports_full1000_20260424/2wikimultihopqa_dense_pool100.json`
- Dataset: `2wikimultihopqa`
- Records: `1000`
- Pool K: `100`

## Overall Support Rank

- count: `2470`
- present_count: `2099`
- missing_count: `371`
- present_fraction: `0.8497975708502025`
- median: `2.0`
- p75: `3.0`
- p90: `19.200000000000045`
- mean: `7.411624583134826`
- beyond_top20_fraction: `0.23441295546558705`
- within_top20_fraction: `0.765587044534413`
- within_top50_fraction: `0.8117408906882592`
- within_top100_fraction: `0.8497975708502025`
- within_pool_fraction: `0.8497975708502025`

## Query Coverage

| K | All Support Within K | Any Support Beyond/Missing | No Support Within K |
|---:|---:|---:|---:|
| 20 | `0.5580` | `0.4420` | `0.0000` |
| 50 | `0.6340` | `0.3660` | `0.0000` |
| 100 | `0.7060` | `0.2940` | `0.0000` |
