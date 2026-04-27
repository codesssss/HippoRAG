# Pool Support Depth Analysis

- Pool: `/mnt/nvme/code/HippoRAG/run_logs/dense_pool_exports_full1000_20260424/hotpotqa_dense_pool100.json`
- Dataset: `hotpotqa`
- Records: `1000`
- Pool K: `100`

## Overall Support Rank

- count: `2000`
- present_count: `1985`
- missing_count: `15`
- present_fraction: `0.9925`
- median: `2.0`
- p75: `2.0`
- p90: `4.0`
- mean: `2.557682619647355`
- beyond_top20_fraction: `0.017`
- within_top20_fraction: `0.983`
- within_top50_fraction: `0.9895`
- within_top100_fraction: `0.9925`
- within_pool_fraction: `0.9925`

## Query Coverage

| K | All Support Within K | Any Support Beyond/Missing | No Support Within K |
|---:|---:|---:|---:|
| 20 | `0.9680` | `0.0320` | `0.0020` |
| 50 | `0.9790` | `0.0210` | `0.0000` |
| 100 | `0.9850` | `0.0150` | `0.0000` |
