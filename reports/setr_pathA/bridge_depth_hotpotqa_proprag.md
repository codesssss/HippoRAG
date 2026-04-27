# Pool Support Depth Analysis

- Pool: `/mnt/nvme/code/HippoRAG/run_logs/proprag_pool_exports_full1000_20260424/hotpotqa_pool100.json`
- Dataset: `hotpotqa`
- Records: `1000`
- Pool K: `100`

## Overall Support Rank

- count: `2000`
- present_count: `1998`
- missing_count: `2`
- present_fraction: `0.999`
- median: `2.0`
- p75: `2.0`
- p90: `4.0`
- mean: `2.3168168168168166`
- beyond_top20_fraction: `0.0075`
- within_top20_fraction: `0.9925`
- within_top50_fraction: `0.9965`
- within_top100_fraction: `0.999`
- within_pool_fraction: `0.999`

## Query Coverage

| K | All Support Within K | Any Support Beyond/Missing | No Support Within K |
|---:|---:|---:|---:|
| 20 | `0.9860` | `0.0140` | `0.0010` |
| 50 | `0.9940` | `0.0060` | `0.0010` |
| 100 | `0.9980` | `0.0020` | `0.0000` |
