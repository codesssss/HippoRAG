# Pool Support Depth Analysis

- Pool: `/mnt/nvme/code/HippoRAG/run_logs/dense_pool_exports_full1000_20260424/musique_dense_pool100.json`
- Dataset: `musique`
- Records: `1000`
- Pool K: `100`

## Overall Support Rank

- count: `2648`
- present_count: `2380`
- missing_count: `268`
- present_fraction: `0.8987915407854985`
- median: `2.0`
- p75: `7.0`
- p90: `23.0`
- mean: `8.169327731092437`
- beyond_top20_fraction: `0.19977341389728095`
- within_top20_fraction: `0.800226586102719`
- within_top50_fraction: `0.868202416918429`
- within_top100_fraction: `0.8987915407854985`
- within_pool_fraction: `0.8987915407854985`

## Query Coverage

| K | All Support Within K | Any Support Beyond/Missing | No Support Within K |
|---:|---:|---:|---:|
| 20 | `0.5970` | `0.4030` | `0.0030` |
| 50 | `0.7070` | `0.2930` | `0.0000` |
| 100 | `0.7660` | `0.2340` | `0.0000` |

## Hop Support Rank

- hop 1: median=`1.0` p75=`2.0` beyond_top20=`0.0290`
- hop 2: median=`3.0` p75=`9.0` beyond_top20=`0.2820`
- hop 3: median=`4.0` p75=`11.0` beyond_top20=`0.2448`
- hop 4: median=`7.0` p75=`19.0` beyond_top20=`0.4096`
