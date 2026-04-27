# Pool Support Depth Analysis

- Pool: `/mnt/nvme/code/HippoRAG/run_logs/proprag_pool_exports_full1000_20260424/musique_pool100.json`
- Dataset: `musique`
- Records: `1000`
- Pool K: `100`

## Overall Support Rank

- count: `2648`
- present_count: `2542`
- missing_count: `106`
- present_fraction: `0.9599697885196374`
- median: `2.0`
- p75: `7.0`
- p90: `20.0`
- mean: `8.124704956726987`
- beyond_top20_fraction: `0.13104229607250756`
- within_top20_fraction: `0.8689577039274925`
- within_top50_fraction: `0.922583081570997`
- within_top100_fraction: `0.9599697885196374`
- within_pool_fraction: `0.9599697885196374`

## Query Coverage

| K | All Support Within K | Any Support Beyond/Missing | No Support Within K |
|---:|---:|---:|---:|
| 20 | `0.7360` | `0.2640` | `0.0020` |
| 50 | `0.8450` | `0.1550` | `0.0000` |
| 100 | `0.9170` | `0.0830` | `0.0000` |

## Hop Support Rank

- hop 1: median=`1.0` p75=`2.0` beyond_top20=`0.0350`
- hop 2: median=`3.0` p75=`7.25` beyond_top20=`0.1190`
- hop 3: median=`4.0` p75=`14.0` beyond_top20=`0.2241`
- hop 4: median=`6.0` p75=`20.0` beyond_top20=`0.3795`
