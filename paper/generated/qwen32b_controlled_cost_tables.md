# Qwen3-32B Controlled Cost Tables

## Table A: Offline Structuring Tokens

Unit: million tokens.

| Method | HotpotQA | 2Wiki | MuSiQue | Avg |
|---|---:|---:|---:|---:|
| EvLink | 13.17 | 7.75 | 14.62 | 11.85 |
| HippoRAG2 | 14.90 | 9.16 | 16.39 | 13.49 |
| PropRAG | 20.85 | 12.17 | 23.49 | 18.84 |
| NeocorRAG | 14.27 | 8.46 | 15.78 | 12.84 |

## Table B: Online Auxiliary Tokens Per Query

Unit: tokens/query. GPT-4o-mini reader excluded.

| Method | HotpotQA | 2Wiki | MuSiQue | Avg |
|---|---:|---:|---:|---:|
| EvLink | 1,403 | 1,456 | 1,682 | 1,514 |
| HippoRAG2 | 639 | 635 | 625 | 633 |
| PropRAG | 0 | 0 | 0 | 0 |
| NeocorRAG | 7,194 | 6,831 | 6,789 | 6,938 |

## Table C: Average Total Auxiliary Generative-Token Cost

Unit: million tokens. Total is offline structuring plus online auxiliary tokens for 1,000 queries.

| Method | Offline Avg M | Online Avg M for 1000q | Total Avg M |
|---|---:|---:|---:|
| EvLink | 11.85 | 1.51 | 13.36 |
| HippoRAG2 | 13.49 | 0.63 | 14.12 |
| PropRAG | 18.84 | 0.00 | 18.84 |
| NeocorRAG | 12.84 | 6.94 | 19.78 |
