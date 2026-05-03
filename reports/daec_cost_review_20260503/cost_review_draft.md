# DAEC / SetR Cost Review Draft - 2026-05-03

This draft uses completed full1000 artifacts available at generation time. SetR is still running; append new rows as reports arrive.

## DAEC-LLM + CTL Selector Cost

| Pool | Dataset | EM | F1 | R@5 | Calls/q | Tokens/q | Prompt/q | Completion/q | Selector latency/q | Empty responses/q |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 2wikimultihopqa | 0.516 | 0.5798 | 0.8167 | 7.43 | 1028.0 | 963.7 | 64.3 | 0.681s | 5.16 |
| HippoRAG | 2wikimultihopqa | 0.572 | 0.6431 | 0.9002 | 7.43 | 1029.0 | 964.8 | 64.2 | 0.651s | 5.17 |
| Dense | hotpotqa | 0.624 | 0.7483 | 0.9535 | 6.32 | 1186.2 | 1114.7 | 71.5 | 0.710s | 3.66 |
| HippoRAG | hotpotqa | 0.620 | 0.7403 | 0.9510 | 6.32 | 1187.7 | 1116.1 | 71.6 | 0.708s | 3.65 |
| Dense | musique | 0.304 | 0.3918 | 0.6863 | 6.70 | 1294.2 | 1218.8 | 75.4 | 0.761s | 3.73 |
| HippoRAG | musique | 0.317 | 0.4142 | 0.6957 | 6.69 | 1298.5 | 1223.3 | 75.3 | 0.736s | 3.73 |
| PropRAG | 2wikimultihopqa | 0.642 | 0.7118 | 0.9410 | 7.44 | 1055.7 | 991.4 | 64.3 | 0.660s | 5.17 |
| PropRAG | hotpotqa | 0.620 | 0.7473 | 0.9605 | 6.32 | 1200.9 | 1129.1 | 71.7 | 0.711s | 3.64 |
| PropRAG | musique | 0.337 | 0.4359 | 0.7269 | 6.72 | 1323.7 | 1247.8 | 75.9 | 0.754s | 3.75 |

- Overall: avg calls/q `6.82`, avg tokens/q `1178.2`, avg selector latency/q `0.708s`, avg empty responses/q `4.18`.
- Dense: avg calls/q `6.82`, avg tokens/q `1169.4`, avg selector latency/q `0.717s`, avg empty responses/q `4.18`.
- HippoRAG: avg calls/q `6.81`, avg tokens/q `1171.7`, avg selector latency/q `0.698s`, avg empty responses/q `4.18`.
- PropRAG: avg calls/q `6.83`, avg tokens/q `1193.4`, avg selector latency/q `0.709s`, avg empty responses/q `4.19`.

## Completed SetR Rows

SetR selection artifacts currently do not store API token usage. Calls/query are exact for direct SetR variants (`1` selector call/query); token/query should be instrumented or estimated from prompt serialization.

| Pool | Dataset | Variant | EM | F1 | R@5 | Selector calls/q |
|---|---|---|---:|---:|---:|---:|
| Dense | 2wikimultihopqa | setr_k100_doc160 | 0.517 | 0.5697 | 0.7935 | 1 |
| Dense | 2wikimultihopqa | setr_k20_doc768 | 0.514 | 0.5655 | 0.7798 | 1 |
| Dense | hotpotqa | setr_k100_doc160 | 0.604 | 0.7217 | 0.9060 | 1 |
| Dense | hotpotqa | setr_k20_doc768 | 0.627 | 0.7470 | 0.9615 | 1 |
| Dense | musique | setr_k20_doc768 | 0.310 | 0.4108 | 0.7045 | 1 |

## Completed SetR Selection Durations From Launcher Log

| Label | Seconds | Seconds/query |
|---|---:|---:|
| `select_musique_dense_setr_k20_doc768` | 18.0 | 0.018 |
| `select_2wikimultihopqa_dense_setr_k100_doc160` | 487.0 | 0.487 |
| `select_hotpotqa_dense_setr_k100_doc160` | 604.0 | 0.604 |
| `select_musique_dense_setr_k100_doc160` | 672.0 | 0.672 |

## Immediate Notes

- DAEC-LLM + CTL averages about `6.82` selector LLM calls/query and `1178` selector tokens/query across 9 pool x dataset settings.
- Empty extractor responses are frequent, about `4.18` per query on average; this supports adding an empty-response audit rather than hand-waving it away.
- Direct SetR gives a clean call-count baseline: `1` selector LLM call/query, but token usage is not yet captured by the adapter.
- For a fair cost table, add token accounting to SetR selection or estimate prompt tokens consistently from the serialized prompt/input.
