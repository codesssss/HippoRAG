# Reviewer Baseline Cost Table - 2026-05-06

Scope: PropRAG full1000 reviewer baselines. Costs are selector-side only; final reader QA cost is shared by all rows and omitted.

Important caveats:

- DAEC token/latency fields measure LLM binding extraction only. Logical calls add one decomposition call/query, but decomposition token usage is not instrumented in these artifacts.
- DAEC-selective cold-equivalent cost equals DAEC because the title-uniqueness gate runs after binding extraction. The fresh run recorded zero binding API calls because it reused the binding cache.
- Historical SetR-style rows predate token instrumentation, so exact token usage is unavailable; token counts are estimated as reconstructed chars/4 and character counts are also shown.
- IRCoT-style local stores LLM call count, retrieval rounds, and wall-clock latency, but not token usage.

## Main Cost Rows

| Dataset | Method | EM | F1 | R@5 | Logical LLM calls/q | Measured/equiv API calls/q | Extra retrieval calls/q | Prompt tok/q | Completion tok/q | Prompt chars/q | Latency/q |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | Top5 / original PropRAG pool order | 0.575 | 0.6457 | 0.9028 | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | n/a |
| 2Wiki | DAEC-LLM+CTL | 0.642 | 0.7118 | 0.9410 | 8.44 | 7.44 | 0.00 | 991.4 | 64.3 | n/a | 0.660 |
| 2Wiki | DAEC-selective titleuniq | 0.642 | 0.7118 | 0.9410 | 8.44 | 7.44 | 0.00 | 991.4 | 64.3 | n/a | 0.660 |
| 2Wiki | SetR-style k20 doc768 | 0.624 | 0.6936 | 0.9423 | 1.00 | 1.00 | 0.00 | 1908.9 | 12.4 | 7635.5 | 0.258 |
| 2Wiki | IRCoT-style local | 0.564 | 0.6293 | 0.8762 | 3.00 | 3.00 | 3.00 | n/a | n/a | n/a | 4.169 |
| 2Wiki | LLM-direct title | 0.519 | 0.5697 | 0.8160 | 1.00 | 1.00 | 0.00 | 1346.6 | 25.7 | n/a | 0.291 |
| 2Wiki | LLM-direct snippet128 | 0.588 | 0.6535 | 0.8882 | 1.00 | 1.00 | 0.00 | 4353.5 | 25.5 | n/a | 0.395 |
| HotpotQA | Top5 / original PropRAG pool order | 0.595 | 0.7227 | 0.9500 | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | n/a |
| HotpotQA | DAEC-LLM+CTL | 0.620 | 0.7473 | 0.9605 | 7.32 | 6.32 | 0.00 | 1129.1 | 71.7 | n/a | 0.711 |
| HotpotQA | DAEC-selective titleuniq | 0.620 | 0.7473 | 0.9605 | 7.32 | 6.32 | 0.00 | 1129.1 | 71.7 | n/a | 0.711 |
| HotpotQA | SetR-style k20 doc768 | 0.629 | 0.7552 | 0.9730 | 1.00 | 1.00 | 0.00 | 2790.1 | 12.6 | 11160.2 | 0.280 |
| HotpotQA | IRCoT-style local | 0.590 | 0.7079 | 0.9150 | 3.00 | 3.00 | 3.00 | n/a | n/a | n/a | 6.635 |
| HotpotQA | LLM-direct title | 0.537 | 0.6478 | 0.8255 | 1.00 | 1.00 | 0.00 | 1334.2 | 25.7 | n/a | 0.287 |
| HotpotQA | LLM-direct snippet128 | 0.545 | 0.6544 | 0.8130 | 1.00 | 1.00 | 0.00 | 4898.2 | 25.6 | n/a | 0.416 |
| MuSiQue | Top5 / original PropRAG pool order | 0.330 | 0.4266 | 0.7131 | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | n/a |
| MuSiQue | DAEC-LLM+CTL | 0.337 | 0.4359 | 0.7269 | 7.72 | 6.72 | 0.00 | 1247.8 | 75.9 | n/a | 0.755 |
| MuSiQue | DAEC-selective titleuniq | 0.353 | 0.4548 | 0.7469 | 7.72 | 6.72 | 0.00 | 1247.8 | 75.9 | n/a | 0.755 |
| MuSiQue | SetR-style k20 doc768 | 0.377 | 0.4761 | 0.7547 | 1.00 | 1.00 | 0.00 | 2829.5 | 13.6 | 11318.0 | 0.307 |
| MuSiQue | IRCoT-style local | 0.327 | 0.4254 | 0.6698 | 3.00 | 3.00 | 3.00 | n/a | n/a | n/a | 6.969 |
| MuSiQue | LLM-direct title | 0.272 | 0.3582 | 0.5562 | 1.00 | 1.00 | 0.00 | 1275.4 | 26.4 | n/a | 0.278 |
| MuSiQue | LLM-direct snippet128 | 0.294 | 0.3850 | 0.5753 | 1.00 | 1.00 | 0.00 | 4727.3 | 26.2 | n/a | 0.412 |

## Cost Sources

| Dataset | Method | Source / Caveat |
|---|---|---|
| 2Wiki | Top5 / original PropRAG pool order | No selector. |
| 2Wiki | DAEC-LLM+CTL | Measured binding extraction only; logical calls add one decomposition call/query whose token usage is not instrumented. |
| 2Wiki | DAEC-selective titleuniq | Table reports cold-equivalent cost, equal to DAEC-LLM+CTL. The fresh run recorded 0.00 binding API calls/query because it used the binding cache. |
| 2Wiki | SetR-style k20 doc768 | Historical run predates token instrumentation; prompt/completion chars reconstructed and token counts estimated as chars/4. |
| 2Wiki | IRCoT-style local | Local IRCoT-style JSON stores calls, retrieval rounds, and wall-clock latency, not token usage. |
| 2Wiki | LLM-direct title | Measured selector usage; parse_success=1.000. |
| 2Wiki | LLM-direct snippet128 | Measured selector usage; parse_success=1.000. |
| HotpotQA | Top5 / original PropRAG pool order | No selector. |
| HotpotQA | DAEC-LLM+CTL | Measured binding extraction only; logical calls add one decomposition call/query whose token usage is not instrumented. |
| HotpotQA | DAEC-selective titleuniq | Table reports cold-equivalent cost, equal to DAEC-LLM+CTL. The fresh run recorded 0.00 binding API calls/query because it used the binding cache. |
| HotpotQA | SetR-style k20 doc768 | Historical run predates token instrumentation; prompt/completion chars reconstructed and token counts estimated as chars/4. |
| HotpotQA | IRCoT-style local | Local IRCoT-style JSON stores calls, retrieval rounds, and wall-clock latency, not token usage. |
| HotpotQA | LLM-direct title | Measured selector usage; parse_success=1.000. |
| HotpotQA | LLM-direct snippet128 | Measured selector usage; parse_success=1.000. |
| MuSiQue | Top5 / original PropRAG pool order | No selector. |
| MuSiQue | DAEC-LLM+CTL | Measured binding extraction only; logical calls add one decomposition call/query whose token usage is not instrumented. |
| MuSiQue | DAEC-selective titleuniq | Table reports cold-equivalent cost, equal to DAEC-LLM+CTL. The fresh run recorded 0.00 binding API calls/query because it used the binding cache. |
| MuSiQue | SetR-style k20 doc768 | Historical run predates token instrumentation; prompt/completion chars reconstructed and token counts estimated as chars/4. |
| MuSiQue | IRCoT-style local | Local IRCoT-style JSON stores calls, retrieval rounds, and wall-clock latency, not token usage. |
| MuSiQue | LLM-direct title | Measured selector usage; parse_success=1.000. |
| MuSiQue | LLM-direct snippet128 | Measured selector usage; parse_success=1.000. |

## Interpretation

- DAEC is not lower-call than one-shot LLM selectors: cold logical cost is about one decomposition call plus 6-7 binding calls per query.
- DAEC can still be prompt-token-efficient relative to snippet-heavy selectors: measured binding prompt tokens are about 1.1-1.3k/query, while LLM-direct snippet128 uses about 4.4-4.9k prompt tokens/query.
- SetR-style k20 has one LLM call/query and estimated prompt tokens around 1.9-2.8k/query under the chars/4 heuristic; this is an estimate, not recorded API usage.
- DAEC-selective should be sold as robustness, not cost reduction.
