# V4 GPT-4o-mini Reader Audit

## Protocol

| Item | Value |
| --- | --- |
| Dataset | 2WikiMultihopQA |
| Limit | 100 |
| Retrieval report | `run_logs/evidence_transition_graphragv4_branch_balanced_gpt4omini_3x100_20260511/2wikimultihopqa/reports/2wikimultihopqa_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json` |
| Retrieval top5 | Fixed |
| Context | Full passage |
| Metric | unchanged EM/F1 |
| Gold usage | none |

## Results

| Reader | Max tokens | R@5 | All-gold@5 | EM | F1 | Long predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3-8b-train` | 64 | 0.9400 | 0.8500 | 0.5100 | 0.6068 | 6 |
| `gpt-4o-mini` | 64 | 0.9400 | 0.8500 | 0.4000 | 0.5202 | 23 |
| `gpt-4o-mini` | 128 | 0.9400 | 0.8500 | 0.6200 | 0.7011 | 1 |

## Buckets

| Reader | Max tokens | Exact all-gold | Wrong all-gold | Exact partial-gold | Wrong partial-gold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen3-8b-train` | 64 | 48 | 37 | 3 | 12 |
| `gpt-4o-mini` | 64 | 37 | 48 | 3 | 12 |
| `gpt-4o-mini` | 128 | 56 | 29 | 6 | 9 |

## Interpretation

| Observation | Meaning |
| --- | --- |
| Retrieval metrics are identical | The experiment isolates reader behavior. |
| `gpt-4o-mini` at 64 tokens is worse | The model often writes longer reasoning and fails to emit a clean short answer under the small budget. |
| `gpt-4o-mini` at 128 tokens is best | The 2Wiki low EM/F1 is largely a reader/protocol bottleneck, not a retrieval bottleneck. |
| Exact all-gold successes rise from 48 to 56 | Better reader recovers cases where all evidence was already in top5. |

## Decision

Use this result as diagnosis, not as the default method score unless all
baselines are rerun under the same `gpt-4o-mini` reader and token budget.
