# V4 Qwen8B Reader Token-Budget Audit

## Protocol

| Item | Value |
| --- | --- |
| Dataset | 2WikiMultihopQA |
| Limit | 100 |
| Retrieval report | `run_logs/evidence_transition_graphragv4_branch_balanced_gpt4omini_3x100_20260511/2wikimultihopqa/reports/2wikimultihopqa_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json` |
| Retrieval top5 | Fixed |
| Context | Full passage |
| Reader | `qwen3-8b-train` |
| Thinking | disabled |
| Metric | unchanged EM/F1 |
| Gold usage | none |

## Results

| Reader | Max tokens | R@5 | All-gold@5 | EM | F1 | Long predictions | Empty predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3-8b-train` | 64 | 0.9400 | 0.8500 | 0.5100 | 0.6068 | 6 | 1 |
| `qwen3-8b-train` | 128 | 0.9400 | 0.8500 | 0.5800 | 0.6462 | 1 | 0 |
| `qwen3-8b-train` | 400 | 0.9400 | 0.8500 | 0.5800 | 0.6462 | 1 | 0 |
| `gpt-4o-mini` | 128 | 0.9400 | 0.8500 | 0.6200 | 0.7011 | 1 | 0 |

## Buckets

| Reader | Max tokens | Exact all-gold | Wrong all-gold | Exact partial-gold | Wrong partial-gold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen3-8b-train` | 64 | 48 | 37 | 3 | 12 |
| `qwen3-8b-train` | 128 | 55 | 30 | 3 | 12 |
| `qwen3-8b-train` | 400 | 55 | 30 | 3 | 12 |
| `gpt-4o-mini` | 128 | 56 | 29 | 6 | 9 |

## Interpretation

| Observation | Meaning |
| --- | --- |
| Retrieval metrics are identical | The experiment isolates reader max-token budget. |
| Qwen8B 128 has 7 EM gains and 0 EM losses over Qwen8B 64 | The previous 64-token setting was under-budgeted for 2Wiki comparison answers. |
| Qwen8B 400 exactly matches Qwen8B 128 | A 400-token budget is safe for this reader and does not make the model produce longer answers under the current prompt. |
| Long predictions drop from 6 to 1 and empty predictions drop from 1 to 0 | The larger budget fixes truncation/unfinished-answer failures without changing retrieval. |
| GPT-4o-mini 128 remains higher | Reader quality still matters after token budget is fixed. |

## Decision

Use `max_new_tokens=400` as the fair default reader protocol only if all compared
retrievers are rerun under the same reader, prompt, context, and metric.
