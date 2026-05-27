# V4 Graph Evidence Reader Context Audit

## Protocol

| Item | Value |
| --- | --- |
| Dataset | 2WikiMultihopQA |
| Limit | 100 |
| Retrieval report | `run_logs/evidence_transition_graphragv4_branch_balanced_gpt4omini_3x100_20260511/2wikimultihopqa/reports/2wikimultihopqa_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json` |
| Retrieval top5 | Fixed across all rows |
| Reader | `qwen3-8b-train` |
| Reader endpoint | `http://localhost:8041/v1` |
| Thinking | disabled |
| Metric | unchanged EM/F1 |
| Gold usage | none in context construction |

## Results

| Context | R@5 | All-gold@5 | EM | F1 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Full passage | 0.9400 | 0.8500 | 0.5100 | 0.6068 | keep |
| Graph facts before passage | 0.9400 | 0.8500 | 0.3900 | 0.5014 | reject |
| Passage before graph facts | 0.9400 | 0.8500 | 0.3600 | 0.4777 | reject |

## Diagnosis

| Observation | Interpretation |
| --- | --- |
| Retrieval metrics are identical across modes | The change only affects reader behavior. |
| Full passage has 48 exact all-gold successes; graph facts before passage has 37 | OpenIE fact exposure causes reader regressions even when all gold documents are present. |
| Passage-before-facts is even worse than facts-before-passage | The issue is not just fact ordering; adding all OpenIE facts increases reader confusion. |
| Many losses are truncated entities or verbose reasoning | The reader stops copying the concise answer span and starts explaining comparisons. |

## Conclusion

All-OpenIE graph evidence should not be adopted as the V4 reader context.  It is
fairly evaluated but empirically harmful.  The next reader-side fix should not
add more fact text; it should either keep the original full-passage context or
use a stricter, separately evaluated reader protocol shared by all compared
methods.
