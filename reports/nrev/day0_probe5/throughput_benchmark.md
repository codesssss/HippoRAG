# NREV Day-0 Sanity

## Configuration

- rows: `5`
- scorer: `qwen_prompt_logprob`
- model: `qwen3-8b-train`
- closed-book correct/wrong: `0` / `5`
- decision: `MARGINAL_REDESIGN_T_MINUS_ONCE`

## Overall Metrics

| Score | AUC | 95% CI | Paired Win Rate |
|---|---:|---:|---:|
| `l_plus` | 0.8000 | [0.6800, 1.0000] | 1.0000 |
| `rev` | 0.6000 | [0.2000, 0.8000] | 0.4000 |
| `nrev_no_alt` | 0.6000 | [0.2000, 0.8000] | 0.4000 |
| `nrev_no_l0` | 0.8800 | [0.6000, 1.0000] | 0.8000 |
| `nrev_full` | 0.8800 | [0.6000, 1.0000] | 0.8000 |

## Closed-Book Stratification

### closed_book_correct

| Score | AUC | 95% CI | Paired Win Rate | N |
|---|---:|---:|---:|---:|
| `l_plus` | n/a | n/a | 0.0000 | 0 |
| `rev` | n/a | n/a | 0.0000 | 0 |
| `nrev_no_alt` | n/a | n/a | 0.0000 | 0 |
| `nrev_no_l0` | n/a | n/a | 0.0000 | 0 |
| `nrev_full` | n/a | n/a | 0.0000 | 0 |

### closed_book_wrong

| Score | AUC | 95% CI | Paired Win Rate | N |
|---|---:|---:|---:|---:|
| `l_plus` | 0.8000 | [0.6800, 1.0000] | 1.0000 | 5 |
| `rev` | 0.6000 | [0.2000, 0.8000] | 0.4000 | 5 |
| `nrev_no_alt` | 0.6000 | [0.2000, 0.8000] | 0.4000 | 5 |
| `nrev_no_l0` | 0.8800 | [0.6000, 1.0000] | 0.8000 | 5 |
| `nrev_full` | 0.8800 | [0.6000, 1.0000] | 0.8000 | 5 |

## Throughput

- elapsed seconds: `10.4799`
- seconds/query: `2.096`
- estimated 100-query seconds: `209.5981`
- estimated 1000-query seconds: `2095.9807`

## Notes

- Day-0 uses only `(a_gold, S_gold)` and `(a_wrong, S_wrong)` pairs.
- `S_gold` is gold support padded to k=5 with top-ranked pool docs.
- `S_wrong` is built from wrong-answer-containing pool docs plus high-ranked distractors when source evidence is unavailable.
- `l_alt` is type-compatible: incompatible alternative answer types are excluded from the null.
- Destructive perturbations use singleton functional-cell deletion/replacement in this Day-0 implementation.
