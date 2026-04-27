# NREV Day-0 Sanity

## Configuration

- rows: `100`
- scorer: `qwen_prompt_logprob`
- model: `qwen3-8b-train`
- closed-book correct/wrong: `2` / `98`
- decision: `STOP_NREV_DAY0_FAIL`

## Overall Metrics

| Score | AUC | 95% CI | Paired Win Rate |
|---|---:|---:|---:|
| `l_plus` | 0.7085 | [0.6273, 0.7790] | 0.7000 |
| `rev` | 0.4266 | [0.3555, 0.4960] | 0.4000 |
| `nrev_no_alt` | 0.4221 | [0.3417, 0.4972] | 0.4100 |
| `nrev_no_l0` | 0.6146 | [0.5241, 0.6969] | 0.5700 |
| `nrev_full` | 0.6042 | [0.5149, 0.6889] | 0.5600 |

## Closed-Book Stratification

### closed_book_correct

| Score | AUC | 95% CI | Paired Win Rate | N |
|---|---:|---:|---:|---:|
| `l_plus` | 0.7500 | [0.0000, 1.0000] | 0.5000 | 2 |
| `rev` | 0.7500 | [0.0000, 1.0000] | 0.5000 | 2 |
| `nrev_no_alt` | 0.2500 | [0.0000, 1.0000] | 0.5000 | 2 |
| `nrev_no_l0` | 0.7500 | [0.0000, 1.0000] | 0.5000 | 2 |
| `nrev_full` | 0.5000 | [0.0000, 1.0000] | 0.5000 | 2 |

### closed_book_wrong

| Score | AUC | 95% CI | Paired Win Rate | N |
|---|---:|---:|---:|---:|
| `l_plus` | 0.7076 | [0.6288, 0.7774] | 0.7041 | 98 |
| `rev` | 0.4233 | [0.3507, 0.4970] | 0.3980 | 98 |
| `nrev_no_alt` | 0.4222 | [0.3430, 0.4984] | 0.4082 | 98 |
| `nrev_no_l0` | 0.6156 | [0.5197, 0.7032] | 0.5714 | 98 |
| `nrev_full` | 0.6057 | [0.5068, 0.6968] | 0.5612 | 98 |

## Diagnostics

### Paired Outcomes

| Score | Gold Wins | Ties | Wrong Wins |
|---|---:|---:|---:|
| `l_plus` | 70 | 0 | 30 |
| `rev` | 40 | 0 | 60 |
| `nrev_no_alt` | 41 | 0 | 59 |
| `nrev_no_l0` | 57 | 0 | 43 |
| `nrev_full` | 56 | 0 | 44 |

### Dominant Null Counts

| Side | l_minus | l0 | l_alt | none |
|---|---:|---:|---:|---:|
| `gold` | 51 | 31 | 18 | 0 |
| `wrong` | 46 | 6 | 48 | 0 |

### Component Means

| Side | l_plus | l_minus | l0 | l_alt |
|---|---:|---:|---:|---:|
| `gold` | -2.220946 | -2.308659 | -3.715661 | -5.235581 |
| `wrong` | -4.017205 | -4.165664 | -6.147633 | -2.436455 |

## Throughput

- elapsed seconds: `44.3521`
- seconds/query: `0.4435`
- estimated 100-query seconds: `44.3521`
- estimated 1000-query seconds: `443.5209`

## Notes

- Day-0 uses only `(a_gold, S_gold)` and `(a_wrong, S_wrong)` pairs.
- `S_gold` is gold support padded to k=5 with top-ranked pool docs.
- `S_wrong` is built from wrong-answer-containing pool docs plus high-ranked distractors when source evidence is unavailable.
- `l_alt` is type-compatible: incompatible alternative answer types are excluded from the null.
- Closed-book stratification uses a separate own-knowledge prompt, not the evidence-only RAG reader prompt.
- Destructive perturbations use singleton functional-cell deletion/replacement in this Day-0 implementation.
