# NREV Day-0 Sanity

## Configuration

- rows: `30`
- scorer: `qwen_prompt_logprob`
- model: `qwen3-8b-train`
- closed-book correct/wrong: `8` / `22`
- decision: `MARGINAL_REDESIGN_T_MINUS_ONCE`

## Overall Metrics

| Score | AUC | 95% CI | Paired Win Rate |
|---|---:|---:|---:|
| `l_plus` | 0.6656 | [0.5400, 0.7944] | 0.6667 |
| `rev` | 0.5278 | [0.4078, 0.6356] | 0.4667 |
| `nrev_no_alt` | 0.5222 | [0.3867, 0.6356] | 0.5000 |
| `nrev_no_l0` | 0.6600 | [0.4911, 0.8167] | 0.5667 |
| `nrev_full` | 0.6644 | [0.4744, 0.8244] | 0.6000 |

## Closed-Book Stratification

### closed_book_correct

| Score | AUC | 95% CI | Paired Win Rate | N |
|---|---:|---:|---:|---:|
| `l_plus` | 0.5781 | [0.3438, 0.7656] | 0.7500 | 8 |
| `rev` | 0.6250 | [0.2500, 0.8906] | 0.6250 | 8 |
| `nrev_no_alt` | 0.5156 | [0.1250, 0.8438] | 0.5000 | 8 |
| `nrev_no_l0` | 0.6406 | [0.2812, 0.9219] | 0.6250 | 8 |
| `nrev_full` | 0.6094 | [0.2500, 0.8906] | 0.6250 | 8 |

### closed_book_wrong

| Score | AUC | 95% CI | Paired Win Rate | N |
|---|---:|---:|---:|---:|
| `l_plus` | 0.6942 | [0.5165, 0.8471] | 0.6364 | 22 |
| `rev` | 0.5021 | [0.3781, 0.6343] | 0.4091 | 22 |
| `nrev_no_alt` | 0.5207 | [0.3843, 0.6694] | 0.5000 | 22 |
| `nrev_no_l0` | 0.6612 | [0.4504, 0.8161] | 0.5455 | 22 |
| `nrev_full` | 0.6777 | [0.4587, 0.8388] | 0.5909 | 22 |

## Diagnostics

### Paired Outcomes

| Score | Gold Wins | Ties | Wrong Wins |
|---|---:|---:|---:|
| `l_plus` | 20 | 0 | 10 |
| `rev` | 14 | 0 | 16 |
| `nrev_no_alt` | 15 | 0 | 15 |
| `nrev_no_l0` | 17 | 0 | 13 |
| `nrev_full` | 18 | 0 | 12 |

### Dominant Null Counts

| Side | l_minus | l0 | l_alt | none |
|---|---:|---:|---:|---:|
| `gold` | 16 | 7 | 7 | 0 |
| `wrong` | 11 | 3 | 16 | 0 |

### Component Means

| Side | l_plus | l_minus | l0 | l_alt |
|---|---:|---:|---:|---:|
| `gold` | -2.46226 | -2.766891 | -3.953159 | -5.196856 |
| `wrong` | -4.577782 | -4.665497 | -6.127359 | -2.122762 |

## Throughput

- elapsed seconds: `64.0642`
- seconds/query: `2.1355`
- estimated 100-query seconds: `213.5472`
- estimated 1000-query seconds: `2135.4718`

## Notes

- Day-0 uses only `(a_gold, S_gold)` and `(a_wrong, S_wrong)` pairs.
- `S_gold` is gold support padded to k=5 with top-ranked pool docs.
- `S_wrong` is built from wrong-answer-containing pool docs plus high-ranked distractors when source evidence is unavailable.
- `l_alt` is type-compatible: incompatible alternative answer types are excluded from the null.
- Closed-book stratification uses a separate own-knowledge prompt, not the evidence-only RAG reader prompt.
- Destructive perturbations use singleton functional-cell deletion/replacement in this Day-0 implementation.
