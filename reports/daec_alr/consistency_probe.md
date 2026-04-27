# DAEC Reader Consistency Probe

## Purpose

Test whether reader answer consistency under DAEC evidence perturbations separates correct from wrong original DAEC contexts.

This is a signal probe only. It does not implement active retrieval or edit admission.

## Configuration

- DAEC report: `run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json`
- Pool JSON: `run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json`
- Reader mode: `openai_compatible_chat`
- Rows: `200`
- Variants: `original, swap01, reverse, rotate_left, drop_last`

## Base Reader Result

| Metric | Value |
|---|---:|
| Original EM | 0.4900 |
| Original F1 | 0.5882 |
| Support Recall | 0.9413 |
| Support Complete | 0.8550 |

## Consistency Means

| Metric | All | Correct F1>=0.5 | Wrong F1<0.5 |
|---|---:|---:|---:|
| majority_fraction | 0.8030 | 0.8992 | 0.6617 |
| inverse_entropy | 0.7377 | 0.8649 | 0.5507 |
| inverse_distinct | 0.7800 | 0.8950 | 0.6111 |
| mean_pairwise_f1 | 0.7766 | 0.8658 | 0.6456 |
| normalized_entropy | 0.2623 | 0.1351 | 0.4493 |
| distinct_answer_count | 1.8800 | 1.4202 | 2.5556 |

## AUC: Consistency Predicting Correctness

### label_em

- positives: `98`
- negatives: `102`

| Metric | AUC | 95% CI |
|---|---:|---:|
| majority_fraction | 0.7683 | [0.7056, 0.8232] |
| inverse_entropy | 0.7700 | [0.7063, 0.8264] |
| inverse_distinct | 0.7648 | [0.7028, 0.8235] |
| mean_pairwise_f1 | 0.7447 | [0.6778, 0.8045] |

### label_f1_positive

- positives: `145`
- negatives: `55`

| Metric | AUC | 95% CI |
|---|---:|---:|
| majority_fraction | 0.5998 | [0.5168, 0.6767] |
| inverse_entropy | 0.5994 | [0.5155, 0.6767] |
| inverse_distinct | 0.6003 | [0.5162, 0.6769] |
| mean_pairwise_f1 | 0.6757 | [0.5830, 0.7551] |

### label_f1_ge_0_5

- positives: `119`
- negatives: `81`

| Metric | AUC | 95% CI |
|---|---:|---:|
| majority_fraction | 0.7572 | [0.6875, 0.8202] |
| inverse_entropy | 0.7598 | [0.6902, 0.8226] |
| inverse_distinct | 0.7599 | [0.6898, 0.8218] |
| mean_pairwise_f1 | 0.7317 | [0.6601, 0.7983] |

## Decision Hint

- Primary metric: `majority_fraction` AUC vs `label_f1_ge_0_5` = `0.757184`
- Decision hint: `PASS_SIGNAL_PROBE`
