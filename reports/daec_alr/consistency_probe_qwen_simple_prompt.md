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
| Original EM | 0.4050 |
| Original F1 | 0.5166 |
| Support Recall | 0.9413 |
| Support Complete | 0.8550 |

## Consistency Means

| Metric | All | Correct F1>=0.5 | Wrong F1<0.5 |
|---|---:|---:|---:|
| majority_fraction | 0.6550 | 0.8133 | 0.4800 |
| inverse_entropy | 0.5444 | 0.7470 | 0.3205 |
| inverse_distinct | 0.5900 | 0.7881 | 0.3711 |
| mean_pairwise_f1 | 0.6446 | 0.7823 | 0.4923 |
| normalized_entropy | 0.4556 | 0.2530 | 0.6795 |
| distinct_answer_count | 2.6400 | 1.8476 | 3.5158 |

## AUC: Consistency Predicting Correctness

### label_em

- positives: `81`
- negatives: `119`

| Metric | AUC | 95% CI |
|---|---:|---:|
| majority_fraction | 0.7823 | [0.7207, 0.8400] |
| inverse_entropy | 0.7853 | [0.7235, 0.8433] |
| inverse_distinct | 0.7856 | [0.7249, 0.8438] |
| mean_pairwise_f1 | 0.7479 | [0.6794, 0.8128] |

### label_f1_positive

- positives: `142`
- negatives: `58`

| Metric | AUC | 95% CI |
|---|---:|---:|
| majority_fraction | 0.6064 | [0.5242, 0.6889] |
| inverse_entropy | 0.6106 | [0.5275, 0.6960] |
| inverse_distinct | 0.6112 | [0.5302, 0.6937] |
| mean_pairwise_f1 | 0.7203 | [0.6353, 0.8029] |

### label_f1_ge_0_5

- positives: `105`
- negatives: `95`

| Metric | AUC | 95% CI |
|---|---:|---:|
| majority_fraction | 0.8107 | [0.7501, 0.8667] |
| inverse_entropy | 0.8153 | [0.7555, 0.8716] |
| inverse_distinct | 0.8158 | [0.7575, 0.8729] |
| mean_pairwise_f1 | 0.7758 | [0.7084, 0.8362] |

## Decision Hint

- Primary metric: `majority_fraction` AUC vs `label_f1_ge_0_5` = `0.810677`
- Decision hint: `PASS_SIGNAL_PROBE`
