# DAEC Reader Consistency Probe

## Purpose

Test whether reader answer consistency under DAEC evidence perturbations separates correct from wrong original DAEC contexts.

This is a signal probe only. It does not implement active retrieval or edit admission.

## Configuration

- DAEC report: `run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json`
- Pool JSON: `run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json`
- Reader mode: `hf_seq2seq`
- Rows: `200`
- Variants: `original, swap01, reverse, rotate_left, drop_last`

## Base Reader Result

| Metric | Value |
|---|---:|
| Original EM | 0.4500 |
| Original F1 | 0.5129 |
| Support Recall | 0.9413 |
| Support Complete | 0.8550 |

## Consistency Means

| Metric | All | Correct F1>=0.5 | Wrong F1<0.5 |
|---|---:|---:|---:|
| majority_fraction | 0.8880 | 0.9132 | 0.8596 |
| inverse_entropy | 0.8566 | 0.8844 | 0.8252 |
| inverse_distinct | 0.8912 | 0.9104 | 0.8697 |
| mean_pairwise_f1 | 0.8294 | 0.8604 | 0.7944 |
| normalized_entropy | 0.1434 | 0.1156 | 0.1748 |
| distinct_answer_count | 1.4350 | 1.3585 | 1.5213 |

## AUC: Consistency Predicting Correctness

### label_em

- positives: `90`
- negatives: `110`

| Metric | AUC | 95% CI |
|---|---:|---:|
| majority_fraction | 0.5910 | [0.5302, 0.6524] |
| inverse_entropy | 0.5920 | [0.5315, 0.6543] |
| inverse_distinct | 0.5855 | [0.5254, 0.6479] |
| mean_pairwise_f1 | 0.5859 | [0.5243, 0.6469] |

### label_f1_positive

- positives: `114`
- negatives: `86`

| Metric | AUC | 95% CI |
|---|---:|---:|
| majority_fraction | 0.5294 | [0.4569, 0.5986] |
| inverse_entropy | 0.5290 | [0.4570, 0.5974] |
| inverse_distinct | 0.5224 | [0.4534, 0.5887] |
| mean_pairwise_f1 | 0.5333 | [0.4607, 0.6024] |

### label_f1_ge_0_5

- positives: `106`
- negatives: `94`

| Metric | AUC | 95% CI |
|---|---:|---:|
| majority_fraction | 0.5666 | [0.4986, 0.6315] |
| inverse_entropy | 0.5658 | [0.4982, 0.6317] |
| inverse_distinct | 0.5576 | [0.4902, 0.6226] |
| mean_pairwise_f1 | 0.5606 | [0.4910, 0.6263] |

## Decision Hint

- Primary metric: `majority_fraction` AUC vs `label_f1_ge_0_5` = `0.56664`
- Decision hint: `FAIL_SIGNAL_PROBE`
