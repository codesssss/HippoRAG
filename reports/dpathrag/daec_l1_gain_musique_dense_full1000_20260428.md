# DAEC-L1 Gain Diagnostics

- Dataset: `musique`
- Source: `dense_pool100`
- Rows: `1000`

## Overall

| Metric | Value |
|---|---:|
| Baseline EM | 0.2980 |
| Selector EM | 0.2900 |
| Delta EM | -0.0080 |
| Baseline F1 | 0.3896 |
| Selector F1 | 0.3818 |
| Delta F1 | -0.0078 |
| Changed-from-baseline rate | 0.8430 |
| Same title set rate | 0.3730 |
| Same doc set rate | 0.3640 |
| Avg title Jaccard | 0.6794 |
| Avg doc Jaccard | 0.6689 |
| Support recall delta | -0.0034 |
| Support complete delta | -0.0140 |
| Noise delta | +0.0237 |
| Duplicate delta | -0.0750 |

## Answer Transitions

| Transition | Rows | Delta F1 | Support Recall Δ | Noise Δ | Title Jaccard |
|---|---:|---:|---:|---:|---:|
| both_correct | 233 | +0.0000 | +0.0043 | +0.0161 | 0.7999 |
| both_wrong | 645 | -0.0043 | +0.0048 | +0.0157 | 0.6626 |
| correct_to_wrong | 65 | -0.8648 | -0.2564 | +0.1887 | 0.4542 |
| wrong_to_correct | 57 | +0.8982 | +0.1608 | -0.0439 | 0.6342 |

## F1 Buckets

| Bucket | Rows | Delta F1 | Support Recall Δ | Noise Δ | Added Gold | Removed Gold | Removed Non-Gold |
|---|---:|---:|---:|---:|---:|---:|---:|
| f1_improved | 88 | +0.7506 | +0.1364 | -0.0356 | 0.4318 | 0.1250 | 0.9545 |
| f1_regressed | 109 | -0.6771 | -0.2141 | +0.1495 | 0.1009 | 0.6789 | 1.0826 |
| f1_unchanged | 803 | +0.0000 | +0.0099 | +0.0131 | 0.1245 | 0.0946 | 0.7522 |
