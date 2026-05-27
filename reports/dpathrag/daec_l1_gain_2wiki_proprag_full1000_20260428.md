# DAEC-L1 Gain Diagnostics

- Dataset: `2wikimultihopqa`
- Source: `proprag_pool100`
- Rows: `1000`

## Overall

| Metric | Value |
|---|---:|
| Baseline EM | 0.5790 |
| Selector EM | 0.6130 |
| Delta EM | +0.0340 |
| Baseline F1 | 0.6503 |
| Selector F1 | 0.6828 |
| Delta F1 | +0.0325 |
| Changed-from-baseline rate | 0.8260 |
| Same title set rate | 0.5470 |
| Same doc set rate | 0.5470 |
| Avg title Jaccard | 0.7959 |
| Avg doc Jaccard | 0.7960 |
| Support recall delta | +0.0187 |
| Support complete delta | +0.0380 |
| Noise delta | -0.0071 |
| Duplicate delta | +0.0000 |

## Answer Transitions

| Transition | Rows | Delta F1 | Support Recall Δ | Noise Δ | Title Jaccard |
|---|---:|---:|---:|---:|---:|
| both_correct | 545 | +0.0000 | +0.0023 | -0.0005 | 0.8132 |
| both_wrong | 353 | +0.0006 | +0.0205 | -0.0074 | 0.8060 |
| correct_to_wrong | 34 | -0.8432 | -0.1250 | +0.0706 | 0.6682 |
| wrong_to_correct | 68 | +0.8966 | +0.2132 | -0.0971 | 0.6691 |

## F1 Buckets

| Bucket | Rows | Delta F1 | Support Recall Δ | Noise Δ | Added Gold | Removed Gold | Removed Non-Gold |
|---|---:|---:|---:|---:|---:|---:|---:|
| f1_improved | 84 | +0.8061 | +0.2143 | -0.0952 | 0.5119 | 0.0357 | 1.1548 |
| f1_regressed | 49 | -0.7186 | -0.1378 | +0.0694 | 0.0204 | 0.3673 | 0.8367 |
| f1_unchanged | 867 | +0.0000 | +0.0087 | -0.0028 | 0.0554 | 0.0415 | 0.5779 |
