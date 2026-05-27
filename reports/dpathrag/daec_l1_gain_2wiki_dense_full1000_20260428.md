# DAEC-L1 Gain Diagnostics

- Dataset: `2wikimultihopqa`
- Source: `dense_pool100`
- Rows: `1000`

## Overall

| Metric | Value |
|---|---:|
| Baseline EM | 0.4550 |
| Selector EM | 0.4980 |
| Delta EM | +0.0430 |
| Baseline F1 | 0.4991 |
| Selector F1 | 0.5541 |
| Delta F1 | +0.0551 |
| Changed-from-baseline rate | 0.8390 |
| Same title set rate | 0.3950 |
| Same doc set rate | 0.3950 |
| Avg title Jaccard | 0.7353 |
| Avg doc Jaccard | 0.7354 |
| Support recall delta | +0.0617 |
| Support complete delta | +0.1150 |
| Noise delta | -0.0292 |
| Duplicate delta | +0.0000 |

## Answer Transitions

| Transition | Rows | Delta F1 | Support Recall Δ | Noise Δ | Title Jaccard |
|---|---:|---:|---:|---:|---:|
| both_correct | 403 | +0.0000 | +0.0143 | -0.0119 | 0.7433 |
| both_wrong | 450 | +0.0287 | +0.0639 | -0.0276 | 0.7659 |
| correct_to_wrong | 52 | -0.9487 | -0.1490 | +0.0654 | 0.5652 |
| wrong_to_correct | 95 | +0.9630 | +0.3684 | -0.1621 | 0.6499 |

## F1 Buckets

| Bucket | Rows | Delta F1 | Support Recall Δ | Noise Δ | Added Gold | Removed Gold | Removed Non-Gold |
|---|---:|---:|---:|---:|---:|---:|---:|
| f1_improved | 128 | +0.8412 | +0.3594 | -0.1547 | 0.7734 | 0.0000 | 1.1094 |
| f1_regressed | 64 | -0.8220 | -0.1211 | +0.0531 | 0.0781 | 0.3438 | 1.1094 |
| f1_unchanged | 808 | +0.0000 | +0.0291 | -0.0158 | 0.1015 | 0.0223 | 0.7884 |
