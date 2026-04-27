# CEE Bucket Diagnostic

- Rows per config: `1000`
- Variant: `learned_edit1`

## margin15

| Bucket | Queries | Support Complete | Avg ΔComplete | Avg ΔRecall | Added Gold | Added Non-Gold | Non-Gold/Gold | Hard Negative Types |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rank_complete_and_edited | 185 | 0.8919 | -0.1081 | -0.0459 | 0 | 185 | 185.0000 | answer_string_distractor:7, bridge_entity_distractor:8, lexical_hard_negative:144, low_signal_deep_negative:23, semantic_hard_negative:3 |
| rank_complete_and_stopped | 587 | 1.0000 | +0.0000 | +0.0000 | 0 | 0 | 0.0000 |  |
| rank_incomplete_and_edited | 89 | 0.3483 | +0.3483 | +0.1854 | 35 | 54 | 1.5429 | answer_string_distractor:2, bridge_entity_distractor:3, lexical_hard_negative:42, low_signal_deep_negative:6, semantic_hard_negative:1 |
| rank_incomplete_and_stopped | 139 | 0.0000 | +0.0000 | +0.0000 | 0 | 0 | 0.0000 |  |
| oracle_beneficial_and_edited | 68 | 0.4559 | +0.4559 | +0.2426 | 35 | 33 | 0.9429 | answer_string_distractor:1, bridge_entity_distractor:2, lexical_hard_negative:27, low_signal_deep_negative:3 |
| oracle_beneficial_and_stopped | 68 | 0.0000 | +0.0000 | +0.0000 | 0 | 0 | 0.0000 |  |
| oracle_not_beneficial_and_edited | 206 | 0.8010 | -0.0971 | -0.0413 | 0 | 206 | 206.0000 | answer_string_distractor:8, bridge_entity_distractor:9, lexical_hard_negative:159, low_signal_deep_negative:26, semantic_hard_negative:4 |
| oracle_not_beneficial_and_stopped | 658 | 0.8921 | +0.0000 | +0.0000 | 0 | 0 | 0.0000 |  |

## margin20

| Bucket | Queries | Support Complete | Avg ΔComplete | Avg ΔRecall | Added Gold | Added Non-Gold | Non-Gold/Gold | Hard Negative Types |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rank_complete_and_edited | 135 | 0.8889 | -0.1111 | -0.0519 | 0 | 135 | 135.0000 | answer_string_distractor:2, bridge_entity_distractor:8, lexical_hard_negative:110, low_signal_deep_negative:14, semantic_hard_negative:1 |
| rank_complete_and_stopped | 637 | 1.0000 | +0.0000 | +0.0000 | 0 | 0 | 0.0000 |  |
| rank_incomplete_and_edited | 64 | 0.3906 | +0.3906 | +0.2070 | 27 | 37 | 1.3704 | answer_string_distractor:2, bridge_entity_distractor:2, lexical_hard_negative:28, low_signal_deep_negative:4, semantic_hard_negative:1 |
| rank_incomplete_and_stopped | 164 | 0.0000 | +0.0000 | +0.0000 | 0 | 0 | 0.0000 |  |
| oracle_beneficial_and_edited | 50 | 0.5000 | +0.5000 | +0.2650 | 27 | 23 | 0.8519 | answer_string_distractor:1, bridge_entity_distractor:2, lexical_hard_negative:17, low_signal_deep_negative:3 |
| oracle_beneficial_and_stopped | 86 | 0.0000 | +0.0000 | +0.0000 | 0 | 0 | 0.0000 |  |
| oracle_not_beneficial_and_edited | 149 | 0.8054 | -0.1007 | -0.0470 | 0 | 149 | 149.0000 | answer_string_distractor:3, bridge_entity_distractor:8, lexical_hard_negative:121, low_signal_deep_negative:15, semantic_hard_negative:2 |
| oracle_not_beneficial_and_stopped | 715 | 0.8909 | +0.0000 | +0.0000 | 0 | 0 | 0.0000 |  |

## pool10_margin25

| Bucket | Queries | Support Complete | Avg ΔComplete | Avg ΔRecall | Added Gold | Added Non-Gold | Non-Gold/Gold | Hard Negative Types |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rank_complete_and_edited | 30 | 0.8667 | -0.1333 | -0.0667 | 0 | 30 | 30.0000 | answer_string_distractor:1, lexical_hard_negative:26, low_signal_deep_negative:3 |
| rank_complete_and_stopped | 742 | 1.0000 | +0.0000 | +0.0000 | 0 | 0 | 0.0000 |  |
| rank_incomplete_and_edited | 15 | 0.4667 | +0.4667 | +0.2667 | 8 | 7 | 0.8750 | answer_string_distractor:1, lexical_hard_negative:6 |
| rank_incomplete_and_stopped | 213 | 0.0000 | +0.0000 | +0.0000 | 0 | 0 | 0.0000 |  |
| oracle_beneficial_and_edited | 13 | 0.5385 | +0.5385 | +0.3077 | 8 | 5 | 0.6250 | lexical_hard_negative:5 |
| oracle_beneficial_and_stopped | 123 | 0.0000 | +0.0000 | +0.0000 | 0 | 0 | 0.0000 |  |
| oracle_not_beneficial_and_edited | 32 | 0.8125 | -0.1250 | -0.0625 | 0 | 32 | 32.0000 | answer_string_distractor:2, lexical_hard_negative:27, low_signal_deep_negative:3 |
| oracle_not_beneficial_and_stopped | 832 | 0.8918 | +0.0000 | +0.0000 | 0 | 0 | 0.0000 |  |
