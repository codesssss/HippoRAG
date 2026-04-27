# C-CEE Decision-Level Separability Dev

- Passed: `False`
- Rows: `200`
- Skipped no-answer-cluster rows: `0`
- Qwen: `{'available': False, 'error': '<urlopen error [Errno 111] Connection refused>'}`
- Sampled counts: `{'oracle_beneficial': 80, 'lexical_HN': 200, 'random_non_beneficial': 200}`

## Oracle-Answer Diagnostic

- AUC beneficial vs lexical HN: `0.713375`
- 95% CI: `[0.6355, 0.785188]`
- AUC beneficial vs random: `0.76725`

## Non-Oracle Decision Diagnostic

- AUC beneficial vs lexical HN: `0.536875`
- 95% CI: `[0.449813, 0.62225]`
- AUC beneficial vs random: `0.490875`
- Summary: `{'queries': 200, 'stop_rate': 0.965, 'top1_beneficial_rate_rank_incomplete': 0.0, 'rank_complete_false_edit_rate': 0.03681, 'added_gold': 0, 'added_non_gold': 7, 'non_gold_per_gold': 7.0, 'support_complete_before': 0.815, 'support_complete_after': 0.805, 'support_complete_gain': -0.01, 'support_recall_before': 0.9175, 'support_recall_after': 0.9125, 'support_recall_gain': -0.005}`
- Checks: `{'auc_ge_0_70': False, 'auc_ci_low_ge_0_65': False, 'top1_beneficial_rate_ge_0_30': False, 'rank_complete_false_edit_rate_le_0_05': True, 'non_gold_per_gold_le_5': False, 'support_complete_gain_ge_1pp': False}`
- Mean reader forwards/query: `108.3`
