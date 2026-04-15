# NQ PopQA Bridge Threshold Diagnosis

Last updated: 2026-04-10

## Scope

This note records the focused `limit=100` diagnosis for why `bridge_append_plus_ce` did not show an advantage on `nq` and `popqa` under the width-matched CE control setup.

The goal was to separate three hypotheses:

1. `bridge` is failing because `query_entity_source=seed` is too weak
2. `bridge` is failing because `expand_min_structure_score=0.35` is too strict
3. `bridge` can append docs, but those extra docs still do not convert into QA utility

## Canonical files

- Diagnosis runner:
  - `run_logs/diagnose_bridge_failure_nq_popqa_100_20260410.sh`
- Diagnosis summary:
  - `run_logs/bridge_failure_diagnosis_nq_popqa_100_20260410bridge_diag.summary.md`
- NQ diagnosis reports:
  - `outputs_step0_general_nq/eval_reports/bridge_diag_nq_seed_thr000_20260410bridge_diag.json`
  - `outputs_step0_general_nq/eval_reports/bridge_diag_nq_question_thr035_20260410bridge_diag.json`
  - `outputs_step0_general_nq/eval_reports/bridge_diag_nq_question_thr000_20260410bridge_diag.json`
- PopQA diagnosis reports:
  - `outputs_step0_general_popqa/eval_reports/bridge_diag_popqa_seed_thr000_20260410bridge_diag.json`
  - `outputs_step0_general_popqa/eval_reports/bridge_diag_popqa_question_thr035_20260410bridge_diag.json`
  - `outputs_step0_general_popqa/eval_reports/bridge_diag_popqa_question_thr000_20260410bridge_diag.json`

## Control baseline

The reference control for both datasets is the same width-matched CE rerank over the top-10 baseline prefix:

- `nq baseline_top10_plus_ce`:
  - `EM/F1 = 0.5300 / 0.6444`
  - `R@5/R@20/R@100 = 0.6882 / 0.9872 / 1.0000`
- `popqa baseline_top10_plus_ce`:
  - `EM/F1 = 0.2000 / 0.4819`
  - `R@5/R@20/R@100 = 0.5000 / 0.5300 / 0.5750`

Default bridge control at `expand_min_structure_score = 0.35` is identical to baseline on both datasets and has:

- `append_query_rate = 0.0000`
- `append_stop_reason = structure_below_threshold:100`

Interpretation:
- the default bridge configuration is not "losing after appending"
- it is mostly failing before any append happens

## NQ diagnosis

### Results

- `question@0.35`:
  - `EM/F1 = 0.5300 / 0.6444`
  - `append_query_rate = 0.0000`
  - `final_appended_rate = 0.0000`
  - `stop_reason = structure_below_threshold:100`
- `seed@0.0`:
  - `EM/F1 = 0.5300 / 0.6444`
  - `R@5 = 0.6922`
  - `append_query_rate = 1.0000`
  - `avg_append_count = 3.0000`
  - `final_appended_rate = 0.1300`
  - `stop_reason = append_cap_reached:100`
- `question@0.0`:
  - identical to `seed@0.0`

### Interpretation

- `query_entity_source` is not the bottleneck for `nq`:
  - `seed` and `question` behave the same at both tested thresholds
- `expand_min_structure_score = 0.35` is the direct reason the default bridge run appends nothing
- lowering the threshold to `0.0` fully unlocks append behavior
- however, once unlocked, the extra appended docs still do not improve QA

Bottom line for `nq`:

> The default bridge line is threshold-gated, not query-source-gated; but even when threshold is removed and bridge appends three docs per query, the extra docs do not yield answer utility.

## PopQA diagnosis

### Results

- `question@0.35`:
  - `EM/F1 = 0.2000 / 0.4819`
  - `append_query_rate = 0.0000`
  - `final_appended_rate = 0.0000`
  - `stop_reason = structure_below_threshold:100`
- `seed@0.0`:
  - `EM/F1 = 0.2100 / 0.4853`
  - `append_query_rate = 1.0000`
  - `avg_append_count = 3.0000`
  - `final_appended_rate = 0.2100`
  - `stop_reason = append_cap_reached:100`
- `question@0.0`:
  - identical to `seed@0.0`

### Interpretation

- `query_entity_source` is again not the bottleneck:
  - `seed` and `question` match each other once threshold is fixed
- `expand_min_structure_score = 0.35` is again the direct cause of the default bridge failure
- lowering the threshold to `0.0` fully unlocks append behavior
- unlike `nq`, `popqa` shows a small positive gain once unlocked:
  - `EM +0.0100`
  - `F1 +0.0034`

Bottom line for `popqa`:

> The default bridge line is again threshold-gated rather than query-source-gated, but threshold-unlocked bridge does produce a small positive QA signal.

## Cross-dataset conclusion

The diagnosis supports four concrete claims:

1. The default bridge failure on `nq` and `popqa` is primarily a threshold issue:
   - `expand_min_structure_score = 0.35` blocks all appended docs.
2. `setwise_query_entity_source` is not the main fix path for these two datasets:
   - `seed` and `question` give the same outcome under matched thresholds.
3. After threshold is removed, bridge behavior becomes active:
   - `append_query_rate = 1.0`
   - `avg_append_count = 3.0`
4. Once threshold is removed, the datasets diverge:
   - `nq`: active bridge still has no QA gain
   - `popqa`: active bridge has a small QA gain

## Research consequence

This diagnosis changes the interpretation of the earlier `nq/popqa` control results.

The previous no-gain result should not be paraphrased simply as:

> bridge has no value on `nq/popqa`

The more accurate statement is:

> under the default bridge threshold, the method is largely inactive on both datasets; once the threshold gate is removed, `nq` still shows no answer-utility gain while `popqa` shows a small positive signal.

## Engineering consequence

If future agents rerun this family, they should keep two rules in mind:

1. Do not use the default `0.35` threshold as the only diagnostic setting for `nq/popqa`.
2. If the question is "is bridge structurally alive?", include at least one `expand_min_structure_score = 0.0` run.

This note is diagnostic evidence, not yet a new default configuration recommendation.
