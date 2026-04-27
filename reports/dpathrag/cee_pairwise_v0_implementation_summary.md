# CEE-v2 Pairwise Admission Implementation Summary

## Implemented Scope

This implementation follows the CEE-v2 decision:

```text
S0 = PropRAG rank top-5
C = PropRAG top20
Edit@1 only
Learn P_beneficial(e | q, S0, d_out, d_in)
STOP is induced by admission:
  if max_e calibrated P_beneficial(e) < 0.5: STOP
  else apply best edit
```

No independent STOP head, no Edit@2, no threshold sweep, no naked Stage 2, and no P_use were implemented.

## New Code

```text
scripts/dpathrag_cee_pairwise_common.py
scripts/dpathrag_cee_diagnostic_c.py
scripts/dpathrag_cee_build_pairwise_dataset.py
scripts/dpathrag_cee_pairwise_eval.py
tests/dpathrag/test_cee_pairwise.py
```

Key implementation details:

- `dpathrag_cee_pairwise_common.py` defines edit enumeration, tuple-objective labels, same-remove pair construction, feature vectors, linear pairwise scorer, MLP pairwise scorer, and Platt calibration helpers.
- `dpathrag_cee_diagnostic_c.py` computes same-remove beneficial-vs-lexical pairs, shallow CEE false positives/false negatives, feature deltas between positives and lexical negatives, and a DAEC feature cache audit.
- `dpathrag_cee_build_pairwise_dataset.py` builds fold-safe train/calib JSONL pairs under `data/dpathrag/cee_pairwise_train_folds/`.
- `dpathrag_cee_pairwise_eval.py` trains cross-fit linear/MLP pairwise scorers, calibrates at the edit-candidate level, and evaluates fixed 0.5 admission-induced STOP.
- Gold labels are used only for oracle labels/evaluation. Feature vectors use rank, retriever score, lexical, query, embedding, add/remove delta, and after-edit title-coherence features.

## Calibration Detail

The first linear eval exposed a calibration mismatch: Platt scaling on balanced pair positives/negatives caused almost every query to admit an edit after max-over-edits inference. This produced:

```text
cee_pairwise_linear_v0 support_complete = 0.701
added_non_gold = 928
stop_rate = 0.030
```

The final code therefore uses fold-safe candidate-level calibration:

```text
For calibration qids in the train fold:
  enumerate all Edit@1 candidates
  label each candidate as beneficial iff J(S_e) > J(S0)
  fit Platt scaling on all candidate scores
Inference:
  fixed cutoff P_beneficial >= 0.5
```

This matches admission-induced STOP better than balanced-pair calibration and avoids threshold sweeping.

## Diagnostic C Results

Output:

```text
reports/dpathrag/cee_diagnostic_c.md
reports/dpathrag/cee_diagnostic_c.json
reports/dpathrag/cee_diagnostic_c_pairs.csv
reports/dpathrag/daec_feature_audit.md
```

Main numbers:

```text
beneficial edits = 543
lexical negative edits = 36,230
same-remove beneficial-vs-lexical pairs = 526
queries with same-remove pairs = 132
shallow CEE false-positive edited queries/edits = 185 / 294
shallow CEE false-negative queries = 109
```

Positive add vs lexical negative add means:

| Feature | Positive | Lexical Negative | Delta |
|---|---:|---:|---:|
| rank | 9.9245 | 12.6372 | -2.7127 |
| q_doc_cosine | 0.2793 | 0.2111 | +0.0682 |
| answer_in_doc | 0.5893 | 0.0000 | +0.5893 |
| bridge_entity_in_doc | 0.8692 | 0.0000 | +0.8692 |
| question_token_coverage | 0.3677 | 0.3509 | +0.0168 |

DAEC audit result:

```text
status = no_direct_joinable_cache_found
decision = run CEE-pairwise v0 first; add DAEC features only after fold-safe per-query/per-doc cache is confirmed
```

## Pairwise Dataset

Output:

```text
data/dpathrag/cee_pairwise_train_folds/
reports/dpathrag/cee_pairwise_dataset_stats.md
reports/dpathrag/cee_pairwise_dataset_stats.json
```

Final dataset after fixing the sampler:

```text
rows = 1000
folds = 5
candidate_pool = top20
train pairs = 2540 across 136 queries
calibration pairs = 636 across 136 queries
same-remove rate = 1.0
```

Train pair types:

```text
lexical = 1259
mixed = 855
graded = 426
```

This approximates the intended 50/30/20 focus and avoids the initial bug where lexical pairs filled all slots before mixed/graded pairs could be sampled.

## Eval1000 Results

Output:

```text
reports/dpathrag/cee_pairwise_v0_eval.md
reports/dpathrag/cee_pairwise_v0_eval.json
reports/dpathrag/cee_pairwise_v0.predictions.jsonl
```

Main table:

| Variant | Support Complete | Support Recall | Added Gold | Added Non-Gold | Non-Gold/Gold | Stop Rate | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| rank top-5 | 0.7720 | 0.9028 | - | - | - | - | - |
| oracle Edit@1 top20 | 0.8950 | 0.9593 | 136 | 0 | 0.0 | 0.864 | - |
| D-PathRAG v1 selector | 0.8050 | 0.9233 | 76 | 1205 | 15.8553 | - | - |
| shallow CEE margin20 | 0.7780 | 0.9070 | 54 | 344 | 6.3704 | 0.801 | - |
| CEE-pairwise-linear v0 | 0.7710 | 0.9032 | 3 | 49 | 16.3333 | 0.948 | 1/3 |
| CEE-pairwise-MLP v0 | 0.7710 | 0.9032 | 9 | 138 | 15.3333 | 0.853 | 1/3 |

Admission diagnostics:

| Variant | Rank-Complete Over-Edit | Oracle Beneficial Recall | Oracle Non-Beneficial FP | Lexical HN Admission |
|---|---:|---:|---:|---:|
| CEE-pairwise-linear v0 | 0.0430 | 0.0221 | 0.0509 | 0.8269 |
| CEE-pairwise-MLP v0 | 0.1190 | 0.0588 | 0.1481 | 0.6327 |

Selection gate:

```text
support_complete >= rank_top5 + 1pp: FAIL
non_gold/gold <= 5: FAIL
added_non_gold <= 50% selector_v1: PASS
overall = 1/3 for both linear and MLP
```

## Interpretation

CEE formulation remains valid because oracle Edit@1 top20 has a large ceiling:

```text
rank top-5 support_complete = 0.772
oracle Edit@1 top20 support_complete = 0.895
```

But CEE-pairwise v0 does not learn useful admission:

- Candidate-level calibration fixes over-editing but makes the model too conservative.
- Linear admits only 52 edits; MLP admits 147 edits, but most admitted non-gold remain lexical hard negatives.
- Beneficial edit recall is very low: 0.0221 for linear and 0.0588 for MLP.
- Support_complete does not improve over rank top-5.
- Non_gold/gold ratio remains around 15-16, close to free-form selector failure rather than the target <=5.

The bottleneck is now more precise:

```text
The current v0 feature set and pairwise loss can induce STOP after proper calibration,
but cannot assign high enough calibrated probability to missing-gold edits while rejecting lexical hard negatives.
```

## Next Modeling Problem

The next step should not be threshold tuning. The problem to solve is:

```text
How to make P_beneficial(e) separable for missing-gold edits vs same-remove lexical hard negatives
under candidate-level calibration?
```

Likely required additions:

- DAEC/demand residual features if a fold-safe per-query/per-doc cache can be built.
- Binding/entity-chain compatibility features, not just lexical overlap and embedding cosine.
- A stronger admission objective that directly optimizes query-level max admission rather than only pairwise local ranking.
- Possibly reader utility (`P_use`) only after support/admission gate improves.

## Verification

```text
.venv-hipporag/bin/python -m pytest tests/dpathrag
42 passed, 2 warnings
```
