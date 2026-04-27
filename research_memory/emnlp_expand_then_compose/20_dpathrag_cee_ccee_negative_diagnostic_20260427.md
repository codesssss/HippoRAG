# D-PathRAG / CEE / C-CEE Negative Diagnostic - 2026-04-27

## Context

This note records the final outcome of the D-PathRAG and Conservative Evidence Editing branches on the 2Wiki PropRAG eval1000 protocol.

The branch asked whether a selector or edit operator could convert the strong PropRAG top-20 pool into a better top-5 evidence set:

- `D-PathRAG v1`: free-form autoregressive top-5 selector over a PropRAG pool.
- `CEE`: Conservative Evidence Editing, starting from PropRAG top-5 and applying at most one replacement edit.
- `C-CEE`: Counterfactual Conservative Evidence Editing, using frozen reader likelihood as a test-time edit admission signal.

Final decision:

```text
D-PathRAG v1: mechanism-positive, answer-level yellow.
CEE-pairwise: stopped.
C-CEE: stopped after Day-1 gate.
This line should not be expanded to full eval/test or other datasets now.
```

## Artifacts

Core D-PathRAG reports:

- `reports/dpathrag/proprag_kfold_eval1000_results.md`
- `reports/dpathrag/hard_negative_characterization_v2_proprag_kfold1000.md`
- `reports/dpathrag/context_ablation_proprag_eval200_results.md`

CEE reports:

- `reports/dpathrag/cee_oracle_stop_eval.md`
- `reports/dpathrag/cee_bucket_diagnostic.md`
- `reports/dpathrag/cee_pairwise_v0_implementation_summary.md`
- `reports/dpathrag/cee_pairwise_v0_eval.md`

C-CEE reports:

- `reports/cee/asset_audit.md`
- `reports/cee/likelihood_sanity.md`
- `reports/cee/reader_semantic_capacity.md`
- `reports/cee/decision_separability_dev.md`
- `reports/cee/day1_summary.md`
- `reports/cee/ccee_negative_result_summary.md`

Implementation:

- `scripts/dpathrag_cee_pairwise_common.py`
- `scripts/dpathrag_cee_diagnostic_c.py`
- `scripts/dpathrag_cee_build_pairwise_dataset.py`
- `scripts/dpathrag_cee_pairwise_eval.py`
- `scripts/cee_day1_diagnostic.py`
- `tests/dpathrag/test_cee_pairwise.py`
- `tests/dpathrag/test_ccee_day1.py`

Validation:

```text
.venv-hipporag/bin/python -m pytest tests/dpathrag
50 passed, 2 warnings
```

## Raw Comparison

| Variant | Support Complete | Support Recall | Added Gold | Added Non-Gold | Non-Gold/Gold | Reader F1 | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| PropRAG rank top-5 | 0.7720 | 0.9028 | - | - | - | 0.4824 | baseline |
| D-PathRAG v1 selector | 0.8050 | 0.9233 | 76 | 1205 | 15.8553 | 0.4634 | mechanism-positive, reader-negative/yellow |
| Shallow CEE margin20 | 0.7780 | 0.9070 | 54 | 344 | 6.3704 | - | stopped |
| CEE-pairwise linear v0 | 0.7710 | 0.9032 | 3 | 49 | 16.3333 | - | stopped |
| CEE-pairwise MLP v0 | 0.7710 | 0.9032 | 9 | 138 | 15.3333 | - | stopped |
| Oracle Edit@1 top20 | 0.8950 | 0.9593 | 136 | 0 | 0.0 | - | ceiling only |

D-PathRAG reader result:

```text
rank top-5 EM/F1 = 0.4250 / 0.4824
D-PathRAG EM/F1 = 0.4110 / 0.4634
F1 delta = -1.90pp, 95% CI [-4.03pp, +0.23pp]
```

C-CEE Day-1 result:

```text
decision = STOP_DECISION_SEPARABILITY_FAIL
reader semantic AUC = 0.988332
oracle-answer beneficial-vs-lexical-HN AUC = 0.713375, CI [0.6355, 0.785188]
non-oracle beneficial-vs-lexical-HN AUC = 0.536875, CI [0.449813, 0.62225]
top1 beneficial rate on rank-incomplete = 0.0
support-complete gain on dev = -0.01
```

## Key Findings

### 1. D-PathRAG v1 is mechanism-positive but not reader-positive

The non-leaky 5-fold eval1000 result is real at the mechanism level:

```text
support_complete: 0.7720 -> 0.8050 (+3.30pp)
support_recall: 0.9028 -> 0.9233 (+2.05pp)
selection_overlap with rank top-5: 0.6192
```

But answer F1 does not improve:

```text
F1: 0.4824 -> 0.4634 (-1.90pp)
```

The correct claim is:

> Autoregressive selection over a strong PropRAG pool can improve evidence coverage, but the current selector does not reliably translate that gain into reader F1.

### 2. Hard-negative import is the central failure mode

D-PathRAG v1 recovers some missing gold docs:

```text
selector_added_gold = 76
mean rank = 9.6842
answer_in_doc = 52.6%
bridge_entity_in_doc = 76.3%
```

But imports far more non-gold:

```text
selector_added_non_gold = 1205
mean rank = 18.0083
answer_in_doc = 4.1%
bridge_entity_in_doc = 4.8%
```

The effective import ratio is about:

```text
1 added gold : 15.86 added non-gold
```

This explains the support/reader decoupling.

### 3. Reader context is distractor-sensitive

The context ablation shows that even when gold support is present, selector distractors hurt:

```text
gold_support_only F1 = 0.7070
gold_plus_selector_distractors F1 = 0.5896
drop = 11.74pp
```

This supports the three-layer bottleneck:

```text
evidence presence != evidence coherence != reader consumability
```

### 4. CEE action space is valid, but admission is not learned

Oracle Edit@1 proves that local conservative edits have large headroom:

```text
rank top-5 support_complete = 0.772
oracle Edit@1 top20 support_complete = 0.895
```

However, learned admission fails:

```text
CEE-pairwise linear beneficial recall = 0.0221
CEE-pairwise MLP beneficial recall = 0.0588
```

Candidate-level calibration solves over-editing but makes the model too conservative. It stops many bad edits, but does not find enough good edits.

### 5. Frozen reader likelihood is not enough under answer uncertainty

C-CEE tested whether reader likelihood could replace learned admission.

Positive result:

```text
reader semantic capacity AUC = 0.988332
```

This means the reader can score correct answers higher than random distractors when evidence is complete.

Negative result:

```text
non-oracle beneficial-vs-lexical-HN AUC = 0.536875
top1 beneficial rate = 0.0
support-complete gain = -1.0pp
```

Therefore the failure is not simply "reader likelihood is meaningless." The failure is:

> Without a reliable answer hypothesis, counterfactual reader likelihood does not identify beneficial evidence edits among lexical hard negatives.

## Interpretation

The full branch yields a coherent negative result:

```text
The oracle edit space is strong, but all practical admission mechanisms tried so far fail:
  free-form selector: high recall, low precision
  shallow CEE: weak admission calibration
  pairwise CEE: local preference does not become global calibrated admission
  C-CEE: reader likelihood fails under non-oracle answer uncertainty
```

This supports moving the paper mainline back to DAEC / Layer-1 fixed-pool composition.

## Paper Use

Recommended framing:

> We find that evidence selection gains can be decoupled from answer accuracy. A free-form selector improves support completeness but imports reader-hostile hard negatives. Conservative editing exposes substantial oracle headroom, but learned pairwise admission and frozen-reader counterfactual admission both fail to recover this headroom under realistic non-oracle conditions. This motivates treating hard-negative admission and reader consumption as distinct bottlenecks rather than assuming support-complete improvements automatically translate to answer F1.

Use D-PathRAG/CEE/C-CEE as:

- a diagnostic section,
- a negative-control appendix,
- evidence for the three-layer bottleneck framing,
- justification for why the main method should remain DAEC / fixed-pool composition rather than a learned selector line.

Do not claim:

- D-PathRAG improves answer F1,
- CEE-pairwise solves conservative editing,
- C-CEE is a viable training-free method,
- reader likelihood alone solves hard-negative admission.

## Stopped Lines

The following should not receive more engineering time in the current paper cycle:

```text
free-form D-PathRAG expansion to HotpotQA/MuSiQue
naked Stage 2 answer-NLL training
prompt/order/two-block mitigation sweeps
rank-anchor blend sweeps
CEE-pairwise threshold or pool sweeps
CEE-pairwise DAEC feature integration
C-CEE Phase 2 implementation
C-CEE held-out eval800/eval1000
```

Only reopen this line if a new mechanism supplies reliable answer hypotheses or direct edit supervision.

## Final Decision

```text
BSGS: stopped; negative graph-native belief diagnostic.
D-PathRAG v1: stopped as main method; keep mechanism-positive diagnostic.
CEE-pairwise: stopped.
C-CEE: stopped after Day-1 decision gate.
DAEC / Layer-1 fixed-pool composition: remains paper floor and mainline.
```
