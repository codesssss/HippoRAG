# CEE-v2 Pairwise Admission Plan

Date: 2026-04-27

## Final Decision

Current CEE route:

> **CEE-v2: Conservative Evidence Editing with Admission-Induced STOP and Pairwise Hard-Negative Learning**

Continue:

```text
S0 = PropRAG rank top-5
C = PropRAG top-20
Edit@1 only
pairwise hard-negative edit scorer
admission-induced STOP
DAEC residual / binding features if available
Platt scaling + fixed 0.5 cutoff
```

Stop:

```text
free-form D-PathRAG selector as main method
naked Stage 2 answer-NLL
independent P_need(q, S0) STOP head
Edit@2
threshold / pool-size sweep
prompt / order / blend mitigation
P_use until selection gate passes
```

Rationale:

```text
rank top-5 support_complete = 0.772
free-form D-PathRAG v1 support_complete = 0.805, but added_non_gold = 1205
oracle Edit@1 top20 support_complete = 0.895, added_non_gold = 0
shallow CEE cannot jointly pass support gain and import precision
oracle STOP makes current scorer pass gate, but learned independent STOP does not
```

The method should therefore learn edit admission directly:

```text
P_beneficial(e | q, S0, d_out, d_in)
```

STOP is induced by admission:

```text
e* = argmax_e P_beneficial(e)
if P_beneficial(e*) < 0.5:
    STOP
else:
    apply e*
```

There is no separate STOP classifier in the main method.

## Evidence From Completed Diagnostics

### Oracle edit headroom

On 2Wiki PropRAG eval1000:

| Variant | Support Complete | Added Gold | Added Non-Gold | Non-Gold/Gold |
|---|---:|---:|---:|---:|
| rank top-5 | 0.772 | - | - | - |
| D-PathRAG v1 | 0.805 | 76 | 1205 | 15.86 |
| oracle Edit@1 top20 | 0.895 | 136 | 0 | 0.0 |
| oracle Edit@2 top20 | 0.900 | 141 | 0 | 0.0 |

Interpretation:

```text
Edit@1 action space is correct.
Edit@2 adds little oracle headroom and should not be pursued now.
The bottleneck is learned admission, not CEE formulation.
```

### Shallow CEE trade-off

| Config | Pool | Support Complete | Delta vs Rank | Added Gold | Added Non-Gold | Non-Gold/Gold |
|---|---:|---:|---:|---:|---:|---:|
| margin15 | 20 | 0.783 | +1.1pp | 35 | 239 | 6.83 |
| margin20 | 20 | 0.782 | +1.0pp | 27 | 172 | 6.37 |
| pool10_margin25 | 10 | 0.775 | +0.3pp | 8 | 37 | 4.63 |

Interpretation:

```text
Support gain and import precision are both reachable, but current shallow scorer cannot achieve both together.
More threshold / pool sweep is not useful.
```

### STOP diagnostics

Completed reports:

```text
reports/dpathrag/cee_bucket_diagnostic.md
reports/dpathrag/cee_oracle_stop_eval.md
reports/dpathrag/cee_learned_stop_eval.md
```

Key results:

```text
margin20 rank_complete_and_edited:
  queries = 135
  added_gold = 0
  added_non_gold = 135
  avg_support_complete_delta = -0.1111

margin20 oracle_not_beneficial_and_edited:
  queries = 149
  added_gold = 0
  added_non_gold = 149
  avg_support_complete_delta = -0.1007

objective_oracle_margin20:
  support_complete = 0.797
  added_gold = 27
  added_non_gold = 23
  non_gold/gold = 0.8519
  gate = 3/3

learned_stop_margin20:
  support_complete = 0.778
  added_gold = 16
  added_non_gold = 103
  non_gold/gold = 6.4375
  gate = 1/3
```

Interpretation:

```text
Correct STOP is sufficient to make current edit scorer pass the selection gate.
But independent learned STOP is not learnable enough with current set-level features.
STOP should be induced by edit admission, not trained as an independent head.
```

## Method Definition

Given query `q`, PropRAG candidate pool:

```text
C_q = {d1, ..., d20}
S0 = rank top-5
```

Candidate edits:

```text
e = (d_out, d_in)
d_out in S0
d_in in C_q \ S0
S_e = S0 - {d_out} + {d_in}
```

Learn a scalar score:

```text
s_theta(e) = scorer(q, S0, d_out, d_in)
```

Calibrate per fold with Platt scaling:

```text
P_beneficial(e) = sigmoid(a * s_theta(e) + b)
```

Inference:

```text
score all Edit@1 candidates
e* = argmax_e P_beneficial(e)

if P_beneficial(e*) < 0.5:
    return S0
else:
    return S_e*
```

No margin sweep. No eval-fold threshold tuning.

## Pairwise Dataset

### Oracle objective

Use the existing tuple objective:

```text
J(S) = (
  support_complete,
  support_recall,
  selected_gold_count,
  bridge_entity_recall,
  -added_non_gold
)
```

An edit is oracle-beneficial if:

```text
J(S_e) > J(S0)
```

### Positive edits

Priority:

```text
1. oracle-beneficial edit
2. add missing gold + remove non-gold
3. support_complete / support_recall / selected_gold_count improving edit
```

If multiple positives exist, use the best by `J`, and optionally include top positives.

### Negative edits

Sampling mix:

```text
50% lexical hard negatives
30% mixed hard negatives
20% graded preference pairs
```

Lexical hard negative:

```text
gold_support = 0
and (
  question_token_coverage >= 0.35
  or title_question_jaccard >= 0.08
  or body_question_jaccard >= 0.08
)
```

Mixed hard negatives:

```text
semantic_hard_negative
answer_string_distractor
bridge_entity_distractor
low-rank non-gold
```

Graded preference:

```text
e1 > e2 iff J(S_e1) > J(S_e2)
```

Prefer same query and same `d_out`:

```text
positive = remove d_out, add missing gold
negative = remove same d_out, add hard negative
```

## Features

### v0 features, no DAEC dependency

Use only features already available in current caches:

```text
query aggregate features
S0 aggregate features
add candidate features
remove candidate features
add-minus-remove deltas
after-edit set features
```

Required feature groups:

```text
rank_reciprocal(add/remove)
rank_delta
retriever_score(add/remove/delta)
q_doc_cosine(add/remove/delta)
title_question_jaccard(add/remove/delta)
body_question_jaccard(add/remove/delta)
question_token_coverage(add/remove/delta)
query type / answer type proxy features
log_doc_chars(add/remove/delta)
duplicate_title_after_edit
unique_title_delta
same_title_family_count_after_edit
title_entity_overlap_delta
```

Gold labels are never input features.

### v1 features, if DAEC audit passes

First audit whether DAEC features are already cached per query/doc:

```text
demand-supply scores
binding consistency
type compatibility
rank prior / confidence
```

If available, add them as model features:

```text
demand residual coverage delta
binding consistency delta
bridge coverage delta
answer type compatibility
relation/type compatibility
whether add connects to retained entity
whether remove breaks existing entity chain
```

If not available, do not block v0. Record DAEC as v1 ablation.

## Model

Implement two scorers:

```text
CEE-Pairwise-Linear
CEE-Pairwise-MLP
```

Linear is for feature/loss sanity. MLP is the main candidate.

MLP default:

```text
hidden_dim = 128 or 256
layers = 2
dropout = 0.1
output = scalar score s_theta(e)
```

Training loss:

```text
L = -log sigmoid(s_theta(e_pos) - s_theta(e_neg))
```

Calibration:

```text
train_inner: train pairwise scorer
calib_inner: fit Platt scaling
eval_fold: held-out evaluation
```

The inference cutoff is fixed:

```text
P_beneficial >= 0.5
```

## Required Diagnostics Before Pairwise Training

### Diagnostic C

Run same-remove false positive / false negative analysis:

```text
objective_oracle_margin20 false positives / false negatives
same query + same remove:
  missing gold edit vs lexical hard negative edit
```

Output:

```text
reports/dpathrag/cee_diagnostic_c.md
```

Report:

```text
number of same-remove positive/lexical-negative pairs
rank distribution of positives and negatives
feature deltas between positives and lexical negatives
how many current shallow false positives are lexical hard negatives
how many current shallow false negatives are missing-gold edits
```

### DAEC feature audit

Audit only; do not build new DAEC pipeline in this step.

Output:

```text
reports/dpathrag/daec_feature_audit.md
```

Report:

```text
available cache files
whether features are per query, per doc, or per pair
which features can be joined to CEE candidates by qid/doc title/rank
estimated cost if features are missing
decision: use DAEC now, defer DAEC, or rebuild DAEC cache
```

## Evaluation

Protocol:

```text
2Wiki PropRAG eval1000
5-fold cross-fit
train 800 / eval 200 per fold
S0 = rank top-5
C = top20
Edit@1 only
```

Baselines:

```text
PropRAG rank top-5
D-PathRAG v1 free-form selector
shallow CEE margin20
oracle Edit@1 top20
CEE-pairwise-v0 no DAEC
CEE-pairwise-v1 + DAEC features, if available
CEE-pairwise w/o lexical negative focus
CEE-pairwise w/o Platt calibration
```

Selection metrics:

```text
support_recall
support_complete
selected_gold_count
bridge_entity_recall
```

Admission metrics:

```text
added_gold
added_non_gold
non_gold/gold
rank_complete_over_edit_rate
oracle_beneficial_edit_recall
oracle_not_beneficial_false_positive_rate
lexical_hard_negative_admission_rate
added_non_gold <= 50% selector_v1
```

Reader metrics are postponed until selection gate passes:

```text
EM
F1
answer-in-context-but-fail
```

## Gate And Decision

Selection gate:

```text
support_complete >= rank_top5 + 1pp
non_gold/gold <= 5
added_non_gold <= 50% * selector_v1_added_non_gold
```

Known selector-v1 reference:

```text
selector_v1_added_non_gold = 1205
50% threshold = 602
```

Decision table:

| Status | Condition | Decision |
|---|---|---|
| Green | selection gate passes and reader F1 >= rank baseline | CEE-v2 can be main method |
| Yellow | selection gate passes but reader F1 flat/slightly down | run P_use diagnostic |
| Orange | support gain passes but ratio > 5 | improve hard-negative sampling / DAEC binding |
| Red | support and ratio both fail | stop CEE, return to DAEC mainline |

## P_use Policy

Do not implement now.

Only run P_use if:

```text
CEE-pairwise passes selection gate
but reader F1 does not improve or stays yellow
```

If needed:

```text
Delta NLL(e) = NLL_reader(a | q, S_e) - NLL_reader(a | q, S0)
```

Use P_use only as a product-of-experts factor with P_mech:

```text
P(e) proportional to P_mech(e) * P_use(e)
```

Never use reader utility alone as the edit admission signal.

## Execution Timeline

### Week 1

Day 1:

```text
Diagnostic C
DAEC integration audit
```

Outputs:

```text
reports/dpathrag/cee_diagnostic_c.md
reports/dpathrag/daec_feature_audit.md
```

Day 2-4:

```text
construct pairwise dataset
positive = oracle beneficial edit
negative = lexical/mixed/graded hard negatives
same-query / same-remove pairing
```

Outputs:

```text
data/dpathrag/cee_pairwise_train_folds/
reports/dpathrag/cee_pairwise_dataset_stats.md
```

Day 5-7:

```text
implement CEE-Pairwise-Linear
implement CEE-Pairwise-MLP
train v0 without DAEC
cross-fit eval1000
```

Output:

```text
reports/dpathrag/cee_pairwise_v0_eval.md
```

### Week 2

Day 8-10:

```text
add DAEC residual / binding features if audit passes
otherwise keep v0 and document missing DAEC integration
```

Day 11-12:

```text
cross-fit eval1000
Platt scaling + fixed 0.5 cutoff
main selection table
```

Day 13-14:

```text
ablation:
  w/o DAEC features
  w/o lexical-focused negatives
  w/o calibration
  linear vs MLP
```

Outputs:

```text
reports/dpathrag/cee_pairwise_v1_eval.md
reports/dpathrag/cee_pairwise_ablation.md
```

### Week 3

If Green:

```text
run reader eval
write method section
```

If Yellow:

```text
run P_use diagnostic only
```

If Orange:

```text
improve negative sampling / DAEC binding features
```

If Red:

```text
stop CEE
return DAEC mainline
write CEE as diagnostic
```

## Paper Framing

Use this story if CEE-pairwise works:

> Free-form autoregressive evidence selection improves support coverage but over-edits strong rank contexts and imports lexical hard negatives. Conservative Evidence Editing starts from a strong rank baseline and learns calibrated single-edit admission. Oracle Edit@1 shows large support-complete headroom, while pairwise hard-negative learning turns that headroom into a controlled edit policy.

Claim:

```text
Multi-hop evidence selection over strong graph/rank substrates should be
formulated as conservative evidence editing, not free-form top-k replacement.
```

Current route status:

| Direction | Status |
|---|---|
| BSGS | stopped, keep as negative diagnostic |
| D-PathRAG v1 free-form selector | keep mechanism-positive failure analysis |
| naked Stage 2 | stopped |
| shallow CEE scorer | keep as baseline |
| independent learned STOP head | stopped |
| CEE-v2 pairwise scorer | main route |
| DAEC residual / binding features | audit, then optional v1 |
| P_use | postponed |

