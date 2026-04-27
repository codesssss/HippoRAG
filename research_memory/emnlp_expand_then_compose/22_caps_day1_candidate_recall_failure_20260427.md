# CAPS Day-1 Candidate Recall Failure - 2026-04-27

## Context

CAPS was opened after stopping D-PathRAG / CEE / C-CEE. The modeling shift was to move from evidence-first selection to answer-conditioned proof search:

```text
old latent object: S or edit e
new latent object: (candidate answer a, proof set S)
```

Day 0 passed:

```text
cross-encoder/nli-deberta-v3-base
template obligation-doc AUC = 0.903
95% CI = [0.8616, 0.9401]
decision = PROCEED_DAY1
```

Day 1 tested the first blocking requirement: candidate answer recall.

## Implementation

Script:

```text
scripts/caps_day1_gates.py
```

Reports:

```text
reports/caps/candidate_answer_recall.md
reports/caps/day1_summary.md
reports/caps/caps_day1_candidate_recall_failure_summary.md
```

Candidate answer sources:

```text
heuristic title/entity/date/number/yes-no spans
Flan-T5 answer on PropRAG top-5
Flan-T5 answer on Dense top-5
Flan-T5 answer on PropRAG/Dense union top-5
Flan-T5 single-passage local answers for top-12 union docs
Flan-T5 pairwise local answers for top-5 union doc pairs
```

Reader generations were cached:

```text
data/dpathrag/cache/caps/caps_day1_answer_cache.jsonl
4000 rows
```

## Result

Final Day-1 candidate recall:

```text
rows = 200
recall@5 = 0.670
recall@10 = 0.740
recall@20 = 0.750
decision = STOP_CANDIDATE_RECALL_FAIL
```

The first version without single/pair reader local answers reached only:

```text
recall@5 = 0.595
recall@10 = 0.660
recall@20 = 0.665
```

The added local-answer generator improved recall but did not approach the gate:

```text
gate = recall@5 >= 0.85
```

## Pool Upper Bound Check

The gold answer string appears in the PropRAG/Dense union top60 text for most dev examples:

```text
answer_in_union60_text = 180/200 = 0.900
answer_in_union60_titles = 91/200 = 0.455
```

So the failure is not primarily pool absence. The answer is often in retrieved text but is not extracted or promoted into the candidate list.

## Interpretation

CAPS v1 fails before proof search:

```text
candidate answer generation is the bottleneck
```

This is different from C-CEE:

```text
C-CEE: non-oracle answer hypotheses collapsed reader counterfactual admission.
CAPS Day 1: answer hypotheses do not reach sufficient recall even before proof verification.
```

The Day-0 verifier result remains useful: template proof obligations can be verified by an NLI model. But proof verification cannot rescue missing candidate answers.

## Decision

Do not proceed to:

```text
oracle-answer proof separability
non-oracle proof-selection pilot
CAPS eval800/eval1000
```

unless a stronger candidate answer generator is introduced.

## Reopen Conditions

CAPS may be reopened if a new candidate generator reaches:

```text
recall@5 >= 0.85 on dev
```

Possible future routes:

```text
relation-template answer extraction
controlled LLM candidate list generation
high-recall answer proposal model
dataset-specific structured extraction for 2Wiki relations
```

Do not reopen by tuning proof scoring alone.
