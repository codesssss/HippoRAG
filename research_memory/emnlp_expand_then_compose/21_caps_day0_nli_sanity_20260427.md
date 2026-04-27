# CAPS Day-0 NLI Sanity - 2026-04-27

## Context

After stopping the D-PathRAG / CEE / C-CEE line, the next candidate direction is:

```text
CAPS: Candidate-Answer Proof Search
```

The modeling change is to move the latent object from an evidence set `S` or edit `e` to an answer-conditioned proof object `(a, S)`. Instead of first choosing evidence and asking a reader for an answer, CAPS first enumerates candidate answers and then searches for a proof set that verifies each candidate.

The main risk is proof verification. Before implementing candidate generation or proof search, Day 0 checks whether a DeBERTa-style NLI verifier can score template-generated proof obligations against gold-support documents.

## Day-0 Setup

Data:

```text
cache: data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl
split: dev fold, rows 0-199
```

Obligation construction:

```text
Use existing 2Wiki evidence triples.
Example:
  ["Lothair II", "mother", "Ermengarde of Tours"]
becomes:
  "The mother of Lothair II is Ermengarde of Tours."
```

Positive pair:

```text
premise = gold-support candidate doc whose title matches the triple subject
hypothesis = template obligation
```

Negative pair:

```text
premise = same-query non-gold candidate doc
hypothesis = same template obligation
```

This uses template obligations rather than free-form LLM decomposition, deliberately avoiding the PCRS requirement-generation failure mode.

## Implementation

```text
scripts/caps_day0_nli_sanity.py
tests/dpathrag/test_caps_day0.py
reports/caps/nli_sanity_day0.md
reports/caps/nli_sanity_day0.json
```

The first attempted model name, `microsoft/deberta-v3-base-mnli`, was invalid. The actual run used:

```text
cross-encoder/nli-deberta-v3-base
```

## Result

```text
decision = PROCEED_DAY1
model = cross-encoder/nli-deberta-v3-base
pairs = 200
positive / negative = 100 / 100
rows_seen = 44
triples_seen = 109
triples_missing_subject_doc = 9
AUC = 0.903
95% CI = [0.8616, 0.9401]
paired_win_rate = 0.88
mean_positive_score = 0.590845
mean_negative_score = 0.001769
throughput = 108.5603 pairs/s
```

Gate:

```text
AUC >= 0.80 -> proceed to CAPS Day 1
0.70 <= AUC < 0.80 -> replace/calibrate verifier
AUC < 0.70 or model unavailable -> stop CAPS with this verifier
```

Day 0 passes.

## Interpretation

This result only validates the verifier substrate for template obligations. It does not yet show that CAPS can recover candidate answers or select the correct answer non-oracle.

What it does establish:

```text
Template obligation-doc verification is not immediately dead.
The NLI verifier can distinguish subject-matched gold proof docs from same-query non-gold docs on 2Wiki dev.
```

What remains unproven:

```text
candidate answer recall@5/@10/@20
oracle-answer proof separability across wrong candidates
non-oracle proof answer selection
reader F1 after proof-set evidence selection
```

## Next Gate

CAPS should continue only through Day-1 diagnostics before any full eval:

1. Candidate answer recall on dev fold.
2. Oracle-answer proof separability.
3. Non-oracle proof-selection pilot on dev200.

No full eval800/eval1000 should run until these Day-1 gates pass.
