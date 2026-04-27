# CAPS Day-2 Oracle-Obligation Proof-Ranker Failure - 2026-04-27

## Context

CAPS Day 1.5 improved broad candidate recall but failed the strict candidate-rank gate:

```text
LLM + v1 reader + string extraction Recall@5  = 0.635
LLM + v1 reader + string extraction Recall@10 = 0.665
LLM + v1 reader + string extraction Recall@20 = 0.840
```

The revised hypothesis was that `Recall@20 = 0.840` might still be sufficient if the proof scorer can act as the final candidate ranker:

```text
candidate generator proposes broad top-20
NLI proof scoring reranks candidates
gold candidate should rise to top-1/top-3 if CAPS proof signal aggregates cleanly
```

This Day-2 experiment tested that hypothesis directly.

## Diagnostic Boundary

This is an oracle-obligation mechanism diagnostic, not a deployable CAPS method.

The obligations are derived from gold 2Wiki evidence triples and then answer-conditioned by substituting each candidate answer into the gold answer slot when possible. Therefore:

```text
PASS -> proof-ranker mechanism may be viable; next test non-oracle obligation generation
FAIL -> CAPS fails even under oracle/template obligations; stop the line
```

## Implementation

```text
script: scripts/caps_day2_proof_separability.py
tests:  tests/dpathrag/test_caps_day2.py
report: reports/caps/caps_day2_proof_separability.md
json:   reports/caps/caps_day2_proof_separability.json
rows:   reports/caps/caps_day2_proof_separability.rows.jsonl
```

Validated tests:

```text
pytest tests/dpathrag/test_caps_day2.py
4 passed

pytest tests/dpathrag/test_caps_day0.py tests/dpathrag/test_caps_day1.py tests/dpathrag/test_caps_day1_5.py tests/dpathrag/test_caps_day2.py
19 passed
```

Run command:

```text
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=3 \
.venv-hipporag/bin/python scripts/caps_day2_proof_separability.py \
  --output_dir reports/caps
```

The run used the local cached `cross-encoder/nli-deberta-v3-base` verifier.

## Configuration

```text
dev fold: first 200 2Wiki examples
candidate source: Day-1.5 union candidates
candidate cap: 20
proof docs: PropRAG/Dense union top30
answer prior: uniform
obligations: oracle/template answer-conditioned obligations from 2Wiki evidence triples
NLI model: cross-encoder/nli-deberta-v3-base
unique NLI pairs: 116,619
candidate scoring tasks: 4,000
```

## Gate

Proceed if any condition passes:

```text
top1_accuracy_all >= 0.55
OR top1_accuracy_cond_gold_present >= 0.55
OR top3_accuracy_cond_gold_present >= 0.70
```

Otherwise stop CAPS proof search.

## Result

Overall:

```text
Decision: STOP_CAPS_PROOF_RANKER_FAIL
Rows: 200
Candidate recall@20: 0.840
Top1 accuracy all: 0.285
Top3 accuracy all: 0.490
Top1 accuracy conditional gold present: 0.339286
Top3 accuracy conditional gold present: 0.583333
MRR conditional gold present: 0.499144
Gold-vs-best-wrong AUC: 0.398136
Paired gold win rate: 0.285714
Mean gold proof score: 0.383771
Mean best wrong proof score: 0.530366
Answer-conditioned available rate: 0.910
Gold substituted rate: 0.750
```

Per type:

| Type | Rows | Recall@20 | Top1 All | Top3 All | Top1 Cond | Top3 Cond | MRR Cond |
|---|---:|---:|---:|---:|---:|---:|---:|
| bridge_comparison | 47 | 0.978723 | 0.531915 | 0.765957 | 0.543478 | 0.782609 | 0.676916 |
| comparison | 51 | 1.000000 | 0.372549 | 0.627451 | 0.372549 | 0.627451 | 0.550211 |
| compositional | 79 | 0.721519 | 0.126582 | 0.265823 | 0.175439 | 0.368421 | 0.328042 |
| inference | 23 | 0.608696 | 0.130435 | 0.391304 | 0.214286 | 0.642857 | 0.425640 |

## Interpretation

The proof ranker fails the intended mechanism test.

The critical failure is not simply candidate absence:

```text
candidate recall@20 = 0.840
conditional rows with gold present = high enough to test reranking
```

The critical failure is that NLI proof aggregation systematically gives higher scores to wrong candidates:

```text
mean gold proof score = 0.383771
mean best wrong proof score = 0.530366
gold-vs-best-wrong AUC = 0.398136
paired gold win rate = 0.285714
```

That is below random ranking at the gold-vs-best-wrong level. Therefore the strong Day-0 obligation-level NLI signal:

```text
Day-0 NLI verifier AUC = 0.903
```

does not aggregate into answer-level proof discrimination under broad candidate competition.

## What Actually Worked

`bridge_comparison` is the only partially positive slice:

```text
Recall@20 = 0.978723
Top1 conditional = 0.543478
Top3 conditional = 0.782609
```

This is close to the gate and suggests that simple comparison/bridge-comparison templates can sometimes be answer-reranked by proof scoring.

However, the main 2Wiki mass is not there:

```text
compositional rows = 79
compositional top1 conditional = 0.175439
compositional top3 conditional = 0.368421
inference rows = 23
inference top1 conditional = 0.214286
```

CAPS cannot be promoted on a narrow type slice that does not solve compositional and inference questions.

## Failure Conclusion

CAPS should stop for the current paper cycle.

The sequence of CAPS diagnostics is now:

```text
Day-0 NLI sanity:
  PASS, obligation-level verifier AUC = 0.903

Day-1 candidate recall:
  FAIL, Recall@5/10/20 = 0.670 / 0.740 / 0.750

Day-1.5 LLM + string candidate rescue:
  FAIL strict rank gate, Recall@5/10/20 = 0.635 / 0.665 / 0.840

Day-2 oracle-obligation proof ranker:
  FAIL, conditional Top1 = 0.339, conditional Top3 = 0.583, gold-vs-best-wrong AUC = 0.398
```

This changes the diagnosis from only "candidate generation is weak" to a stronger method-level conclusion:

```text
Even when the gold answer is present in the broad candidate set and obligations are oracle/template-derived,
noisy-OR NLI proof scoring does not reliably rerank the gold answer above plausible wrong candidates.
```

## Paper-Ready Negative Finding

Candidate-Answer Proof Search was tested as a training-free alternative to evidence-editing methods by making the answer an explicit latent variable and using NLI-based proof coverage as the answer scorer. Although obligation-level NLI discrimination was strong in isolation (`AUC=0.903`), the signal did not compose into answer-level proof ranking: on a 200-query dev fold with broad top-20 answer candidates, oracle/template proof scoring achieved only `0.339` conditional top-1 accuracy and assigned higher scores to the best wrong candidate than to the gold candidate on average (`0.530` vs `0.384`, gold-vs-best-wrong `AUC=0.398`). This indicates that local entailment verification plus noisy-OR aggregation is insufficient for answer-conditioned proof search under hard candidate competition.

## Decision

```text
STOP CAPS
Do not proceed to Day-3 non-oracle obligation generation
Do not tune answer priors, proof thresholds, or proof set cover on this line
```

Only reopen CAPS if the core proof object changes, not by local tuning. Plausible reopen conditions:

```text
supervised/in-domain answer proposal plus supervised proof verifier
structured 2Wiki relation parser with explicit variable binding
LLM verifier with calibrated pairwise answer comparison and cost budget
dataset-specific symbolic proof extraction rather than generic NLI noisy-OR
```

For the current paper, CAPS belongs in the negative diagnostic section together with BSGS, D-PathRAG/CEE, and C-CEE.
