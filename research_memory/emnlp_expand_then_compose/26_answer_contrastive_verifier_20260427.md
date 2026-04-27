# Answer-Contrastive Verifier v0 Diagnostic

Date: 2026-04-27

## Goal

Test whether the CAPS Day-2 residual failure is recoverable by a lightweight supervised answer-level ranker over existing cached candidate/proof features.

Constraints:

- No new LLM calls.
- No DeBERTa/Qwen fine-tuning.
- Use cached CAPS Day-1.5 union top-20 answer candidates.
- Use CAPS Day-2 oracle/template proof-score construction.
- Evaluate with GroupKFold by `qid`.

## Implementation

- Script: `scripts/run_answer_contrastive_verifier.py`
- Tests: `tests/dpathrag/test_answer_contrastive_verifier.py`
- Report: `reports/contrastive_verifier/day1_gate.md`
- JSON: `reports/contrastive_verifier/day1_gate.json`
- Candidate rows: `reports/contrastive_verifier/candidate_rows.jsonl`
- Predictions: `reports/contrastive_verifier/predictions.jsonl`

Candidate-level construction:

- 200 dev queries.
- 20 candidates per query.
- 4000 candidate rows.
- 168/200 queries have a gold answer in top-20.
- 116619 unique NLI pairs reused from the CAPS Day-2 proof-scoring protocol.

Features:

- CAPS proof score.
- Candidate generator score/rank/source-count.
- Source indicators: LLM candidate list, reader contexts, title/date/number/entity spans, yes/no prior.
- Simple answer-type and lexical features.

Models:

- Proof-score baseline.
- Logistic regression.
- HistGradientBoosting.
- GradientBoosting.

## Results

| Model | Candidate AUC | Gold-vs-best-wrong AUC | 95% CI | Top1 Cond | Top3 Cond | MRR Cond | Top1 All |
|---|---:|---:|---:|---:|---:|---:|---:|
| proof_score | 0.636895 | 0.398136 | [0.366337, 0.425542] | 0.339286 | 0.583333 | 0.498152 | 0.285 |
| logistic | 0.936787 | 0.551410 | [0.500673, 0.607355] | 0.511905 | 0.821429 | 0.680145 | 0.430 |
| hist_gradient_boosting | 0.943125 | 0.570826 | [0.514952, 0.624433] | 0.595238 | 0.827381 | 0.729908 | 0.500 |
| gradient_boosting | 0.938759 | 0.530967 | [0.477785, 0.583847] | 0.517857 | 0.839286 | 0.693229 | 0.435 |

Decision: `PARTIAL_TOPK_RECOVERY_NOT_MAINLINE`.

The best model is `hist_gradient_boosting`. It improves conditional top1 from `0.339286` to `0.595238` and top3 from `0.583333` to `0.827381`, but the hard metric remains weak: gold-vs-best-wrong AUC is only `0.570826` with 95% CI `[0.514952, 0.624433]`.

## Per-Type Pattern

Best model by type:

| Type | Queries | Recall@20 | Top1 Cond | Top3 Cond | MRR Cond | Gold-vs-Best-Wrong AUC |
|---|---:|---:|---:|---:|---:|---:|
| bridge_comparison | 47 | 0.978723 | 0.847826 | 1.000000 | 0.920290 | 0.759452 |
| comparison | 51 | 1.000000 | 0.725490 | 0.980392 | 0.857843 | 0.740100 |
| compositional | 79 | 0.721519 | 0.350877 | 0.666667 | 0.542119 | 0.322869 |
| inference | 23 | 0.608696 | 0.285714 | 0.357143 | 0.402887 | 0.229592 |

The model learns useful shallow/type/source shortcuts for comparison-style questions, but fails on compositional and inference questions. This explains the high candidate-level AUC and top-k recovery: the ranker is good at separating many easy negatives, but not the hardest same-query best wrong candidates.

## Interpretation

This is not a green method result.

The supervised feature ranker proves that cached candidate/proof features contain some recoverable signal, because conditional top1 improves by about 25.6 points over proof-score ranking. However, the low gold-vs-best-wrong AUC means the method does not solve the actual residual problem: discriminating the gold answer from the strongest wrong answer in the same query.

Mechanism diagnosis:

- Local NLI proof score is anti-helpful in the hardest cases.
- Scalar candidate/source features can recover top-k placement, mostly through answer-type/source priors.
- Those features do not create robust answer-level separability for compositional/inference questions.
- A stronger semantic verifier might help, but that becomes a new learned neural method line rather than a cheap v0 recovery of CAPS.

## Paper Use

Use as a sixth diagnostic, not as a main method:

> A lightweight supervised answer-contrastive verifier over cached CAPS proof and candidate-source features partially recovers top-k ranking (conditional top1 0.339 to 0.595), but fails to produce clean gold-vs-best-wrong separability (AUC 0.571, 95% CI upper 0.624). This suggests that the residual answer-level discrimination problem is not solved by scalar proof/source features even with supervision; the remaining errors require richer semantic modeling or a different proof object.

## Next Action

Stop method exploration for the current cycle and move to paper writing around:

- DAEC as the positive method floor.
- BSGS, D-PathRAG/CEE/C-CEE, CAPS, CPAG/RRF, and Answer-Contrastive as mechanism-specific negative diagnostics.
