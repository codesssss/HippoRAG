# Answer-Contrastive Verifier Day-1 Gate

- Decision: `PARTIAL_TOPK_RECOVERY_NOT_MAINLINE`
- Candidate rows: `4000`
- Queries: `200`
- Gold-present queries: `168`
- Feature mode: `cached candidate/proof features only`
- No new LLM calls: `true`
- No DeBERTa/Qwen fine-tuning: `true`
- Best supervised model: `hist_gradient_boosting`

## Model Comparison

| Model | Cand AUC | Gold-vs-best-wrong AUC | 95% CI | Top1 Cond | Top3 Cond | MRR Cond | Top1 All |
|---|---:|---:|---:|---:|---:|---:|---:|
| proof_score | 0.636895 | 0.398136 | [0.366337, 0.425542] | 0.339286 | 0.583333 | 0.498152 | 0.285 |
| logistic | 0.936787 | 0.55141 | [0.500673, 0.607355] | 0.511905 | 0.821429 | 0.680145 | 0.43 |
| hist_gradient_boosting | 0.943125 | 0.570826 | [0.514952, 0.624433] | 0.595238 | 0.827381 | 0.729908 | 0.5 |
| gradient_boosting | 0.938759 | 0.530967 | [0.477785, 0.583847] | 0.517857 | 0.839286 | 0.693229 | 0.435 |

## Gate

- Green if best supervised gold-vs-best-wrong AUC >= 0.75, CI lower >= 0.70, and conditional top1 >= 0.55.
- Yellow if AUC >= 0.70 or conditional top3 >= 0.70; treat as partial top-k recovery, not a mainline method.
- Red otherwise; write as the sixth negative diagnostic and stop method exploration.

## Best Supervised Per-Type Breakdown

| Type | Queries | Recall@20 | Top1 Cond | Top3 Cond | MRR Cond | Gold-vs-Best-Wrong AUC |
|---|---:|---:|---:|---:|---:|---:|
| bridge_comparison | 47 | 0.978723 | 0.847826 | 1.0 | 0.92029 | 0.759452 |
| comparison | 51 | 1.0 | 0.72549 | 0.980392 | 0.857843 | 0.7401 |
| compositional | 79 | 0.721519 | 0.350877 | 0.666667 | 0.542119 | 0.322869 |
| inference | 23 | 0.608696 | 0.285714 | 0.357143 | 0.402887 | 0.229592 |

## Interpretation

This v0 tests whether cached answer/proof features contain enough residual signal for answer-level ranking.
It is not an end-to-end neural verifier because the model only consumes scalar candidate/proof/source features under GroupKFold by qid.
