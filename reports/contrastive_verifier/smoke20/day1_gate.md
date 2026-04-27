# Answer-Contrastive Verifier Day-1 Gate

- Decision: `FAIL_ANSWER_CONTRASTIVE_VERIFIER`
- Candidate rows: `400`
- Queries: `20`
- Gold-present queries: `17`
- Feature mode: `cached candidate/proof features only`
- No new LLM calls: `True`
- No DeBERTa/Qwen fine-tuning: `true`
- Best supervised model: `hist_gradient_boosting`

## Model Comparison

| Model | Cand AUC | Gold-vs-best-wrong AUC | 95% CI | Top1 Cond | Top3 Cond | MRR Cond | Top1 All |
|---|---:|---:|---:|---:|---:|---:|---:|
| proof_score | 0.559668 | 0.275087 | [0.16263, 0.373702] | 0.235294 | 0.529412 | 0.411851 | 0.2 |
| logistic | 0.756259 | 0.224913 | [0.096886, 0.377163] | 0.235294 | 0.470588 | 0.422899 | 0.2 |
| hist_gradient_boosting | 0.834434 | 0.429066 | [0.268166, 0.532872] | 0.411765 | 0.588235 | 0.570614 | 0.35 |

## Gate

- Green if best supervised gold-vs-best-wrong AUC >= 0.75, CI lower >= 0.70, and conditional top1 >= 0.55.
- Yellow if AUC >= 0.70 or conditional top3 >= 0.70.
- Red otherwise; write as the sixth negative diagnostic and stop method exploration.

## Best Supervised Per-Type Breakdown

| Type | Queries | Recall@20 | Top1 Cond | Top3 Cond | MRR Cond | Gold-vs-Best-Wrong AUC |
|---|---:|---:|---:|---:|---:|---:|
| bridge_comparison | 5 | 1.0 | 0.6 | 0.6 | 0.69 | 0.54 |
| comparison | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 0.75 |
| compositional | 10 | 0.7 | 0.0 | 0.428571 | 0.308503 | 0.204082 |
| inference | 1 | 1.0 | 0.0 | 0.0 | 0.090909 | 0.0 |

## Interpretation

This v0 tests whether cached answer/proof features contain enough residual signal for answer-level ranking.
It is not an end-to-end neural verifier because the model only consumes scalar candidate/proof/source features under GroupKFold by qid.
