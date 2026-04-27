# CAPS Day-2 Oracle-Obligation Proof Separability

- Decision: `STOP_CAPS_PROOF_RANKER_FAIL`
- Rows: `200`
- Candidate recall@20: `0.84`
- Top1 accuracy all: `0.285`
- Top3 accuracy all: `0.49`
- Top1 accuracy conditional gold present: `0.339286`
- Top3 accuracy conditional gold present: `0.583333`
- MRR conditional gold present: `0.499144`
- Gold-vs-best-wrong AUC: `0.398136`
- Paired gold win rate: `0.285714`
- Mean gold proof score: `0.383771`
- Mean best wrong proof score: `0.530366`
- Answer-conditioned available rate: `0.91`
- Gold substituted rate: `0.75`
- Unique NLI pairs: `116619`

## Gate

- Proceed if all-query top1 >= 0.55, conditional top1 >= 0.55, or conditional top3 >= 0.70.
- Otherwise stop CAPS proof search.

## Per-Type Breakdown

| Type | Rows | Recall@20 | Top1 All | Top3 All | Top1 Cond | Top3 Cond | MRR Cond | Answer-Cond Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bridge_comparison | 47 | 0.978723 | 0.531915 | 0.765957 | 0.543478 | 0.782609 | 0.676916 | 0.893617 |
| comparison | 51 | 1.0 | 0.372549 | 0.627451 | 0.372549 | 0.627451 | 0.550211 | 0.745098 |
| compositional | 79 | 0.721519 | 0.126582 | 0.265823 | 0.175439 | 0.368421 | 0.328042 | 1.0 |
| inference | 23 | 0.608696 | 0.130435 | 0.391304 | 0.214286 | 0.642857 | 0.42564 | 1.0 |

## Interpretation Boundary

This is an oracle-obligation mechanism diagnostic because obligations are derived from gold 2Wiki evidence triples.
A pass would justify testing non-oracle obligation generation; it is not itself a deployable CAPS result.
