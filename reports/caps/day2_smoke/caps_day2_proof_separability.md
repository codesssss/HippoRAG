# CAPS Day-2 Oracle-Obligation Proof Separability

- Decision: `STOP_CAPS_PROOF_RANKER_FAIL`
- Rows: `5`
- Candidate recall@20: `0.8`
- Top1 accuracy all: `0.2`
- Top3 accuracy all: `0.4`
- Top1 accuracy conditional gold present: `0.25`
- Top3 accuracy conditional gold present: `0.5`
- MRR conditional gold present: `0.45`
- Gold-vs-best-wrong AUC: `0.28125`
- Paired gold win rate: `0.0`
- Mean gold proof score: `0.25192`
- Mean best wrong proof score: `0.488601`
- Answer-conditioned available rate: `0.8`
- Gold substituted rate: `0.6`
- Unique NLI pairs: `2556`

## Gate

- Proceed if all-query top1 >= 0.55, conditional top1 >= 0.55, or conditional top3 >= 0.70.
- Otherwise stop CAPS proof search.

## Per-Type Breakdown

| Type | Rows | Recall@20 | Top1 All | Top3 All | Top1 Cond | Top3 Cond | MRR Cond | Answer-Cond Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bridge_comparison | 1 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 | 1.0 |
| comparison | 2 | 1.0 | 0.5 | 1.0 | 0.5 | 1.0 | 0.75 | 0.5 |
| compositional | 2 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 1.0 |

## Interpretation Boundary

This is an oracle-obligation mechanism diagnostic because obligations are derived from gold 2Wiki evidence triples.
A pass would justify testing non-oracle obligation generation; it is not itself a deployable CAPS result.
