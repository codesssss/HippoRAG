# ETV2 Channel-Direct MuSiQue Limit100 Findings

Date: 2026-05-09

Setup:

- Dataset: MuSiQue, first 100 queries
- Index: reused fresh ET index from `run_logs/evidence_transition_qwen8b_nv2_limit100_musique_20260509`
- LLM/index name: `qwen3-8b-train`
- Embedding: `nvidia/NV-Embed-v2`
- ET v1 runner: `query_grounded_sto_graph_native`
- ETV2 diagnostic runner: `evidence_transition_v2`

## Result

ETV2 channel-direct is a failed readout ablation. It should not be used as a paper-facing mainline.

| Method | R@5 | all-gold@5 | mean certified docs@5 |
| --- | ---: | ---: | ---: |
| ET v1 | 0.7033 | 0.3800 | 2.46 |
| ETV2 channel-direct | 0.4825 | 0.1900 | 4.11 |

Per-depth retrieval:

| Gold docs | n | ET v1 R@5 | ET v1 all@5 | ETV2 R@5 | ETV2 all@5 | Delta R@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 48 | 0.8333 | 0.6667 | 0.6146 | 0.3333 | -0.2188 |
| 3 | 30 | 0.6778 | 0.2000 | 0.4000 | 0.1000 | -0.2778 |
| 4 | 22 | 0.4545 | 0.0000 | 0.3068 | 0.0000 | -0.1477 |

Per-query change:

| Outcome | Count |
| --- | ---: |
| ETV2 improves hit count | 1 |
| ETV2 worsens hit count | 52 |
| Same hit count | 47 |

## Slate Audit

The failure is not only a reader-top5 problem. The channel exposure slate also does not improve early support exposure over the existing candidate/source-prior order.

Overall:

| Ordering | K | R@K | all-gold@K |
| --- | ---: | ---: | ---: |
| candidate/source-prior | 5 | 0.6600 | 0.3300 |
| candidate/source-prior | 10 | 0.7567 | 0.4500 |
| candidate/source-prior | 20 | 0.8192 | 0.5600 |
| candidate/source-prior | 50 | 0.8608 | 0.6400 |
| ETV2 channel slate | 5 | 0.4825 | 0.1900 |
| ETV2 channel slate | 10 | 0.6225 | 0.3300 |
| ETV2 channel slate | 20 | 0.7100 | 0.4300 |
| ETV2 channel slate | 50 | 0.8383 | 0.6200 |

4-doc subset:

| Ordering | K | R@K | all-gold@K |
| --- | ---: | ---: | ---: |
| candidate/source-prior | 5 | 0.4318 | 0.0000 |
| candidate/source-prior | 10 | 0.5682 | 0.0000 |
| candidate/source-prior | 20 | 0.6932 | 0.1818 |
| candidate/source-prior | 50 | 0.7614 | 0.3182 |
| ETV2 channel slate | 5 | 0.3068 | 0.0000 |
| ETV2 channel slate | 10 | 0.4432 | 0.0000 |
| ETV2 channel slate | 20 | 0.5455 | 0.0455 |
| ETV2 channel slate | 50 | 0.7500 | 0.3182 |

## Interpretation

The channel-direct readout promotes graph-neighbor channels before the source-prior channel. This demotes dense/source-prior supports that ET v1 already had in the first five passages. Many failures are broad or hub-like graph transitions: geography/state/country pages, repeated high-frequency titles, or topical-but-not-supporting transition neighbors.

Concrete symptom:

- `source_prior` often contains the correct gold supports early.
- `sentence_grounded_transition`, `title_role_grounding`, and `role_bridge` inject plausible but wrong branch documents before source-prior evidence.
- As a result, selected positions move much deeper in the candidate order.

4-doc selected-position diagnostic:

| Method | Mean selected candidate position | Selected slots with position >= 20 |
| --- | ---: | ---: |
| ET v1 | 6.16 | 12 / 110 |
| ETV2 channel-direct | 30.07 | 62 / 110 |

## Decision

Keep this implementation as a diagnostic/failed ablation only.

Do not run reader QA for ETV2 channel-direct: retrieval-side R@5 and all-gold@5 are already substantially worse than ET v1.

## Next Direction

The clean next direction is not channel-first top5 readout. It should be:

1. Preserve ET v1 source-prior order as the stable reader-presentation prior.
2. Use evidence channels as annotations or candidate-slate expansion, not as direct top5 ranking.
3. If adding selection, use one explicit membership decision over a bounded slate, then present selected documents in source-prior/candidate order.
4. Treat any LLM membership prompt as diagnostic/ablation until validated across datasets.

