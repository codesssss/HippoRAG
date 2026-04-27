# D-PathRAG PropRAG K-Fold Eval1000 Results

Date: 2026-04-26

## Protocol

- Dataset/protocol: 2Wiki local1000 fixed PropRAG pool.
- Selector protocol: 5-fold cross-fit, fold size 200, each fold trains on the 800-query complement.
- Leakage check: train/eval qid overlap is 0 across all folds.
- Selector features: base rank/lexical features plus cached RP embedding features from `selector_embedding_proprag_local1000_rp64.npz`.
- Reader: fine-tuned `data/dpathrag/models/flan_t5_base_gold_k5_5k`.
- Reader input: top-5 documents, same reader for rank and selector.

## Selector-Level Result

| Variant | Support Recall | Support Complete | Selected Gold | Bridge Entity Recall | Selection Overlap |
|---|---:|---:|---:|---:|---:|
| PropRAG rank top-5 | 0.9028 | 0.7720 | 2.2270 | 0.9365 | 1.0000 |
| D-PathRAG selector | 0.9233 | 0.8050 | 2.2660 | 0.9416 | 0.6192 |
| Delta | +0.0205 | +0.0330 | +0.0390 | +0.0051 | -0.3808 |

Interpretation: the non-leaky eval1000 sanity confirms the mechanism-level gain. The selector is not just reproducing rank order; average top-5 overlap is 0.6192 while support-complete improves by +3.3pp.

## Reader-Level Result

| Variant | EM | F1 | Support Recall | Support Complete |
|---|---:|---:|---:|---:|
| PropRAG rank top-5 | 0.4250 | 0.4824 | 0.9028 | 0.7720 |
| D-PathRAG selector | 0.4110 | 0.4634 | 0.9233 | 0.8050 |
| Delta | -0.0140 | -0.0190 | +0.0205 | +0.0330 |

Paired bootstrap over 1000 queries:

- Support-complete delta: +3.30pp, 95% CI [+1.30pp, +5.20pp].
- Support-recall delta: +2.05pp, 95% CI [+1.20pp, +2.93pp].
- F1 delta: -1.90pp, 95% CI [-4.03pp, +0.23pp].
- EM delta: -1.40pp, 95% CI [-3.60pp, +0.70pp].

Interpretation: PropRAG Yellow is systematic at the mechanism level, while answer-level degradation is directionally negative but not significant at 95% under paired bootstrap. The robust claim is mechanism gain without reliable F1 translation, not statistically certain F1 harm.

## Hard-Negative Diagnosis

| Category | Docs | Queries | Mean Rank | q-doc Cosine | Answer In Doc | Bridge In Doc |
|---|---:|---:|---:|---:|---:|---:|
| selector_added_non_gold | 1205 | 807 | 18.0083 | 0.1895 | 0.0407 | 0.0481 |
| rank_retained_non_gold | 1529 | 827 | 3.3479 | 0.2629 | 0.0837 | 0.1262 |
| rank_removed_non_gold | 1244 | 826 | 4.3344 | 0.2128 | 0.0362 | 0.0908 |
| selector_added_gold | 76 | 76 | 9.6842 | 0.2635 | 0.5263 | 0.7632 |

Key differences:

- Selector-added non-gold docs are much deeper than rank-retained non-gold docs: mean rank 18.0 vs 3.35.
- Selector-added non-gold docs are weaker by q-doc cosine than rank-retained non-gold docs: 0.1895 vs 0.2629.
- Selector-added non-gold docs contain answer/bridge signals less often than rank-retained non-gold docs: answer 4.1% vs 8.4%, bridge 4.8% vs 12.6%.
- Selector-added gold docs are rare but high quality: 76 docs, answer-in-doc 52.6%, bridge-in-doc 76.3%.

Interpretation: the selector does recover real missed gold evidence, but it also imports many low-rank non-gold documents that are less reader-useful than PropRAG's retained non-gold context. This explains why support-complete rises while answer F1 falls.

## Decision

Current D-PathRAG status on PropRAG pool: mechanism-positive Yellow.

This is not an eval200 artifact for evidence selection. The eval1000 cross-fit result supports the three-layer bottleneck framing:

- Retrieval substrate: PropRAG rank top-5 is already strong.
- Evidence selection: D-PathRAG improves support-complete by +3.3pp.
- Reader consumption: the fine-tuned reader does not reliably translate the support-complete gain into F1; observed F1 is -1.9pp, but the paired 95% CI crosses zero.

Do not continue simple mitigation sweeps. The next method-level optimization, if pursued, should target hard-negative rejection or reader-aware selection rather than rank-anchor blending or prompt ordering.
