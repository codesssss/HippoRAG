# D-PathRAG PropRAG Context Ablation

- Reader: `data/dpathrag/models/flan_t5_base_gold_k5_5k`
- Rows per config: 200

| Config | Support Recall | Support Complete | Answer EM | Answer F1 |
|---|---:|---:|---:|---:|
| rank_top5 | 0.8962 | 0.7500 | 0.4850 | 0.5426 |
| selector_ar_order | 0.9137 | 0.7800 | 0.4750 | 0.5344 |
| selector_rank_order | 0.9137 | 0.7800 | 0.4650 | 0.5229 |
| selector_gold_first | 0.9137 | 0.7800 | 0.4800 | 0.5369 |
| gold_support_only | 0.9888 | 0.9700 | 0.6300 | 0.7070 |
| gold_plus_selector_distractors | 0.9888 | 0.9700 | 0.5350 | 0.5896 |

## Interpretation

- The selector mechanism is still positive: `selector_ar_order` improves support-complete over `rank_top5` from 0.7500 to 0.7800.
- Re-sorting selector docs by retriever rank does not help. `selector_rank_order` drops to 0.5229 F1, so the PropRAG Yellow result is not mainly an AR-order artifact.
- Oracle gold-first ordering helps only slightly: 0.5344 to 0.5369 F1. Front-loading known support documents is insufficient by itself.
- The upper reference is much higher: `gold_support_only` reaches 0.7070 F1. The fine-tuned reader can answer well when distractors are removed.
- Adding selector distractors to the oracle gold set drops F1 from 0.7070 to 0.5896. This isolates a large distractor/context-consumption effect.

## Decision

The PropRAG Yellow result is primarily a distractor sensitivity / context consumption bottleneck, not a simple ordering problem. D-PathRAG should next test conservative blending or evidence-gated formatting rather than only reordering selected documents.

Recommended next variants:

1. `rank-anchor blend`: keep PropRAG rank top-2 or top-3, let D-PathRAG fill the remaining slots.
2. `support-confidence gate`: prefer selector docs only when semantic/gold-likeness margin beats the displaced rank doc.
3. `two-block prompt`: put high-confidence support block first, then label remaining docs as auxiliary candidates to reduce distractor competition.
