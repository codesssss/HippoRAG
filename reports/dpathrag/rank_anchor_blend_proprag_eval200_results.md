# D-PathRAG PropRAG Context Ablation

- Reader: `data/dpathrag/models/flan_t5_base_gold_k5_5k`
- Rows per config: 200

| Config | Support Recall | Support Complete | Answer EM | Answer F1 | Anchor Docs | Selector Added | Added Gold | Rank Gold Removed | Net Gold Gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank_anchor_m0 | 0.9137 | 0.7800 | 0.4750 | 0.5344 | 0.0000 | 5.0000 | 2.2350 | 0.0400 | 2.1950 |
| rank_anchor_m1 | 0.9137 | 0.7800 | 0.4750 | 0.5344 | 1.0000 | 4.0000 | 1.2850 | 0.0400 | 1.2450 |
| rank_anchor_m2 | 0.9137 | 0.7800 | 0.4700 | 0.5319 | 2.0000 | 3.0000 | 0.5700 | 0.0400 | 0.5300 |
| rank_anchor_m3 | 0.9125 | 0.7750 | 0.4800 | 0.5404 | 3.0000 | 2.0000 | 0.2700 | 0.0450 | 0.2250 |
| rank_anchor_m4 | 0.9100 | 0.7750 | 0.4800 | 0.5346 | 4.0000 | 1.0000 | 0.1150 | 0.0400 | 0.0750 |
| rank_anchor_m5 | 0.8962 | 0.7500 | 0.4850 | 0.5426 | 5.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation

- Sanity check passes: `rank_anchor_m5` exactly matches the PropRAG rank top-5 baseline (`support_complete=0.7500`, `F1=0.5426`).
- Rank anchoring reduces the unrestricted selector's distractor damage only partially. The best blend is `m=3` with `F1=0.5404`, close to rank top-5 but still `-0.22pp` below it.
- Support metrics remain better than rank for all `m<5`, but answer F1 does not exceed rank. This confirms that higher support-complete alone is not sufficient under the current reader/context format.
- `m=0..2` all keep `support_complete=0.7800`, but F1 falls as more selector-added docs are used. The selector adds gold evidence on average, yet the added documents are not reader-friendly enough to improve answer F1.
- Because PropRAG does not beat rank for any `m`, there is no reason to run Dense cross-pool validation for a candidate `m*` yet. The blend sweep is diagnostic rather than a new main-method win.

## Decision

Rank-anchor blending does not turn the PropRAG Yellow case Green. The bottleneck is stricter than simple unrestricted replacement: selector-added evidence improves support coverage, but the current reader still prefers the original PropRAG top-5 context.

Next priority should be a `support-confidence gate`: allow selector replacement only when a candidate has a clear semantic/support margin over the displaced rank document. If that also fails, keep D-PathRAG as a mechanism-positive but answer-F1-limited diagnostic on strong PropRAG pools, while using Dense-pool Green as the positive method signal.
