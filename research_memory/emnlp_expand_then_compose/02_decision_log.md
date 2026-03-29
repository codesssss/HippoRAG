# Decision Log

Last updated: 2026-03-28

## 2026-03-26: Freeze the retrieval backbone

Decision:
- Use aligned `legacy_fact_graph` HippoRAG retrieval as the fixed backbone.

Reason:
- It remains the strongest validated base retriever.
- General-graph and causal-retrieval variants underperformed materially.

Consequence:
- Future paper work should not spend the main novelty budget on another retrieval-graph variant.

## 2026-03-27: Stop treating the problem as causal reasoning

Decision:
- Do not frame multi-hop QA as a causal-query routing problem.

Reason:
- Multi-hop QA bridge relations are far broader than `causes / enables / prevents`.
- The semantic intent router and causal-only extraction did not match the task structure.

Consequence:
- Negative causal results become motivation for a pivot, not an unfinished branch.

## 2026-03-28: Pivot from pointwise reranking to evidence composition

Decision:
- Reframe the problem from pointwise document ranking to setwise evidence composition.

Trigger evidence:
- Multiple pointwise-style interventions failed or hurt.
- Oracle reorder ceiling was materially lower than oracle select ceiling.
- `2Wiki` support depth split is extreme: `2-doc median depth = 3` vs `4-doc median depth = 64`.
- Oracle-select gains remain large as `K` increases.

Consequence:
- Active story becomes `Expand-then-Compose`.
- The main method target is a simple setwise selector over a widened candidate pool.

## 2026-03-28: Target EMNLP main, not theory-first venues

Decision:
- Optimize for an EMNLP main-track story: strong analysis, clean framing, simple method, solid experiments.

Reason:
- The novelty is problem reframing plus strong empirical evidence, not a deep new optimization algorithm.

Consequence:
- Prioritize: `MuSiQue` generalization, `2Wiki` bridge analysis, and one simple non-oracle baseline.
- Do not overbuild a complicated planner unless the simple method fails.
