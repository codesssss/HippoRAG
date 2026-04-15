# Decision Log

Last updated: 2026-04-14

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

## 2026-04-02: Freeze the current simple line and open a parallel PCRS-RAG V1 branch

Decision:
- Keep the current paper-facing mainline as the simple selector stack:
  - `bridge_beam + set_closure + pathcore_guard + reserve3 + dedup`
- Do not silently replace that line with the new requirement-aware method.
- Open a separate experimental branch, `feature/pcrs-rag-v1`, around a new selector:
  - `requirement_beam`

Reason:
- The current simple line is still the strongest validated practical story.
- The new branch changes the objective itself, not just a search hyperparameter:
  - positive requirement support
  - counterfactual leakage
  - Pareto beam selection
- That makes it a real method branch, not a safe micro-tweak to merge into the paper line before evidence exists.

Consequence:
- Treat `PCRS-RAG V1` as a parallel high-information branch.
- Require explicit promotion gates before it can replace the current simple line:
  - cache quality gate
  - oracle selector gate
  - learned matcher gate
- Keep the paper story honest:
  - the current mainline remains the simple bridge-aware composition story
  - the new branch is an attempt to fix the `MuSiQue` objective mismatch rather than a confirmed replacement
- Record implementation and operating details in:
  - `research_memory/emnlp_expand_then_compose/07_pcrs_rag_v1.md`

## 2026-04-02: Use PCRS-RAG V1 as the default development branch, while keeping the paper mainline frozen

Decision:
- Treat `feature/pcrs-rag-v1` as the default code development branch for ongoing selector work.
- Keep the paper-facing method definition frozen on the current simple line:
  - `bridge_beam + set_closure + pathcore_guard + reserve3 + dedup`

Reason:
- The old simple line is already committed and stable enough to serve as the paper-facing baseline.
- Ongoing work now centers on requirement quality, counterfactual construction, and selector diagnostics, all of which belong to the new branch.
- Keeping engineering on the branch avoids repeated branch hopping without forcing an early paper-story promotion.

Consequence:
- New selector-side implementation and diagnostics should land on `feature/pcrs-rag-v1` by default.
- Paper claims must continue to describe the simple `pathcore_guard` stack as the current mainline until the PCRS branch clears promotion gates.
- Branch operating status and diagnostics should be documented in:
  - `research_memory/emnlp_expand_then_compose/07_pcrs_rag_v1.md`
  - `research_memory/emnlp_expand_then_compose/03_experiment_board.md`

## 2026-04-10: Treat NQ/PopQA bridge failure as threshold-gated by default, not query-source-gated

Decision:
- For the `nq/popqa` width-matched bridge diagnosis, interpret the default no-gain result as a threshold-gated failure mode.
- Do not describe `setwise_query_entity_source` as the main blocker on these datasets.
- Do not promote `expand_min_structure_score = 0.0` to the new default from this evidence alone.

Reason:
- On both datasets, the default bridge line at `expand_min_structure_score = 0.35` appends no documents.
- Switching `query_entity_source` from `seed` to `question` does not change that failure mode.
- Lowering the threshold to `0.0` fully activates append behavior on both datasets.
- After activation, outcomes diverge:
  - `nq`: no QA gain
  - `popqa`: small positive QA gain

Consequence:
- The right diagnosis is:
  - default bridge is threshold-gated on `nq/popqa`
  - query-source choice is secondary
  - answer-utility after unlocking remains dataset-dependent
- Future diagnostic reruns on `nq/popqa` should include at least one `threshold = 0.0` point.
- Persist the detailed evidence in:
  - `research_memory/emnlp_expand_then_compose/11_nq_popqa_bridge_threshold_diagnosis_20260410.md`

## 2026-04-14: Keep actionized repair as analysis evidence; stop the current NoisyOR controller line

Decision:
- Keep the paper-facing method frozen as:
  - `bridge-aware Expand + CE-centered Assemble`
- Keep `action_swap_v0_dryrun` and `action_swap_v0_judge` as analysis/intervention evidence rather than promoting them to the main method.
- Stop iterating on the current zero-shot `action_swap_noisyor_{flat,dep}` controller family.

Reason:
- The actionized `keep/swap` implementation already showed that changing the execution unit can change outcomes.
- Current vs relaxed oracle one-swap ceilings showed a useful decomposition:
  - `MuSiQue`: legality recall is a major bottleneck
  - `2Wiki`: selection under drift is the main bottleneck
- The offline `noisyor` audit then failed the continuation gate:
  - oracle-positive actions are mostly scored as nonpositive
  - dryrun/judge positive swaps are not reliably lifted
  - dependency graph barely changes action ranking

Consequence:
- The `action_swap` line remains valuable, but as diagnosis and intervention evidence:
  - execution-unit repair
  - oracle legality vs policy decomposition
  - failed zero-shot utility-controller case
- Do not spend more near-term effort on:
  - margin tuning
  - more zero-shot aggregation variants
  - expanding online reader experiments for this controller family
- Record the frozen status in:
  - `research_memory/emnlp_expand_then_compose/12_actionized_interface_repair_status_20260414.md`
