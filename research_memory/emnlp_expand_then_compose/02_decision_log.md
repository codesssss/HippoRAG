# Decision Log

Last updated: 2026-04-22

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

## 2026-04-17: Preserve Requirement-Aware Strongest as a documented backup line

Decision:
- Keep `Requirement-Aware Strongest` as a documented backup method line.
- Do not promote it over the current `Expand-then-Compose` mainline.
- Do not let it displace `PCRS-RAG` as the main requirement-first development branch.

Reason:
- The migrated strongest sidecar plus `clean GBC` now forms a coherent, runnable graph-side reserve path.
- The current evidence is mixed:
  - positive smoke-scale signal on `2Wiki`
  - stabilizing but still sub-baseline end-to-end behavior on `HotpotQA`
- This makes it valuable as a backup idea, but not yet honest to present as the new default story.

Consequence:
- Keep the line alive as a recoverable option if the current mainline stalls or if a graph-specific backup story is needed.
- Document the method, current evidence, and promotion gates in:
  - `research_memory/emnlp_expand_then_compose/10_requirement_aware_strongest_backup.md`
  - `outputs_step0_general_hotpotqa/eval_reports/strongest_gbc_e2e_audit_hotpotqa20_20260416.md`

## 2026-04-22: Promote DtC as the active fixed-pool composition line

Decision:
- Treat `DtC-Embed` as the active method branch for `Expand-then-Compose`.
- Keep the canonical method as fixed-pool post-retrieval composition, not open iterative retrieval.
- Use NV-Embed-v2 without instruction prefixes for the aligned paper protocol.
- Do not promote hard-crossing or a fixed depth cutoff as the main method.
- Next implementation target: rank-regularized demand coverage.

Reason:
- Full1000 NV non-instruction DtC is positive on all three datasets:
  - `2Wiki`: `F1 +0.0082`
  - `HotpotQA`: `F1 +0.0175`
  - `MuSiQue`: `F1 +0.0245`
- MMR/DPP structure-blind diversity does not recover the oracle gap.
- Hard-crossing ablation removes too many wins and does not meaningfully reduce losses.
- Depth analysis shows losses disproportionately pull from deeper pool positions, but a fixed `D` gate is too brittle for a paper-facing method.

Consequence:
- Mainline method work should modify the DtC objective, not add another hard per-candidate gate.
- Implement `rank_weight` as a rank prior inside DtC scoring and sweep on pilot100 before full1000.
- Keep all claims tied to same-pool, same-reader evidence composition.
- Canonical note:
  - `research_memory/emnlp_expand_then_compose/14_dtc_nv_full1000_rank_prior_20260422.md`
