# Decision Log

Last updated: 2026-04-27

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

## 2026-04-23: Demote demand gate to ablation and start SC-SER

Decision:
- Do not promote hard demand-gated DtC as the main paper method.
- Keep demand gate as a diagnostic / negative ablation.
- Start `SC-SER` as the next method line: sufficiency-calibrated selective evidence repair.

Reason:
- Pilot100 demand-gate sweep shows consistent over-abstention:
  - `2Wiki`: best gate `F1 +0.0382` vs no-gate `+0.0456`
  - `HotpotQA`: best gate `F1 +0.0050` vs no-gate `+0.0310`
  - `MuSiQue`: best gate `F1 +0.0034` vs no-gate `+0.0397`
- The hard gate reduces some losses but sacrifices too many wins.
- This supports a softer repair objective rather than an if-else preserve policy.

Consequence:
- Treat fixed-pool composition as conservative repair from baseline top-5.
- Use residual-demand coverage and a sufficiency-calibrated baseline preservation term.
- Optimize by warm-start 1-swap and stop when no positive repair exists.
- Canonical note:
  - `research_memory/emnlp_expand_then_compose/15_demand_gate_ablation_20260423.md`

## 2026-04-24: Promote DAEC as the frozen positive floor

Decision:
- Treat `DAEC/DtC` fixed-pool evidence composition as the current positive method floor.
- Keep the method framing as retriever-agnostic post-retrieval composition over a fixed top-100 pool.
- Treat dependency binding as the strongest supported component.
- Do not overclaim rank prior or repair typing as necessary components.

Reason:
- Layer-1 full1000 runs are positive across both PropRAG top100 and dense top100 pools.
- All six tested `(dataset, pool)` F1 gains have bootstrap confidence intervals above zero.
- Targeted `nobinding` ablations reduce gains across tested dataset/pool settings.
- Generic diversity, QBF, graph belief propagation, free path selection, proof ranking, and agreement closure all failed or stayed diagnostic-only.

Consequence:
- DAEC is the stable baseline for future residual-route experiments.
- New branches must beat DAEC without damaging support_complete or importing hard negatives.
- Canonical notes:
  - `research_memory/emnlp_expand_then_compose/16_layer1_retriever_agnostic_composition_20260424.md`
  - `research_memory/emnlp_expand_then_compose/17_layer1_followup_taxonomy_significance_20260424.md`

## 2026-04-27: Stop DAEC-ALR Step-1 as currently formulated

Decision:
- Stop `DAEC-ALR Step-1` single-edit repair.
- Keep Day-0 reader self-consistency as a context-stability diagnostic only.
- Do not tune the inverse-entropy margin, skip threshold, `K_edit`, or perturbation set.
- Do not launch active retrieval on top of the current admission rule.

Reason:
- Day-0 consistency signal passed, but edit admission failed on the pre-registered 2Wiki eval200 gate.
- Valid 64-token run matched Day-0 no-edit baseline: EM/F1 `0.4900 / 0.5881`.
- Main `double_gate_skip` dropped F1 from `0.5881` to `0.5757`.
- support_complete dropped from `0.8550` to `0.8150`.
- Edited-subset F1 dropped from `0.2910` to `0.2343`.
- Flip balance failed: wrong-to-correct `3`, correct-to-wrong `5`.
- Hard-negative import failed: added gold/non-gold `4/42`, ratio `10.50`.

Consequence:
- Reader self-consistency should not be used as a local edit-admission utility in the current DAEC residual route.
- Reopen only if the exact DAEC embedding scorer is reconstructed for arbitrary candidate edits, or if the admission object changes materially.
- Canonical notes:
  - `research_memory/emnlp_expand_then_compose/30_daec_alr_step1_plan.md`
  - `research_memory/emnlp_expand_then_compose/31_daec_alr_step1_single_edit_failure_20260427.md`

## 2026-04-27: Audit NREV before final stop

Decision:
- Do not launch the full `NREV-GraphRAG` fixed-pool prototype.
- Do not launch open-pool NREV rescue.
- Do not tune thresholds around the current destructive-null construction.
- After audit, allow at most one bounded T-minus redesign if this line is reopened.

Reason:
- Full NREV failed the pre-registered Day-0 gate on `2Wiki` first-100 gold-vs-plausible-wrong pairs.
- Overall full NREV AUC was `0.6042`, with 95% CI `[0.5149, 0.6889]`.
- Closed-book-wrong subset AUC was `0.6057`, below the scoped proceed gate.
- The simplest evidence-world likelihood signal was stronger: `l_plus` AUC `0.7085`.
- Original REV was below random: AUC `0.4266`.
- Audit corrected two implementation/interpretation issues:
  - closed-book `2/100` is prompt/parsing sensitive; direct short-answer closed-book first-20 is `7/20`;
  - same-title matched replacement was a real bug and has been fixed.
- The fixed first-30 rerun improved full NREV to AUC `0.6644`, but this is still below the audit keep-alive threshold `0.70` and still below `l_plus`.

Consequence:
- Treat current NREV as an informative but not-passed diagnostic for reader-likelihood falsification.
- Do not enter full implementation.
- If reopened, the only acceptable next step is one bounded T-minus redesign, not threshold tuning or open-pool rescue.
- Canonical notes:
  - `research_memory/emnlp_expand_then_compose/32_nrev_day0_sanity_20260427.md`
  - `reports/nrev/day0_sanity.md`
  - `reports/nrev/day0_audit.md`

## 2026-04-27: Close same-title paper-integrity audit

Decision:
- Do not rerun D-PathRAG, CPAG, or DAEC solely because NREV exposed a same-title replacement bug.
- Keep same-title exclusion as required hygiene for future destructive perturbation or edit-replacement methods.
- Treat same-title duplication as a documented corpus/pool confound, not as the explanation for current main results.

Reason:
- Static audit found D-PathRAG selector_v1 added `1205` non-gold documents, but `0` were same-title additions versus rank top-5.
- CEE learned_edit2 has `92` operation-level same-title non-gold additions, but these are multi-step oscillation/reinsertion; final selected-set same-title non-gold is `0`.
- CPAG anchored added `436` non-gold documents versus PropRAG rank with `0` same-title added non-gold; its sharper failure is shared cross-pool distractor agreement (`341 / 554` selected cross-pool gold/non-gold).
- DAEC 2Wiki/HotpotQA have negligible same-title exposure.
- MuSiQue has high duplicate-title exposure, but baseline already has more selected duplicate-title queries than DAEC (`0.388` vs `0.364` for PropRAG, `0.365` vs `0.349` for dense).

Consequence:
- Preserve D-PathRAG, CPAG, and DAEC conclusions.
- In paper writing, mention same-title replacement as a perturbation-method hygiene issue and MuSiQue duplicate-title exposure as a dataset/pool property.
- No new method exploration is justified by this audit.
- Canonical notes:
  - `research_memory/emnlp_expand_then_compose/33_same_title_integrity_audit_20260427.md`
  - `reports/paper/same_title_audit.md`
  - `reports/paper/same_title_audit.json`
