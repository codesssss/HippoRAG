# MuSiQue Failure Synthesis

Date: 2026-05-07

This memo consolidates the MuSiQue diagnostics run after the DBEC support-repair audit. It is an internal go/no-go document: the goal is to decide whether more MuSiQue-fix engineering is justified, and what can be safely written in the DBEC paper.

## Executive Decision

Stop the current MuSiQue-fix line and move to DBEC paper writing.

The evidence is now consistent across six independent diagnostics:

1. DBEC's core support-repair mechanism is real, but MuSiQue is the hardest residual case.
2. MuSiQue residual error is dominated by candidate visibility and fixed-pool limits, not by backend admission, scoring, or rank arbitration.
3. Chain-walking entity anchors have signal, but the signal is weak: `10.5%` New@5 and `19.8%` New@20 on the query-primary source-visible bucket.
4. Strict anchor policies do not fix the weakness: `title_strict` drops to `7.0%` New@5.
5. Deterministic dependency-bound retrieval expansion does not show an independent advantage: `dbir_det` is weaker than `context_iterative_lite` on source-visible Final New@5 and weaker than `independent_demand` / `rank_expansion` on pool-absent recovery.

The recommended next action is DBEC paper writing with a quantified MuSiQue limitation section. A D-BIR-LLM probe is only worth doing as a final, strictly budgeted kill-switch experiment, not as the default next step.

## Evidence Table

| Route | Hypothesis | Key Evidence | Decision |
|---|---|---|---|
| Support-repair audit | DBEC gains are caused by support-chain repair. | MuSiQue gold_doc_count>=3 and SetR-underselected dF1 `+0.0819`; 2Wiki gold_doc_count>=4 and SetR-underselected dF1 `+0.2851`. | Keep as main DBEC mechanism. |
| Repair-gated arbitration | Backend admission/repair arbitration can recover MuSiQue. | MuSiQue best repair-gated F1 `0.2922`, below DBEC by `-0.0171` and below rank_fill5 by `-0.0470`. | Stop. |
| Candidate-quality audit | Locate where missing gold disappears. | MuSiQue missing-gold bucket: `47.1%` source-only not rank/DBEC, `12.3%` absent from source pool, only `3.0%` rank/admission/scoring headroom. | Core limitation diagnosis. |
| Chain-walking probe | Upstream-doc entity extraction can rescue deep fixed-pool gold. | Query-primary: parse ok `100.0%`, New@5 `10.5%`, New@10 `18.6%`, New@20 `19.8%`. | Future-work only. |
| Anchor variants | Strict or demand-conditioned anchor matching can clean up broad anchors. | `current_all` New@5 `10.5%`; `title_strict` `7.0%`; `demand_entity_title` `9.3%`. | Stop anchor tuning. |
| D-BIR lexical pilot | Dependency-bound iterative retrieval has independent expansion value. | Source-visible Final New@5: `dbir_det` `2.3%` vs `context_iterative_lite` `5.8%`; pool-absent@100: `dbir_det` `12.0%` vs `independent_demand` / `rank_expansion` `28.0%`. | Stop or pivot. |

Full machine-readable table: `reports/musique_failure_synthesis_20260507/evidence_table.csv`.

## What The Data Says

### 1. DBEC is still a valid first-paper contribution

The support-repair audit shows that DBEC is not merely a generic reranker. Its strongest gains appear when SetR omits annotated support and DBEC recovers it:

- 2Wiki `gold_doc_count>=4 and SetR count-underselected`: dF1 `+0.2851`.
- MuSiQue `gold_doc_count>=3 and SetR count-underselected`: dF1 `+0.0819`.
- MuSiQue all: dF1 only `+0.0082`, which means MuSiQue should be framed as the difficult boundary case, not as the main proof of the method.

Paper implication: the DBEC story should be "structured support-chain repair helps under-selection," not "DBEC universally dominates every multi-hop setting."

### 2. MuSiQue is not mainly a backend selection problem

The candidate-quality audit is the most important diagnostic. Among `204` MuSiQue missing-gold support titles:

| Bucket | Count | Rate | Interpretation |
|---|---:|---:|---|
| `gold_source_only_not_rank_or_dbec` | 96 | 47.1% | Candidate generation / visibility gap. |
| `gold_recovered_by_dbec_final` | 77 | 37.7% | Already recovered by current DBEC final selection. |
| `gold_absent_from_source_pool` | 25 | 12.3% | Fixed-pool ceiling. |
| `gold_in_rank_fill_not_dbec` | 4 | 2.0% | Rank-fill sees it but DBEC misses it. |
| `gold_in_binding_candidates_not_selected` | 2 | 1.0% | True scoring/admission miss. |

The largest unresolved source-visible bucket has mean source rank `36.0`, p50 `27`, p75 `58`, and p90 `79`. This is too deep for a top-20 rerank tweak and too far upstream for admission/scoring changes.

Paper implication: MuSiQue failure should be described as front-end candidate visibility under long dependency chains.

### 3. Chain-walking found weak but real signal

The chain-walking probe tested whether an LLM can read an upstream document, extract a latent entity/title anchor, and rerank the fixed pool. On the query-primary source-visible bucket:

- Queries: `62`.
- Missing gold titles: `86`.
- Prompt rows: `324`.
- Parse ok rate: `100.0%`.
- New@5: `10.5%`.
- New@10: `18.6%`.
- New@20: `19.8%`.
- Mean source rank: `34.4`.
- Mean probe rank: `28.5`.

This rules out "the LLM cannot parse the task" as the primary explanation. The problem is that entity anchors are often too broad or semantically incomplete for the desired downstream document.

Paper implication: chain-walking can be cited as a lightweight future-work probe, not as a method component to implement now.

### 4. Anchor tuning does not rescue chain-walking

The anchor variant analysis reuses the same LLM outputs and changes only the matching policy:

| Policy | New@5 | New@10 | New@20 | Decision |
|---|---:|---:|---:|---|
| `current_all` | 10.5% | 18.6% | 19.8% | weak future work |
| `title_strict` | 7.0% | 9.3% | 3.5% | worse |
| `title_only` | 10.5% | 18.6% | 11.6% | no New@5 gain |
| `demand_entity_title` | 9.3% | 17.4% | 11.6% | worse |

The strict-title result is especially informative: weak body/token overlap is not pure noise. It contributes several points of New@5 recovery. The missing component is likely semantic query reformulation, not stricter entity matching.

Paper implication: do not claim high-precision anchoring is the path forward. The more honest future-work direction is semantic query reformulation or retrieval expansion.

### 5. D-BIR-det is a negative pilot for a second-paper dependency-bound framing

The D-BIR lexical pilot tested a second-paper idea: expand retrieval step-by-step under dependency-slot constraints. It used no LLM and no reader, so it is a structural feasibility test, not a final system result.

On the source-visible query-primary slice:

| Policy | Exp@100 | Final New@5 |
|---|---:|---:|
| `rank_expansion` | 20.9% | 5.8% |
| `independent_demand` | 9.3% | 2.3% |
| `context_iterative_lite` | 19.8% | 5.8% |
| `dbir_det` | 18.6% | 2.3% |

On the pool-absent slice:

| Policy | Pool-absent@100 | Final New@5 |
|---|---:|---:|
| `rank_expansion` | 28.0% | 8.0% |
| `independent_demand` | 28.0% | 8.0% |
| `context_iterative_lite` | 12.0% | 8.0% |
| `dbir_det` | 12.0% | 8.0% |

The important point is not that all expansion is impossible. It is that dependency-bound control did not show independent value in this deterministic substrate. Slot coverage reached `100%`, but that only means the procedure formally assigned bindings; it does not mean the bindings are correct evidence.

Paper implication: do not launch a full "Dependency-Bound Iterative Retrieval" paper from this pilot. At minimum, it would need an LLM reformulation kill-switch comparison against an equally strong no-dependency LLM reformulation baseline.

## Unified Interpretation

The consistent mechanism is:

1. DBEC can repair under-selected support chains when the needed evidence is visible to its candidate path.
2. MuSiQue often requires documents that are deep in the source pool or absent from it.
3. The reachable-but-deep documents are not recovered by simple backend fixes.
4. Entity-anchor chain-walking only recovers a small fraction of them.
5. Stricter matching hurts rather than helps.
6. Deterministic dependency-bound expansion is not stronger than simpler expansion baselines.

Therefore, the remaining MuSiQue gap likely requires a stronger front-end retrieval mechanism: semantic query reformulation, iterative retrieval with real retrieval expansion, or a different learned policy. That is beyond the current fixed-pool DBEC composition claim.

## Practical Ceiling Estimate

This is a diagnostic estimate, not a formal theorem.

The repair-gated reader report implies `rank_fill5` on the MuSiQue under-selected slice is around `0.3392` F1, while the earlier oracle-fill reference is around `0.5405`. Treat the oracle number as contextual rather than regenerated by this synthesis. The gap is about `0.20` F1. The candidate-quality audit says `47.1%` of missing gold is in the source-visible candidate-generation gap, and the chain-walking probe recovers only `19.8%` New@20 of that bucket under the fixed-pool probe.

A conservative fixed-pool anchor-style recovery estimate is:

```text
0.20 * 0.471 * 0.198 ~= 0.019 F1
```

Even with optimistic composition and reader effects, this supports a practical headroom of roughly `+0.02` to `+0.05` F1 for simple train-free MuSiQue fixes, not a large second-paper-scale gain.

## Paper-Facing Claims

Allowed claims:

- DBEC's gains are best understood as support-chain repair under evidence under-selection.
- MuSiQue exposes a candidate-generation limitation of fixed-pool composition methods.
- In MuSiQue missing-gold diagnostics, backend scoring/admission accounts for only a small visible bucket: `2.0%` rank-fill-not-DBEC plus `1.0%` binding-candidate-not-selected.
- Chain-walking entity extraction has weak but nonzero fixed-pool signal.
- Strict title anchoring does not improve the chain-walking signal.
- Deterministic dependency-bound retrieval expansion did not show independent value over simpler lexical expansion baselines.

Do not claim:

- D-BIR solves fixed-pool ceiling.
- Dependency-bound control is the active retrieval-expansion component.
- Strict anchors improve chain-walking.
- Admission/scoring is the main MuSiQue bottleneck.
- The `+0.02` to `+0.05` practical headroom estimate is a formal upper bound.

## Recommended Next Steps

1. Write the DBEC paper now.
2. Use this synthesis as Section 6 material: "Failure analysis and fixed-pool limitations on MuSiQue."
3. Include the candidate-quality audit table and one compact probe table, not every negative ablation.
4. If a second paper is still desired, pause engineering until a literature review identifies a defensible non-overlapping claim.
5. If the user insists on one last D-BIR check, run only a `<=3` day D-BIR-LLM kill-switch probe:
   - Compare `D-BIR-LLM with upstream binding` against `LLM-reformulate-independent-demand`.
   - Use the same MuSiQue source-visible and pool-absent slices.
   - Continue only if D-BIR-LLM beats the no-dependency LLM baseline by at least `10` percentage points on Exp@50 and reaches at least `15-20%` Final New@5.

## Source Artifacts

- `reports/support_repair_mechanism_20260507/summary.md`
- `reports/repair_gated_arbitration_offline_20260507/summary.md`
- `reports/repair_gated_arbitration_reader_20260507/summary.md`
- `reports/repair_candidate_quality_audit_20260507/summary.md`
- `reports/chain_walking_binding_probe_primary_20260507/summary.md`
- `reports/chain_walking_anchor_variants_primary_20260507/summary.md`
- `reports/dbir_pilot_musique_20260507/summary.md`
- `reports/dbir_pilot_musique_alltitles_20260507/summary.md`

## Companion Files

- Evidence table: `reports/musique_failure_synthesis_20260507/evidence_table.csv`
- Decision matrix: `reports/musique_failure_synthesis_20260507/decision_matrix.csv`
- Machine-readable summary: `reports/musique_failure_synthesis_20260507/summary.json`
