# MuSiQue Failure Synthesis

Date: 2026-05-08

This memo consolidates the MuSiQue diagnostics run after the DBEC support-repair audit. It is an internal go/no-go document: the goal is to decide whether more MuSiQue-fix engineering is justified, and what can be safely written in the DBEC paper.

## Executive Decision

Stop the current MuSiQue-fix line and move to DBEC paper writing.

The evidence is now consistent across eight independent diagnostics:

1. DBEC's core support-repair mechanism is real, but MuSiQue is the hardest residual case.
2. MuSiQue residual error is dominated by candidate visibility and fixed-pool limits, not by backend admission, scoring, or rank arbitration.
3. Repair-gated arbitration does not rescue MuSiQue: the best reader F1 is `0.2922`, below both DBEC and rank-fill controls on the under-selected slice.
4. Chain-walking entity anchors have signal, but the signal is weak: `10.5%` New@5 and `19.8%` New@20 on the query-primary source-visible bucket.
5. Strict anchor policies do not fix the weakness: `title_strict` drops to `7.0%` New@5.
6. Deterministic dependency-bound retrieval expansion does not show an independent advantage: `dbir_det` is weaker than `context_iterative_lite` on source-visible Final New@5 and weaker than `independent_demand` / `rank_expansion` on pool-absent recovery.
7. A stricter fixed-pool LLM probe changes the ceiling story: query reformulation and demand-HyDE are still weak, but listwise LLM selection over the given pool100 reaches `25.0-26.7%` New@5.
8. A full1000 RankGPT-style check is now paper-ready: the sliding-window local adaptation is lower than DAEC/SetR on reader F1 across 2Wiki, HotpotQA, and MuSiQue, while still recovering `24.0%` New@5 on the MuSiQue source-visible missing-gold slice.

The recommended next action is still DBEC paper writing, but the limitation wording should be more precise: fixed-pool candidate generation is not theoretically exhausted; current DBEC-style binding and cheap reformulation probes are insufficient, while expensive listwise LLM pool selection exposes remaining headroom.

## Evidence Table

| Route | Hypothesis | Key Evidence | Decision |
|---|---|---|---|
| Support-repair audit | DBEC gains are caused by support-chain repair. | MuSiQue gold_doc_count>=3 and SetR-underselected dF1 `+0.0819`; 2Wiki gold_doc_count>=4 and SetR-underselected dF1 `+0.2851`. | Keep as main DBEC mechanism. |
| Repair-gated arbitration | Backend admission/repair arbitration can recover MuSiQue. | MuSiQue best repair-gated F1 `0.2922`, below DBEC by `-0.0171` and below rank_fill5 by `-0.0470`. | Stop. |
| Candidate-quality audit | Locate where missing gold disappears. | MuSiQue missing-gold bucket: `47.1%` source-only not rank/DBEC, `12.3%` absent from source pool, only `3.0%` rank/admission/scoring headroom. | Core limitation diagnosis. |
| Chain-walking probe | Upstream-doc entity extraction can rescue deep fixed-pool gold. | Query-primary: parse ok `100.0%`, New@5 `10.5%`, New@10 `18.6%`, New@20 `19.8%`. | Future-work only. |
| Anchor variants | Strict or demand-conditioned anchor matching can clean up broad anchors. | `current_all` New@5 `10.5%`; `title_strict` `7.0%`; `demand_entity_title` `9.3%`. | Stop anchor tuning. |
| D-BIR lexical pilot | Dependency-bound iterative retrieval has independent expansion value. | Source-visible Final New@5: `dbir_det` `2.3%` vs `context_iterative_lite` `5.8%`; pool-absent@100: `dbir_det` `12.0%` vs `independent_demand` / `rank_expansion` `28.0%`. | Stop or pivot. |
| Fixed-pool candidate generation V2 | Query reformulation, demand-HyDE, or listwise LLM pool selection can rescue source-visible gold without pool expansion. | Query-primary New@5: `llm_query_reform` `5.8%`, `llm_demand_hyde` `11.6%`, `llm_listwise_select` `26.7%`. All-title New@5: `5.2%`, `11.5%`, `25.0%`. | Mixed signal; not a new-paper trigger. |
| RankGPT-style fixed-pool baseline | A stronger listwise reranker may dominate DBEC/SetR on the same PropRAG pool100. | Sliding-window reader F1: 2Wiki `0.6659` vs DAEC `0.7118` / SetR `0.6746`; HotpotQA `0.6845` vs `0.7473` / `0.7435`; MuSiQue `0.4093` vs `0.4548` / `0.4467`. Sliding selector Support R@5: `91.2%`, `88.0%`, `66.3%`. MuSiQue missing-slice New@5: `24.0%`. | Keep as reviewer-defense baseline; not a method replacement. |

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

### 6. Fixed-pool candidate generation is not exhausted

The follow-up fixed-pool candidate-generation probe tested query reformulation, demand-HyDE, and listwise LLM pool selection while keeping the candidate universe strictly fixed to the original PropRAG pool100. It used `/no_think` prompts, three Qwen3-8B endpoints, and no reader.

Query-primary results:

| Policy | New@5 | New@10 | Policy R@5 | Policy R@10 |
|---|---:|---:|---:|---:|
| `llm_query_reform` | 5.8% | 4.7% | 5.8% | 7.0% |
| `llm_demand_reform` | 5.8% | 8.1% | 5.8% | 10.5% |
| `llm_demand_hyde` | 11.6% | 14.0% | 11.6% | 17.4% |
| `llm_listwise_select` | 26.7% | 27.9% | 26.7% | 33.7% |

All-title results are similar:

| Policy | New@5 | New@10 | Policy R@5 | Policy R@10 |
|---|---:|---:|---:|---:|
| `llm_query_reform` | 5.2% | 4.2% | 5.2% | 6.2% |
| `llm_demand_reform` | 5.2% | 8.3% | 5.2% | 10.4% |
| `llm_demand_hyde` | 11.5% | 13.5% | 11.5% | 16.7% |
| `llm_listwise_select` | 25.0% | 27.1% | 25.0% | 32.3% |

This revises the earlier limitation story. Cheap reformulation and demand-HyDE do not materially beat chain-walking. However, when the LLM is allowed to inspect the fixed pool100 title/snippet list and choose document ids directly, it recovers about a quarter of the source-visible missing gold into top-5. That means the fixed pool itself still contains usable signal, but extracting it requires an expensive listwise pool-selection operation rather than the current DBEC binding candidate path.

Paper implication: do not claim fixed-pool candidate generation is at a fundamental ceiling. The safer claim is that simple binding anchors, query reformulation, demand-HyDE, and deterministic dependency expansion are insufficient, while listwise LLM pool selection exposes remaining headroom that is not yet converted into a practical low-cost DBEC component.

### 7. RankGPT-style listwise selection is useful but not stronger overall

The follow-up RankGPT-style baseline tested whether a paper-recognizable listwise reranker would dominate DBEC/SetR when constrained to the same fixed PropRAG pool100. The final version uses the more faithful sliding-window local adaptation: window `20`, step `10`, back-to-front permutation updates over ranks `0..100`, Qwen3-8B, `/no_think`, 1-based passage ids, and `1000` queries per dataset.

Single-pass RankGPT-style selection was useful as an ablation, but sliding-window is the right named comparison:

| Dataset | Variant | Calls/query | Support R@5 | Complete@5 |
|---|---:|---:|---:|---:|
| 2Wiki | `rank5_no_think` | 1 | 89.6% | 78.7% |
| 2Wiki | `sliding20_step10_no_think` | 9 | 91.2% | 80.0% |
| HotpotQA | `rank5_no_think` | 1 | 86.6% | 76.2% |
| HotpotQA | `sliding20_step10_no_think` | 9 | 88.0% | 78.4% |
| MuSiQue | `rank5_no_think` | 1 | 65.8% | 37.4% |
| MuSiQue | `sliding20_step10_no_think` | 9 | 66.3% | 39.5% |

Corrected reader results show that the listwise signal does not convert into an overall replacement for DAEC/SetR:

| Dataset | Method | EM | F1 | Reader R@5 |
|---|---|---:|---:|---:|
| 2Wiki | DAEC-selective | 0.6420 | 0.7118 | 94.1% |
| 2Wiki | SetR-faithful | 0.6030 | 0.6746 | 88.3% |
| 2Wiki | RankGPT-style sliding | 0.6020 | 0.6659 | 91.2% |
| HotpotQA | DAEC-selective | 0.6200 | 0.7473 | 96.0% |
| HotpotQA | SetR-faithful | 0.6250 | 0.7435 | 92.3% |
| HotpotQA | RankGPT-style sliding | 0.5660 | 0.6845 | 87.8% |
| MuSiQue | DAEC-selective | 0.3530 | 0.4548 | 74.7% |
| MuSiQue | SetR-faithful | 0.3440 | 0.4467 | 65.9% |
| MuSiQue | RankGPT-style sliding | 0.3190 | 0.4093 | 62.5% |

Paired bootstrap confirms the reader-F1 gap against DAEC-selective on all three datasets: 2Wiki `+0.0460` F1, HotpotQA `+0.0628`, MuSiQue `+0.0455`, all with 95% CIs excluding zero. Against SetR-faithful, RankGPT-style sliding is tied on 2Wiki F1 but lower on HotpotQA and MuSiQue.

The MuSiQue source-visible missing-gold slice still shows local listwise headroom:

| Method | Missing titles | Queries | New@5 |
|---|---:|---:|---:|
| `dbec_selective` | 96 | 72 | 0.0% |
| `rankgpt_rank5_no_think` | 96 | 72 | 20.8% |
| `rankgpt_select5_no_think` | 96 | 72 | 25.0% |
| `rankgpt_sliding20_step10_no_think` | 96 | 72 | 24.0% |
| `setr_faithful` | 96 | 72 | 0.0% |

Paper implication: include this as a reviewer-defense baseline and a complementary-headroom finding. The honest claim is that, under the controlled Qwen3-8B `/no_think` substrate, DAEC/DBEC is stronger than this RankGPT-style local adaptation on reader F1, while RankGPT-style listwise selection confirms that some MuSiQue residual supports are still discoverable by expensive full-pool inspection. Do not claim this is a full GPT-3.5/4 RankGPT reproduction.

## Unified Interpretation

The consistent mechanism is:

1. DBEC can repair under-selected support chains when the needed evidence is visible to its candidate path.
2. MuSiQue often requires documents that are deep in the source pool or absent from it.
3. The reachable-but-deep documents are not recovered by simple backend fixes.
4. Entity-anchor chain-walking only recovers a small fraction of them.
5. Stricter matching hurts rather than helps.
6. Deterministic dependency-bound expansion is not stronger than simpler expansion baselines.
7. Strict fixed-pool LLM listwise selection can recover a nontrivial fraction, so the fixed pool is not theoretically exhausted.
8. RankGPT-style full-pool listwise selection does not dominate DAEC/SetR on reader F1, so listwise headroom is local rather than a replacement for the DBEC selector.

Therefore, the remaining MuSiQue gap likely requires a stronger front-end candidate-generation mechanism. The new evidence narrows that statement: lightweight semantic query reformulation is not enough, but direct listwise reasoning over the fixed pool can find some missing supports. The corrected sliding-window reader run shows that this signal is complementary rather than a drop-in method replacement under the current cost/substrate constraints.

## Practical Ceiling Estimate

This is a diagnostic estimate, not a formal theorem.

The repair-gated reader report implies `rank_fill5` on the MuSiQue under-selected slice is around `0.3392` F1, while the earlier oracle-fill reference is around `0.5405`. Treat the oracle number as contextual rather than regenerated by this synthesis. The gap is about `0.20` F1. The candidate-quality audit says `47.1%` of missing gold is in the source-visible candidate-generation gap, and the chain-walking probe recovers only `19.8%` New@20 of that bucket under the fixed-pool probe.

A conservative fixed-pool anchor-style recovery estimate is:

```text
0.20 * 0.471 * 0.198 ~= 0.019 F1
```

Even with optimistic composition and reader effects, this supports a practical headroom of roughly `+0.02` to `+0.05` F1 for anchor-style and cheap reformulation fixes. The listwise fixed-pool probe changes the upper-bound intuition: fixed-pool headroom is larger than the chain-walking estimate, but accessing it currently requires having the LLM read and select from pool100 directly. The corrected RankGPT-style sliding reader run measures that costlier path and still does not beat DAEC/SetR overall, so it is better framed as complementary evidence and future hybrid direction, not a low-cost DBEC repair.

## Paper-Facing Claims

Allowed claims:

- DBEC's gains are best understood as support-chain repair under evidence under-selection.
- MuSiQue exposes a candidate-generation limitation of fixed-pool composition methods.
- In MuSiQue missing-gold diagnostics, backend scoring/admission accounts for only a small visible bucket: `2.0%` rank-fill-not-DBEC plus `1.0%` binding-candidate-not-selected.
- Chain-walking entity extraction has weak but nonzero fixed-pool signal.
- Strict title anchoring does not improve the chain-walking signal.
- Deterministic dependency-bound retrieval expansion did not show independent value over simpler lexical expansion baselines.
- Fixed-pool LLM listwise selection shows nontrivial candidate-generation headroom: `25.0-26.7%` New@5 on the source-visible MuSiQue bucket.
- Under the controlled Qwen3-8B `/no_think` fixed-pool setup, RankGPT-style sliding-window local adaptation is weaker than DAEC/SetR on reader F1 across 2Wiki, HotpotQA, and MuSiQue, while still showing MuSiQue missing-slice New@5 headroom of `24.0%`.

Do not claim:

- D-BIR solves fixed-pool ceiling.
- Dependency-bound control is the active retrieval-expansion component.
- Strict anchors improve chain-walking.
- Admission/scoring is the main MuSiQue bottleneck.
- The `+0.02` to `+0.05` practical headroom estimate is a formal upper bound.
- Fixed-pool candidate generation has been proven exhausted.
- Query reformulation or demand-HyDE alone solves MuSiQue.
- The RankGPT-style baseline is a full GPT-4 RankGPT reproduction; it is a Qwen3-8B `/no_think` sliding-window local adaptation.
- RankGPT-style listwise selection is universally worse or useless; it recovers a complementary MuSiQue missing-support subset even though its full1000 reader F1 is lower.

## Recommended Next Steps

1. Write the DBEC paper now.
2. Use this synthesis as Section 6 material: "Failure analysis and fixed-pool limitations on MuSiQue."
3. Include the candidate-quality audit table and one compact probe table, not every negative ablation.
4. If a second paper is still desired, pause engineering until a literature review identifies a defensible non-overlapping claim.
5. Do not run more RankGPT variants now. Use the corrected sliding-window reader run as the paper-ready listwise comparison; leave with-thinking and stronger-LLM RankGPT variants as future work under cross-substrate evaluation.

## Source Artifacts

- `reports/support_repair_mechanism_20260507/summary.md`
- `reports/repair_gated_arbitration_offline_20260507/summary.md`
- `reports/repair_gated_arbitration_reader_20260507/summary.md`
- `reports/repair_candidate_quality_audit_20260507/summary.md`
- `reports/chain_walking_binding_probe_primary_20260507/summary.md`
- `reports/chain_walking_anchor_variants_primary_20260507/summary.md`
- `reports/dbir_pilot_musique_20260507/summary.md`
- `reports/dbir_pilot_musique_alltitles_20260507/summary.md`
- `reports/fixed_pool_candidate_generation_probe_primary_20260508/summary.md`
- `reports/fixed_pool_candidate_generation_probe_alltitles_20260508/summary.md`
- `reports/rankgpt_fixed_pool_baseline_full1000_1based_20260508/summary.md`
- `reports/rankgpt_fixed_pool_baseline_sliding_full1000_20260508/summary.md`
- `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/summary.md`

## Companion Files

- Evidence table: `reports/musique_failure_synthesis_20260507/evidence_table.csv`
- Decision matrix: `reports/musique_failure_synthesis_20260507/decision_matrix.csv`
- Machine-readable summary: `reports/musique_failure_synthesis_20260507/summary.json`
