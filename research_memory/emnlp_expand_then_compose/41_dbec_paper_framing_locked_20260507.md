# DBEC Paper Framing — SUPERSEDED

> ⚠️ **SUPERSEDED 2026-05-15**. This document is no longer the active paper framing.
>
> The active framing is **v3.2.2 EvidenceFlow** in
> `46_evidence_flow_over_graphs_framing_skeleton_20260514.md`.
>
> What 41_ still has value for:
> - Historical experiment data anchors (oracle@100 gaps, under-selection rates)
> - The 12 locked-out language entries (still useful as paper writing rules)
> - The 6 honest limitations list (template, still adaptable)
> - SetR-style baseline design context (40_ for full details)
>
> What 41_ should NOT be used for any more:
> - Method name (DBEC → being replaced; see 46_ for naming discussion)
> - Title (was "DBEC: Dependency-Bound Evidence Composition..."; now "Evidence Flow over Graphs")
> - Section 4 method structure (was demand decomposition / binding / coverage / IG gate; now mechanism components of EvidenceFlow as a single end-to-end process — see 46_ §4)
> - Abstract (entirely re-drafted under v3.2.2)
> - Contribution wording (4 contributions of 41_ → 3 contributions of 46_)
> - "SetR as main competitor" framing (now appendix-only under v3)

---

(Historical content below preserved for data anchor / limitation template reference.)

Locked: 2026-05-07
Status: Final framing for paper writing. Do not modify without explicit re-review.

This document supersedes any prior DAEC / Demand-Aware framing in research memo. All paper-facing content from this point uses **DBEC** naming and the framing below.

---

## Method Naming Decision

| Old (internal) | New (paper-facing) |
|---|---|
| DAEC | **DBEC** |
| Demand-Aware Evidence Composition / Coverage | **Dependency-Bound Evidence Composition** |
| DAEC-selective | **DBEC-IG** (Identifiability-Gated) |
| DAEC base / nobinding | DBEC base / DBEC-nobinding (ablation) |

Internal documentation prior to 2026-05-07 may use DAEC; these refer to the same method. Naming was updated to (a) avoid collision with SetR's "information requirements" framing and (b) better reflect the dependency-binding mechanism that distinguishes DBEC from prompt-only set selection.

---

## Paper Title

**DBEC: Dependency-Bound Evidence Composition for Multi-Hop Retrieval-Augmented Generation**

Rationale:
- "Multi-Hop RAG" is the precise scope (broader than Multi-Hop QA, narrower than general RAG)
- "Dependency-Bound" is the real differentiator from SetR (which has information requirements but no entity-anchored binding)
- Avoids "Set Selection" / "Demand-Aware" language (collides with SetR)

---

## One-Sentence Claim

DBEC is a fixed-pool evidence composition method for multi-hop RAG. It does not treat passage selection as black-box prompt-only set selection; instead it explicitly models query demands, dependency bindings, and evidence coverage. This produces more reliable multi-hop evidence composition under tight reader budgets, particularly mitigating the under-selection failure mode of prompt-only adaptive selectors on deep dependency queries.

---

## Three-Layer Position

### Layer 1 — Setting Claim

**Multi-hop RAG bottleneck is not only retrieval recall but evidence composition.**

Anchor data:
- On retrieval pools with R@100 ≥ 0.95, naive top-5 selection still leaves +0.124 / +0.082 / +0.179 EM oracle gap on 2Wiki / HotpotQA / MuSiQue
- Reorder@20 < Select@100 → gap is in set composition, not pointwise reranking

This is the motivation, not dependent on any baseline comparison.

### Layer 2 — Method Claim

**Dependency-bound coverage objective closes part of this gap, with structural interpretability that prompt-only LLM selection lacks.**

Empirical support:
- DBEC monotonically outperforms Top-5, IRCoT-style, LLM-direct (title), LLM-direct (snippet) across 3 datasets × 3 retrieval pools
- DBEC-IG provides empirical safety (no negative regressions across 9 evaluated combinations)

### Layer 3 — Mechanism Claim

**DBEC's advantage is concentrated on deep dependency queries where prompt-only adaptive selection systematically under-selects.**

Empirical support:
- 2Wiki 4-doc hard subset: DBEC vs SetR-style +0.144 F1 (CI [0.089, 0.200])
- Budget-matched: DBEC vs SetR-Fill@5 still +0.045 F1 (CI [0.003, 0.089])
- Conditional: SetR under-select cases DBEC +0.285, SetR selects-enough cases tied
- Cross-dataset replication: MuSiQue ≥3-doc under-selected subset DBEC +0.082 F1

---

## Four Contributions (Locked)

### Contribution 1 — Problem Formulation

> "We identify and formalize a fixed-pool evidence composition problem in multi-hop RAG, where the goal is to select a compact evidence set that jointly covers mutually dependent information demands."

Anchor: composition gap data + reorder@20 vs select@100 evidence

### Contribution 2 — DBEC Method

> "We propose DBEC, a train-free evidence composer that decomposes queries into dependency-linked demands, grounds latent dependencies through entity-anchored bindings, and selects evidence by optimizing a transparent noisy-OR coverage objective over **demand-binding units**."

Three structural components:
1. Demand decomposition with explicit dependency edges
2. Entity-anchored latent binding (Cartesian product over candidates with bound size)
3. Transparent noisy-OR coverage objective with greedy top-K selection

Key term: **demand-binding units** (distinguishes DBEC from SetR's information requirements)

### Contribution 3 — Identifiability-Gated DBEC (DBEC-IG)

> "We further introduce an identifiability gate that prevents DBEC from applying entity bindings when the extracted referents cannot be uniquely resolved. In our Wikipedia-derived benchmarks, a title-uniqueness instantiation avoids the negative regressions observed in ungated binding across the evaluated pool-dataset combinations."

Key qualifiers:
- "instantiation" not "framework solved"
- "Wikipedia-derived benchmarks" limits scope
- "avoids negative regressions across evaluated combinations" not "monotone non-negative property"

Negative results to discuss in Section 4:
- C1: Embedding-based binding posterior — fails (cosine similarity ≠ factual entailment)
- C2: Co-mention verifier — fails (entity co-mention ≠ relation entailment)
- C: Title-uniqueness gate — works (selection-independent signal)

### Contribution 4 — Controlled Evaluation and Mechanism Analysis

> "Across three benchmarks, three retrieval pools, and multiple controlled baselines, we show that DBEC significantly outperforms standard top-K, IRCoT-style, and LLM-direct selector baselines, remains competitive with SetR-style overall, and significantly improves deep-dependency slices. Conditional analysis shows that the gains are concentrated on under-selection failures in prompt-only adaptive set selection."

Key wording (locked):
- "conditional analysis shows that the gains are concentrated on" (NOT "causal mediation evidence")
- "the strongest SetR-style baseline in our controlled substrate" (NOT "the strongest LLM-based set selection baseline")
- "competitive with SetR-style overall" (NOT "outperforms SetR")

---

## Final Abstract (Locked)

```
Multi-hop retrieval-augmented generation requires composing a compact 
set of mutually dependent evidence passages from a fixed retrieval pool. 
We show that even high-recall pools leave a substantial composition 
gap under small reader budgets, suggesting that the bottleneck is not 
only retrieval recall but evidence set construction.

We introduce DBEC, Dependency-Bound Evidence Composition, a train-free 
fixed-pool evidence composer for multi-hop RAG. DBEC decomposes a 
query into atomic demands with dependency edges, grounds latent 
dependencies through entity-anchored bindings, and selects evidence 
by optimizing a transparent noisy-OR coverage objective over 
demand-binding units. We further add an identifiability gate that 
abstains from binding when extracted dependencies cannot be uniquely 
resolved.

Across three multi-hop benchmarks and three retrieval pools, DBEC 
significantly outperforms top-K, IRCoT-style, and prompt-only LLM 
selector baselines. Against a SetR-style adaptive selector, DBEC 
is competitive overall and significantly improves deep-dependency 
queries, including +0.144 F1 on the 2Wiki 4-document subset. 
Mechanism analysis shows that this advantage is concentrated on 
under-selection failures: SetR-style selects fewer than the 
required supports on 51.9% of 2Wiki 4-document queries, where DBEC 
improves F1 by +0.285, while the methods are tied when SetR selects 
sufficient evidence. Additional budget controls show that larger 
reader budgets and selection ordering are separable effects. These 
results position DBEC as a structured and interpretable alternative 
to prompt-only evidence selection under tight multi-hop reader budgets.
```

---

## Final Paper Identity Statement

> "DBEC is a train-free, interpretable, fixed-pool evidence composer that **trades additional inference-time structure** for better tight-budget composition and diagnosability on deep multi-hop queries."

Use this statement when responding to "what is your paper about" / "why DBEC over X" questions during reviews.

---

## Claims to Make (with data support)

| Claim | Support |
|---|---|
| DBEC significantly outperforms 4 baselines | Top-5 / IRCoT / LLM-direct title / LLM-direct snippet, 3 datasets, all CI excludes 0 |
| DBEC competitive with SetR-style overall | Full-set: 1 sig win + 2 tied |
| DBEC significantly improves deep dependency queries | 2Wiki 4-doc +0.144 F1, CI excludes 0 |
| DBEC significantly improves budget-matched comparison on hard subset | 2Wiki 4-doc vs SetR-Fill@5 +0.045 F1, CI excludes 0 |
| DBEC's gains are concentrated on under-selected cases | Conditional: SetR under-select +0.285, selects-enough tied |
| DBEC-IG avoids negative regressions across evaluated combinations | 9 pool×dataset cells, no negative regression |
| Identifiability gate uses selection-independent signal | Selection-independent computation; contrast with C1/C2 negative results |
| Composition gap exists | Oracle@100 vs top-5 = +0.12-0.18 EM |
| Reader budget and selection quality are separable | Top-10 retriever baseline experiment |

---

## Claims NOT to Make (locked-out language)

| ❌ Forbidden phrasing | ✅ Replacement |
|---|---|
| "DBEC outperforms SetR" | "DBEC is competitive with SetR-style overall, significantly improves deep-dependency queries" |
| "DBEC is more efficient than SetR" | "DBEC trades additional inference-time structure for interpretability" |
| "DBEC achieves SOTA" | "DBEC achieves competitive performance with strong LLM-based selectors" |
| "Causal mediation evidence" | "Conditional analysis shows the gains are concentrated on" |
| "Monotone non-negative property" | "Avoids negative regressions across evaluated combinations" |
| "Our strongest LLM-based set selection baseline" | "Our strongest SetR-style baseline in the controlled substrate" |
| "We reproduce SetR" | "We implement SetR-style, a local adaptation of the SetR IRI prompt" |
| "Binding always helps" | "Binding is load-bearing on identifiable dependencies; the gate handles ambiguous cases" |
| "DBEC beats Top-10 retriever" | "DBEC's strict 5-doc budget is sometimes insufficient on long-chain queries" |
| "Identifiability gate is corpus-general" | "Title-uniqueness instantiation in Wikipedia-derived corpora" |
| "We solve the multi-hop RAG composition problem" | "We close part of the composition gap" |
| "No mainstream paper uses non-Wiki datasets" | "We follow the dominant multi-hop RAG evaluation protocol" |

---

## Honest Limitations (must include in paper)

1. **Cost**: DBEC uses 8.4 calls/query vs SetR's 1 call/query. Token total ~8400 vs ~1900. DBEC trades calls for structure, not for raw efficiency.

2. **Budget cap on long-chain queries**: MuSiQue DBEC strict 5-doc significantly underperforms Top-10 retriever (-0.028 F1). DBEC's structured composition does not subsume retriever recall benefit at larger budgets. Adaptive budget is future work.

3. **Identifiability instantiation limited to Wikipedia-derived corpora**: Title-uniqueness gate is the Wikipedia instantiation. General framework (entity resolution uniqueness) requires adapted instantiation for non-Wiki corpora; not validated on non-Wiki datasets.

4. **SetR-style is local adaptation**: SetR-style / Fill@5 / Fill@10 are local Qwen3-8B prompt adaptations, not official SetR reproduction (official selector weights not publicly released). Strictly a controlled prompt baseline, not a SetR performance benchmark.

5. **Single LLM family**: Main experiments use Qwen3-8B only. LLM-family generalization not validated.

6. **Wikipedia-derived dataset coverage**: Evaluation uses 2Wiki, HotpotQA, MuSiQue — all Wikipedia-derived. Extension to broader non-Wikipedia multi-hop RAG corpora (e.g., MultiHop-RAG) is left for future work.

---

## Paper Structure (Locked)

```
Section 1: Introduction
  - Para 1: Multi-hop RAG composition setting
  - Para 2: Composition gap (oracle ceiling data)
  - Para 3: Existing approaches fall short (rerankers / iterative / prompt-only)
  - Para 4: DBEC contribution preview

Section 2: Related Work
  - 2.1 Retrieval-augmented evidence selection (rerankers, set selectors)
  - 2.2 LLM-based passage selection (SetR and variants, with explicit DBEC vs SetR distinction)
  - 2.3 Multi-hop RAG (HippoRAG, PropRAG, IRCoT — substrate / iterative)

Section 3: Problem Formulation
  - 3.1 Multi-hop RAG composition setting
  - 3.2 Demand decomposition with dependency edges (formal)
  - 3.3 Coverage objective definition

Section 4: DBEC Method
  - 4.1 Demand decomposition
  - 4.2 Entity-anchored latent binding
  - 4.3 Noisy-OR coverage objective and greedy selection
  - 4.4 Identifiability-gated binding (DBEC-IG)
        Including negative results C1 (embedding posterior) and C2 (co-mention verifier)

Section 5: Experiments
  - 5.1 Setup (controlled substrate, baselines, pools, datasets)
  - 5.2 Main results (Table 1: 5 baselines + 3 SetR variants on PropRAG full1000)
  - 5.3 When does DBEC's advantage manifest? (Mechanism analysis with conditional slicing)
  - 5.4 SetR variants ablation (style / Fill@5 / Fill@10 / Top-10 retriever)
  - 5.5 Cross-pool stability (Table 3: Dense / HippoRAG / PropRAG)
  - 5.6 Cost analysis (with explicit caveats)
  - 5.7 Method ablations (nobinding, IG gate)

Section 6: Discussion and Limitations
  - All 6 limitations from above

Section 7: Conclusion
```

---

## Key Terminology (Locked)

| Term | Definition |
|---|---|
| **Composition gap** | Difference between oracle@100 and top-5 baseline in answer F1/EM, indicating headroom achievable through better evidence selection (not retrieval) |
| **Demand-binding units** | DBEC's atomic selection units: a (demand, entity binding) pair grounding a dependency to canonical referents. Distinguishes from SetR's information requirements which are entity-agnostic |
| **Under-selection failure mode** | Behavior where prompt-only adaptive selectors choose fewer passages than required to support the query, measured as `n_selected < n_gold_supports` |
| **Identifiability gate** | Selection-independent abstention mechanism that prevents binding when extracted referents cannot be uniquely resolved in the candidate pool |
| **SetR-style** | Local Qwen3-8B implementation of SetR's IRI prompt with adaptive context length matching the original SetR paper. Not an official reproduction |
| **SetR-Fill@5** | SetR-style + rank-order fallback to top-5, providing budget-matched comparison with DBEC |
| **SetR-Fill@10** | SetR-style + rank-order fallback to top-10, matching the official SetR repository's `convert_rankify.py` k=10 behavior |
| **Controlled substrate** | Same Qwen3-8B LLM, same NV-Embed embedding, same retrieval pool across all baselines and methods, isolating method effects from substrate effects |

---

## Decisions Locked Out

These decisions are FINAL and should not be re-litigated without strong new evidence:

1. **Method name is DBEC**, not DAEC. Paper-facing content uses DBEC exclusively.
2. **Paper scope is Multi-Hop RAG**, not general RAG and not Multi-Hop QA.
3. **No new experiments before paper writing completion**. Specifically:
   - No non-Wikipedia dataset experiments
   - No stronger-LLM rerun
   - No weaker-LLM robustness study
   - No additional SetR variants
   - Robustness experiments are deferred until paper draft completion reveals specific gaps
4. **SetR-style is one of 5 baselines**, not paper narrative center.
5. **Composition gap is paper motivation anchor**, not "DBEC beats SetR".
6. **All abstract / contribution / claim wording follows the locked language above**.

---

## Writing Sequence (Recommended)

To minimize backtracking:

1. **Day 1-2**: Section 5.3 (Mechanism analysis) — strongest section, write first
2. **Day 3**: Section 5.1 + 5.2 (Setup + Main results) — table-heavy, mechanical
3. **Day 4**: Section 5.4-5.7 (SetR ablation, cross-pool, cost, method ablations)
4. **Day 5-7**: Section 3-4 (Problem formulation + Method) — careful exposition
5. **Day 8-9**: Section 1 (Introduction) — last, after Section 5 reveals narrative center
6. **Day 10**: Section 2 (Related Work) + Section 6 (Discussion) + Section 7 (Conclusion)
7. **Day 11**: Abstract — last, compress entire paper into 4 paragraphs

---

## Cross-Reference

- Method implementation: `scripts/dtc_embed_utils.py` (binding extraction, noisy-OR), `scripts/eval_causal_qwen3.py` (full pipeline)
- Main results data: `run_logs/daec_llm_wiki_title_proprag_full1000_20260503/`, `run_logs/daec_selective_titleuniq_proprag_full1000_20260506/`, `run_logs/daec_selective_titleuniq_dense_hipporag_full1000_20260506/`
- SetR variant data: `reports/setr_full1000_20260503/`, `reports/setr_faithful_proprag_full1000_20260507/`, `reports/setr_fill10_proprag_full1000_20260507/`
- Top-10 retriever baseline: `reports/top10_retriever_proprag_full1000_20260507/`
- Hard subset analysis: `reports/setr_faithful_proprag_full1000_20260507/2wiki_4doc_hard_slice.md`
- Under-selection mechanism: `reports/setr_faithful_proprag_full1000_20260507/underselection_mechanism_slices.md`
- Identifiability gate Phase-0/1 reports: `reports/daec_selective_binding_phase0_20260506/`, `reports/daec_selective_binding_phase1_20260506/`
- Negative results (C1/C2): `research_memory/emnlp_expand_then_compose/38_daec_binding_posterior_negative_20260506.md`

---

## EMNLP-Specific Strengthening (Added 2026-05-07)

After a simulated EMNLP-strict review, four concrete improvements are adopted to strengthen the method-paper presentation. **These are concrete additions, not framing reorientations**. The locked framing above (composition gap → method → mechanism → limitation) remains primary.

Specifically rejected from the simulated review: reframing the paper as "semantic composition failure mode" or "linguistic dependency binding" paper. The paper is honestly an evidence-composition method paper, not a compositional-semantics paper. Overclaiming semantic-modeling depth would open an attack surface (e.g., comparison to lambda calculus / formal binding / semantic parsing) that the actual method does not support.

### Strengthening 1 — Formalize Demand-Binding Unit in Section 3

Add formal definition to make the central abstraction precise. This addresses the "is DBU just renaming SetR's information requirement?" attack.

**Definition to write into Section 3:**

> Given a query q, demand decomposition produces a demand graph G_q = (R, E), where R = {r_1, ..., r_n} is a set of atomic demands and E ⊂ R × R is a set of dependency edges. A dependency edge (r_i, r_j) ∈ E indicates that satisfying r_j requires a referent (entity, document, or intermediate answer) introduced by r_i.
>
> A **binding assignment** b is a partial mapping from dependency variables in G_q to canonical referents in the candidate pool P. A **demand-binding unit** (DBU) is a pair u = (r_i, b), where r_i is an atomic demand and b is a binding assignment that resolves all dependency variables on which r_i depends.
>
> DBEC selects evidence by optimizing coverage over demand-binding units rather than over demands alone. This contrasts with prompt-only adaptive selectors, whose selection is conditioned only on the query text and passage candidates, without an explicit binding mapping.

Concrete illustrative example to include (use a real query from 2Wiki):

> Query: "What is the birthplace of the director of the film that starred X?"
>
> Demands (after decomposition):
> - r_1: identify the film starring X
> - r_2: identify the director of [the film from r_1]
> - r_3: identify the birthplace of [the director from r_2]
>
> Without explicit binding, an evidence selector can satisfy r_2 with documents about a director and r_3 with documents about a birthplace, without ensuring the director in r_2 is the director of the film resolved in r_1, or that the birthplace in r_3 corresponds to the same director. Demand-binding units enforce that coverage is evaluated under a coherent binding assignment across the dependency chain.

### Strengthening 2 — Move Nobinding Ablation into Main Section 5.3

The DBEC-nobinding ablation is load-bearing evidence that binding is the active mechanism, not decorative. Move it from Section 5.7 (method ablations, originally appendix-flavored) into Section 5.3 (main mechanism analysis), placed alongside the conditional under-selection slicing.

**Section 5.3 structure after this change:**

```
5.3 When Does DBEC's Advantage Manifest?

5.3.1 Query depth modulates the gain
  - Hard subset (2Wiki 4-doc): DBEC vs SetR-style +0.144 F1
  - Shallow subset (HotpotQA, all 2-doc): tied
  - Cross-dataset replication on MuSiQue ≥3-doc

5.3.2 Under-selection rate scales with depth
  - 2Wiki 2-doc 4.6% / 4-doc 51.9% / HotpotQA 3.2% / MuSiQue ≥3-doc 22.0%
  - Pattern is property of prompt-only adaptive selection, replicates across datasets

5.3.3 Binding is load-bearing under identifiable dependencies
  - DBEC-nobinding ablation: removes binding mechanism while keeping decomposition + coverage
  - Result: nobinding underperforms DBEC on 2Wiki (F1 -0.10) but recovers DBEC on MuSiQue
  - This is the dependency conditioning evidence: when removed, advantage disappears precisely on the slices DBEC was designed to handle

5.3.4 Conditional under-selection slicing
  - SetR under-selects: DBEC +0.285 F1 (2Wiki 4-doc)
  - SetR selects ≥ required: DBEC tied (-0.009)
  - This identifies under-selection — not general LLM weakness — as the failure mode DBEC addresses
```

The key claim to write into Section 5.3.3:

> Removing the binding mechanism (DBEC-nobinding) preserves demand decomposition and the coverage objective but does not condition coverage on dependency assignments. On 2Wiki (where binding is most beneficial), this ablation costs F1 -0.10. On MuSiQue (where ungated binding can over-constrain), the ablation recovers DBEC's performance. This pattern indicates that binding is load-bearing on identifiable deep-dependency cases and that the identifiability gate is necessary to handle ambiguous cases — not that binding is universally helpful.

### Strengthening 3 — Reframe Cost as Reader-Context Budget Optimization

Replace ambiguous "trades calls for structure" cost framing with a precise resource-axis distinction. DBEC and SetR optimize different cost dimensions.

**Replace this phrasing throughout the paper:**

❌ "DBEC trades higher LLM call count for structured interpretability"

❌ "DBEC is not a low-call selector relative to one-shot LLM set selection"

**With this phrasing:**

✅ "DBEC optimizes evidence compactness and dependency coverage under a fixed reader budget, not selector-side inference cost."

✅ "DBEC is not proposed as a cheaper replacement for prompt-only selectors. It optimizes a different resource axis: tighter reader-context composition under a fixed K, with diagnostic structure."

**Section 5.6 (Cost Analysis) opening sentence:**

> We separate two resource axes: selector-side inference cost (LLM calls and tokens consumed during evidence selection) and reader-side context budget (number and total length of passages presented to the answer reader). DBEC and prompt-only selectors trade off differently on these axes.

Then present the cost table with this resource-axis framing applied. Selector-side: DBEC > SetR. Reader-side: matched (both K=5).

Limitation 1 (cost) updates accordingly:

> DBEC uses 8.4 LLM calls and ~8400 prompt tokens per query for selector-side computation, compared to SetR-style's 1 call and ~1900 tokens. This is a deliberate trade: DBEC optimizes reader-context budget, not selector-side cost. Whether this trade is worthwhile depends on application requirements (e.g., reader compute is dominant, evidence interpretability is required, or downstream auditing matters). It is not appropriate to view DBEC as a more efficient alternative to prompt-only selection.

### Strengthening 4 — Demote Train-Free from Novelty Claim to Setting Constraint

"Train-free" should not appear as a contribution claim word. It is an experimental constraint that defines the setting (no in-domain training, no selector training, no retriever training).

**Replace this kind of phrasing:**

❌ "We propose a train-free evidence composer that achieves..."

❌ "DBEC is a train-free, interpretable, fixed-pool evidence composer..."

**With this:**

✅ "Under a no-training, fixed-pool setting, we study whether explicit dependency binding can repair prompt-only evidence composition failures."

✅ "DBEC operates inference-time only, on a fixed retrieval pool and a fixed LLM, without selector or retriever training."

This appears in:
- Section 1 (Introduction): mentioned once as setting, not as contribution
- Section 5.1 (Setup): explicitly listed as constraint, alongside "controlled substrate"
- Removed from Section 1 contribution preview
- Removed from Abstract method description

### Cross-Reference: Where the 4 Strengthening Items Land

| Item | Section | Word/term changes |
|---|---|---|
| Formalize DBU | Section 3.2 (new) | Add definition box, illustrative example |
| Nobinding ablation | Section 5.3.3 (moved from 5.7) | Reposition table, write load-bearing claim |
| Reader-context budget framing | Section 5.6, Limitation 1, Abstract | Replace "trades calls for structure" globally |
| Train-free demotion | Abstract, Section 1, Section 5.1 | Remove "train-free" from novelty contexts; keep in setup |

### What Remains Unchanged (Locked, Not Reframed)

- Title: DBEC: Dependency-Bound Evidence Composition for Multi-Hop Retrieval-Augmented Generation
- Method name: DBEC (and DBEC-IG variant)
- Scope: Multi-Hop RAG (not Multi-Hop QA, not general RAG)
- Four-contribution structure (problem formulation, method, IG, mechanism analysis)
- Composition gap as motivation anchor
- SetR is one of 5 baselines, not narrative center
- Honest limitations list (6 items)
- Locked-out language list (12 forbidden phrasings)

The strengthening above operates within the locked framing. It does not replace any locked decision.

---

## Status

**LOCKED 2026-05-07**. Paper writing starts from this framing.

EMNLP-specific strengthening (4 items above) added 2026-05-07. These are concrete writing-time additions; they do not modify the locked framing decisions.

**Naming update 2026-05-07**: "SetR-faithful" renamed to "SetR-style" throughout paper-facing content. Rationale: "faithful" implicitly suggests best-effort reproduction and invites comparison against the original SetR paper's reported numbers; "SetR-style" is the standard academic phrasing for "method adapted from another paper's idea but on a different substrate" (cf. "BERT-style", "GPT-style"). This rename does not change implementation, only paper-facing nomenclature. Existing file paths and report directory names on disk (e.g., `reports/setr_faithful_proprag_full1000_20260507/`) are kept as-is — they are filesystem artifacts and not paper-facing.

