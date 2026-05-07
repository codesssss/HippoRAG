# EMNLP Main North Star

Last updated: 2026-05-07

## Current Paper Framing (LOCKED 2026-05-07)

Final framing is in `41_dbec_paper_framing_locked_20260507.md`. All paper-facing content from this point uses **DBEC** naming. Read that document before writing or re-litigating positioning.

Locked decisions:
- Method name: **DBEC** (Dependency-Bound Evidence Composition), not DAEC
- Title: **DBEC: Dependency-Bound Evidence Composition for Multi-Hop Retrieval-Augmented Generation**
- Scope: Multi-Hop RAG (not general RAG, not Multi-Hop QA)
- Story arc: composition gap → dependency-bound mechanism → controlled evaluation → under-selection mechanism → honest budget limitation
- SetR is one of 5 baselines, not paper narrative center

The "Expand-then-Compose" framing below is superseded; it is preserved here for historical context only.

---

## Objective (historical, superseded)

Build an EMNLP main-track paper around a clean empirical and methodological claim:

> The main bottleneck in multi-hop GraphRAG is not better pointwise reranking of individual documents. The bottleneck is composing a top-5 evidence set that contains a complete reasoning chain. Harder queries therefore require expanding the candidate pool before composing the final evidence subset.

Short name for the story:
- `Expand-then-Compose` (historical; superseded by DBEC framing)

Alternative method names:
- `Chain-Complete Evidence Selection`
- `Bridge-Aware Evidence Composition`

## Canonical Framing

1. Keep the retriever backbone fixed to the validated HippoRAG `legacy_fact_graph` baseline.
2. Treat prior causal / general-graph retrieval variants as negative evidence that motivates a pivot.
3. Focus the paper on evidence composition over retrieved candidates, not on reader prompting and not on new extraction graphs.
4. Use oracle-select results to establish headroom, then close part of that headroom with a simple non-oracle setwise selector.

## Main Claims We Want To Support

1. Pointwise ranking under-serves hard multi-hop queries because bridge docs receive low direct query relevance.
2. Hard queries require much deeper support pools than easy queries.
3. Oracle evidence selection from a larger pool yields a substantial EM/F1 ceiling gain.
4. The gain comes from evidence composition, not only from reordering the existing top-5.
5. A simple setwise or bridge-aware heuristic can recover a meaningful fraction of the oracle headroom.

## Scope

Backbone:
- HippoRAG aligned `legacy_fact_graph` retrieval

Primary datasets:
- `2WikiMultihopQA` for structural analysis
- `HotpotQA` for cross-dataset ceiling generalization
- `MuSiQue` for hard multi-hop generalization

Primary metrics:
- `EM`, `F1`
- `FS@K` or `full_support_in_pool_rate`
- minimal full-support depth
- bucketed results by query difficulty
- bridge/anchor recall and depth on `2Wiki`

## Non-Goals

Do not drift into these paper framings unless new evidence clearly forces it:

- "causal retrieval"
- "better pointwise reranker"
- "reader prompt engineering"
- "candidate injection into top-K"
- "new extraction graph beats legacy retrieval"

## Success Criteria

Minimum paper-ready package:

1. Oracle-select sweep on `2Wiki`, `HotpotQA`, and `MuSiQue`
2. Strong structural analysis on `2Wiki`
3. Bridge-vs-anchor evidence on `2Wiki`
4. One simple non-oracle method that improves over baseline
5. One cross-dataset summary table / figure

## Read Order

1. `00_north_star.md`
2. `01_claim_ledger.md`
3. `04_result_registry.md`
4. `03_experiment_board.md`
5. `02_decision_log.md`
