# DAEC / DAPG Layered Execution Plan

Date: 2026-04-24

2026-04-28 status update:

- DAPG / local absorption is no longer the active plan.
- Phase 2.5 full1000 controls showed graph absorption is consistently below
  direct dense-query retrieval in the current substrate.
- The active direction is DAEC-L1 fixed-pool projection plus negative ablation
  analysis.
- Current source of truth: `docs/daec_l1_findings_decision_20260428.md`.

Historical full specification:

- `docs/daec_dapg_full_spec_20260424.md`

This file is the historical execution summary. It is no longer the active source
of truth for paper framing or execution. Use
`docs/daec_l1_findings_decision_20260428.md` for the current decision.

## Historical Framing

The paper should be framed around **Demand-Aligned Evidence Composition (DAEC)**:

- Multi-hop QA has a fixed-budget evidence-set composition bottleneck, not just a pointwise ranking bottleneck.
- This bottleneck should be tested across retrieval substrates, especially HippoRAG v2, dense-only retrieval, and PropRAG.
- PropRAG is not only a competitor. It is also a strong retrieval substrate. If `PropRAG top-100 + DAEC > PropRAG top-5`, the result supports a stronger claim: even proposition-level graph retrievers leave composition headroom.

The end-to-end extension is **DAPG: Demand-Aligned Proposition Graph Retrieval**:

- DAPG is a proposition-graph instantiation of DAEC.
- DAPG should only be promoted if Layer 1 and Layer 2 gates pass.
- The safe paper does not require DAPG to beat PropRAG as a retriever. The safe paper requires DAEC to improve strong fixed pools.

## Method Contract

Main method claim:

> DAEC selects a reader-ready top-k evidence set by maximizing demand coverage under frozen binding assignments, rather than reranking documents independently.

Theoretical contract:

- Enumerate a small set of frozen binding assignments `B_q`.
- For each binding `b`, precompute document/proposition support scores `phi_{i,b}(d)`.
- Define noisy-OR coverage:

```text
F_b(S) = sum_i pi_i * (1 - prod_{d in S}(1 - phi_{i,b}(d)))
```

- For fixed `b`, `F_b` is monotone submodular.
- Greedy selection has the standard `(1 - 1/e)` guarantee for the fixed-binding objective.
- Final selection chooses the best coherent binding assignment:

```text
(b*, S*) = argmax_b p(b) * F_b(S_b)
```

Do not claim adaptive submodularity. Dynamic binding refresh can be an empirical appendix variant only.

## Binding Policy

Binding enumeration must be explicit:

- Single dependency chain: enumerate top-`M` candidate entities for the dependent slot.
- Multi-chain dependency: factorize independent chains to avoid combinatorial explosion.
- No dependency: use a single empty binding assignment.
- Comparison/aggregation nodes are operator nodes. They do not create binding slots and do not enter the witness coverage objective.

Binding prior:

- Main result: uniform `p(b) = 1 / |B_q|`.
- Appendix sensitivity: retrieval-proportional prior.

Diagnostics required:

- Binding recall@`M`, with `M in {1, 3, 5, 10, 20}`.
- Final EM/F1 sensitivity over `M`.
- Binding-error reduction on 2Wiki and MuSiQue.

## Layer 1: Retriever-Agnostic Composition

Layer 1 is the current priority and paper safety layer.

Required experiments:

1. PropRAG clean no-think full-test on 2Wiki / HotpotQA / MuSiQue.
2. Export fixed top-100 pools for PropRAG.
3. Compute oracle select@100 on three pools:
   - dense-only pool
   - HippoRAG v2 pool
   - PropRAG pool
4. Run DAEC/DtC composer on:
   - PropRAG pool
   - dense-only pool
   - HippoRAG v2 pool
5. Internal ablations:
   - `-binding`
   - `-repair typing`
   - `-rank prior`

Go/no-go:

- `PropRAG top-100 + DAEC` improves `PropRAG top-5` by at least `+2 F1` on at least 2/3 datasets: retriever-agnostic claim is strong.
- Oracle select@100 has at least `+3 F1` headroom on at least two pools: composition bottleneck is substrate-general.
- If PropRAG pool has little oracle headroom on HotpotQA, narrow the claim: composition headroom is large on bridge-heavy datasets and limited on shallow/high-recall datasets.

## Layer 2: Query-Local Proposition Graph

Layer 2 should start only after Layer 1 has usable results.

Goal:

- Test whether proposition-level DAEC is stronger than document-level DAEC inside the same fixed top-100 pool.

Protocol:

- Cache per-document propositions offline.
- At query time, retrieve top-100 docs, load their cached propositions, and build a query-local graph.
- Run DAEC over propositions, then project to reader-ready documents.

Go/no-go:

- Proposition-level DAEC improves over document-level DAEC on 2Wiki or MuSiQue.
- Binding-error cases reduce by at least 30%.

## Layer 3: Full Offline DAPG

Layer 3 is high-risk and should start only if Layer 2 passes.

Components:

- Full-corpus proposition extraction.
- Entity linking plus alias fallback.
- Proposition graph with entity-sharing and predicate-compatibility edges.
- Demand-conditioned graph expansion.
- DAEC final composition.

Success condition:

- Either DAPG own pool + DAEC beats PropRAG top-5 on at least one dataset, or PropRAG pool + DAEC beats PropRAG top-5 on at least two datasets.

The second condition is more important for the paper's core claim.

## Immediate Implementation State

Implemented on 2026-04-24:

- `scripts/export_proprag_pool.py`: exports PropRAG fixed top-100 pools.
- `scripts/eval_causal_qwen3.py --external_pool_json`: evaluates an external fixed pool under HippoRAG reader, oracle, and setwise selector code.
- External-pool smoke passed on 2Wiki:
  - retrieval-only smoke: `run_logs/external_pool_eval_smoke_2wiki1_20260424.json`
  - DtC smoke: `run_logs/external_pool_dtc_smoke_2wiki1_20260424.json`
  - mapped PropRAG pool docs: 100/100

Layer-1 completed queue:

1. Exported PropRAG top-100 pools for 2Wiki / HotpotQA / MuSiQue.
2. Ran oracle select@100 on PropRAG pools.
3. Ran DAEC/DtC on PropRAG pools.
4. Exported dense-only top-100 pools and ran the same oracle + DAEC protocol.
5. Ran failure taxonomy and paired significance analysis.
6. Ran targeted `nobinding` ablations to test dependency binding beyond `2Wiki + PropRAG`.

Frozen Layer-1 memos:

- `research_memory/emnlp_expand_then_compose/16_layer1_retriever_agnostic_composition_20260424.md`
- `research_memory/emnlp_expand_then_compose/17_layer1_followup_taxonomy_significance_20260424.md`

Historical next step before the 2026-04-28 decision:

- Write the paper skeleton around retriever-agnostic fixed-pool composition.
- Manually audit `MuSiQue` loss cases before making fine-grained qualitative claims.
