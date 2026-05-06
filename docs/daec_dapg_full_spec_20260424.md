# DAEC / DAPG Full Specification

Date: 2026-04-24

2026-04-28 status update:

This specification is historical. Phase 2.5 full1000 controls rejected the
DAEC-DAPG / local absorption route as the active method plan: direct NV-Embed
question cosine outperformed query absorption and demand-union absorption, and
multi-channel delayed aggregation failed as a mechanism. The current source of
truth is `docs/daec_l1_findings_decision_20260428.md`.

This document was the converged research, method, and execution specification
for **Demand-Aligned Evidence Composition (DAEC)** and its end-to-end
instantiation **Demand-Aligned Proposition Graph Retrieval (DAPG)** before the
2026-04-28 negative gates.

Original short version:

- **DAEC is the paper's main line**: a retriever-agnostic evidence composition framework that selects a reader-ready top-`k` document set from a fixed pool by optimizing demand-aligned noisy-OR coverage under frozen binding assignments.
- **DAPG is an instantiation, not the survival condition**: it builds a proposition graph and uses the same DAEC objective for pool expansion and final composition.
- **The decisive Layer-1 experiment is `PropRAG top-100 pool + DAEC > PropRAG top-5`**. If this holds on at least 2/3 datasets, the paper has a strong retriever-agnostic composition claim even if DAPG does not beat PropRAG as a standalone retriever.

## A. Paper Framing

### A.1 Core Claims

Claims should be ordered by reviewer value:

1. Multi-hop QA's main bottleneck is evidence set composition, not only single-document ranking. The quantitative backbone is the oracle decomposition between `reorder@20` and `select@100`.
2. This bottleneck is retriever-agnostic. The strongest test is whether `PropRAG pool + DAEC` improves over `PropRAG top-5` under the same reader and top-5 protocol.
3. DAEC solves the bottleneck with a demand-aware, binding-aware, fixed-budget set-level objective that can plug into different retriever pools.
4. DAPG shows the same composition objective can drive an end-to-end proposition-graph retrieval system.

Claims 1 and 2 are the paper's safety layer. Claims 3 and 4 are upside.

### A.2 Title Candidates

Preferred:

```text
Demand-Aligned Evidence Composition for Multi-Hop Retrieval
```

Alternative:

```text
Multi-Hop QA is a Composition Problem: Evidence Set Selection under Variable Binding
```

Avoid making "graph" or "proposition" the headline. Those are instantiations, not the core contribution.

### A.3 Venue Targeting

- Main target: EMNLP / ACL main, if Layer 3 or strong Layer 1 transfer holds.
- Safety target: EMNLP / ACL Findings, if Layer 1 holds.
- Not primary targets: ICLR / NeurIPS main, because the theory novelty is not the dominant contribution; CIKM / SIGIR, because the audience is less aligned with the reader-utility and multi-hop QA framing.

## B. Problem Definition

### B.1 Symbols

- Corpus `C = {d_1, ..., d_n}`. A document is the smallest reader-visible unit.
- Each document `d` is parsed into propositions `Z(d) = {z_{d,1}, ..., z_{d,m_d}}`.
- DAPG uses a tripartite graph `G = (V_D union V_Z union V_E, E)`:
  - `V_D`: documents.
  - `V_Z`: propositions.
  - `V_E`: canonical entities.
  - Edges: `z <-> d` and `z <-> e`.
- Proposition-proposition relations are induced by two-hop paths `z_a -> e -> z_b`. Do not introduce extra heterogeneous edge types in the core method unless an ablation proves they matter.
- Question `q`.
- Frozen reader `R`, currently Qwen3-8B no-think.
- Pool budget `B = 100`.
- Final evidence budget `k = 5`.

### B.2 Demand Graph

The query analyzer outputs a demand graph:

```text
D_q = (R_q, Dep_q, T_q)
```

where:

- `R_q = {r_1, ..., r_m}` are evidence requirements.
- `Dep_q` is a DAG. Edge `r_i -> r_j` means `r_j` depends on the answer or entity bound by `r_i`.
- `T_q` maps each requirement to one of:
  - `lookup`
  - `bridge`
  - `comparison`
  - `aggregation`

The type distinction is method-critical:

- `lookup` and `bridge` are document-satisfiable retrieval demands. They participate in witness modeling and the DAEC objective.
- `comparison` and `aggregation` are answer operators. They do not create binding slots, do not enter `F_b`, and are handled by the reader using the selected evidence.

This is the paper-safe formalization of the earlier repairable filter. It avoids a regex veto list in the main method.

### B.3 Binding Space

For dependent requirements, latent slots are introduced. A complete assignment is a binding `b`. The enumerated binding set is:

```text
B_q = {b_1, ..., b_M'}
```

where `M'` is controlled by top-`M` enumeration per dependency chain.

Rules:

- Single chain: enumerate top-`M` candidate entities.
- Multi-chain: enumerate independently per chain and combine, with factorization to avoid blow-up.
- No dependency: use a single empty binding.
- Operator nodes (`comparison`, `aggregation`) do not introduce binding slots.

Main prior:

```text
p(b) = 1 / |B_q|
```

Appendix sensitivity:

- retrieval-proportional binding prior.

### B.4 Output

The system outputs:

```text
(b*, S*)
```

where:

- `b* in B_q`
- `S* subset P_q`
- `|S*| <= k`
- `P_q subset C`
- `|P_q| <= B`

The reader receives the top-`k` document set `S*` and produces answer `a = R(q, S*)`.

## C. Method

### C.1 Offline Stage: Proposition Graph Construction

DAPG performs offline proposition graph construction. DAEC itself can be used without this stage when applied to existing pools such as HippoRAG, dense-only, or PropRAG.

#### Step 1: Proposition Extraction

For each document `d`, extract self-contained propositions with Qwen3-8B no-think and structured JSON:

```json
[
  {
    "proposition": "atomic self-contained claim",
    "entities": ["entity 1", "entity 2"],
    "confidence": 0.0
  }
]
```

Prompt contract:

```text
Decompose the following passage into self-contained propositions.
Each proposition must be a single atomic claim that can be understood
without the rest of the passage. List the named entities mentioned.

Passage: {d.text}
```

Cache path:

```text
propositions_cache/{dataset}/{doc_id}.json
```

#### Step 2: Soft Entity Linking

For each proposition `z`, produce soft `(z, e, conf)` links:

1. Match against title and alias tables.
2. Cluster or match unmatched surface forms by embedding.
3. Store confidence in `[0, 1]`.

Do not use hard entity linking as the sole path. It is a single point of failure, especially on MuSiQue.

#### Step 3: Graph Build

Graph nodes:

- documents
- propositions
- canonical entities

Graph edges:

- `z <-> d`: proposition source document.
- `z <-> e`: proposition mentions entity, weighted by mention confidence.

Embeddings:

- proposition embedding `v_z`
- document embedding `v_d`
- entity embedding `v_e`

### C.2 Query-Time Stage 1: Demand Graph Extraction

Use one Qwen3-8B no-think call with structured JSON:

```json
{
  "requirements": [
    {
      "id": "r1",
      "type": "lookup",
      "text": "...",
      "parents": [],
      "variable": null
    },
    {
      "id": "r2",
      "type": "bridge",
      "text": "...",
      "parents": ["r1"],
      "variable": "x"
    },
    {
      "id": "r3",
      "type": "comparison",
      "text": "...",
      "parents": ["r2"],
      "variable": null
    }
  ]
}
```

The prompt must define all four requirement types and include examples. The key task is separating retrieval demands (`lookup`, `bridge`) from answer operators (`comparison`, `aggregation`).

### C.3 Query-Time Stage 2: Binding Enumeration

For every independent bindable chain:

1. Retrieve top documents or propositions for the root requirement.
2. Extract candidate entities from linked entities in the retrieved evidence.
3. Keep top-`M` candidates.
4. Substitute each candidate into dependent child requirements.
5. Repeat for downstream requirements until a leaf or operator node.

Main value:

```text
M = 5
```

Diagnostics must report:

- binding recall@`M` for `M in {1, 3, 5, 10, 20}`;
- final EM/F1 sensitivity over `M`;
- binding-error case reduction.

### C.4 Query-Time Stage 3: Requirement-Conditioned Absorbing Diffusion

For each `(requirement r_i, binding b)` pair, compute a proposition-level witness signal.

For a bound requirement `r_i^b`, define a seed distribution over propositions by semantic similarity:

```text
s_{i,b}(z) proportional to exp(cos(v_{r_i^b}, v_z) / tau)
```

limited to top-`K_seed` propositions.

Absorbing walk:

- Random walk is on `V_Z union V_E`.
- Transition `z -> e` uses mention confidence.
- Transition `e -> z` uses normalized edge weights.
- Document nodes are absorbing targets through `z -> d`.
- Hitting probabilities are computed by PPR-style sparse iteration.

Soft binding compatibility:

```text
beta_{i,b}(z) =
  max(
    max_e alias_match(z, e),
    alpha_fuzzy * max_e cos(v_e, v_z)
  )
```

Main value:

```text
alpha_fuzzy = 0.7
```

Proposition witness probability:

```text
omega_{i,b}(z) = sim(r_i^b, z) * h_{i,b}(z) * beta_{i,b}(z)
```

Document witness probability uses noisy-OR over propositions:

```text
phi_{i,b}(d) = 1 - prod_{z in Z(d)} (1 - omega_{i,b}(z))
```

Critical theoretical constraint:

```text
All phi_{i,b}(d) must be precomputed before selecting S.
```

This is what makes the frozen-binding submodular theorem valid.

### C.5 Query-Time Stage 4: Pool Expansion

Use the mixture objective:

```text
F_mix(S) =
  sum_{b in B_q} p(b)
    sum_{i: T(r_i) in {lookup, bridge}}
      pi_i * (1 - prod_{d in S}(1 - phi_{i,b}(d)))
```

where:

```text
pi_i = 1 / number_of_retrieval_demands
```

Pool expansion:

```text
P_q = Greedy_{|P| = B}(F_mix, C_cand)
```

Candidate space `C_cand` is the set of documents reached by diffusion above a small threshold, typically yielding 500 to 2000 candidate documents per query.

Rationale:

- Pool expansion should preserve multiple plausible bindings for recall.
- It is acceptable for the pool to be diverse across bindings.

### C.6 Query-Time Stage 5: Final Composition

For each binding `b`, independently solve:

```text
F_b(S) =
  sum_i pi_i * (1 - prod_{d in S}(1 - phi_{i,b}(d)))

S_b = Greedy_{|S| = k}(F_b, P_q)
```

Then choose:

```text
(b*, S*) = argmax_{b in B_q} p(b) * F_b(S_b)
```

Rationale:

- Pool expansion uses mixture for recall.
- Final composition enforces binding coherence.
- The final top-5 must not mix evidence from incompatible entity assignments.

### C.7 Query-Time Stage 6: Reader

Reader protocol:

```text
Frozen Qwen3-8B no-think
Top-5 document context
Raw document content in the main result
```

Reader prompt shape:

```text
Context:
[1] {title_1}: {content_1}
[2] {title_2}: {content_2}
...
[5] {title_5}: {content_5}

Question: {q}
Answer:
```

Appendix-only variant:

- include witness propositions as provenance hints in the context.

### C.8 Sufficiency Is Analysis, Not a Main Gate

Do not use a threshold sufficiency gate as the main method.

Instead:

- report prefix curves for `S_1, ..., S_5`;
- analyze whether reader utility is non-monotone in evidence set size;
- keep variable-`k` inference as an appendix variant only.

This removes a major reviewer attack around threshold tricks.

## D. Algorithm

### D.1 Offline

```python
def build_graph(corpus):
    graph = empty_tripartite_graph()
    prop_cache = {}
    for doc in corpus:
        propositions = extract_propositions_via_llm(doc)
        prop_cache[doc.id] = propositions
        for prop in propositions:
            prop.embedding = embed(prop.text)
            for entity, conf in soft_entity_link(prop):
                graph.add_edge(prop, entity, weight=conf)
            graph.add_edge(prop, doc)
    return graph, prop_cache
```

### D.2 Online

```python
def daec_retrieve(question, graph, corpus, B=100, k=5, M=5):
    demand_graph = extract_demand_graph(question)
    retrieval_reqs = [
        r for r in demand_graph.requirements
        if r.type in {"lookup", "bridge"}
    ]

    bindings = enumerate_bindings(demand_graph, graph, corpus, M=M)

    phi = {}
    for i, req in enumerate(retrieval_reqs):
        for b in bindings:
            bound_req = substitute_binding(req, b)
            seeds = seed_distribution(bound_req, graph)
            hits = absorbing_diffusion(graph, seeds)
            for doc in corpus:
                miss_prob = 1.0
                for prop in prop_cache[doc.id]:
                    sim = cos_sim(bound_req.embedding, prop.embedding)
                    beta = soft_binding_compat(prop, b)
                    omega = sim * hits[prop] * beta
                    miss_prob *= (1.0 - omega)
                phi[i, b, doc] = 1.0 - miss_prob

    p_b = {b: 1.0 / len(bindings) for b in bindings}

    def F_mix(S):
        total = 0.0
        for b in bindings:
            for i in range(len(retrieval_reqs)):
                total += p_b[b] * pi[i] * noisy_or(phi[i, b, d] for d in S)
        return total

    pool = greedy_submodular(
        F_mix,
        candidates=diffusion_reachable_docs,
        budget=B,
    )

    best_score, best_set, best_binding = float("-inf"), None, None
    for b in bindings:
        def F_b(S):
            return sum(
                pi[i] * noisy_or(phi[i, b, d] for d in S)
                for i in range(len(retrieval_reqs))
            )

        selected = greedy_submodular(F_b, candidates=pool, budget=k)
        score = p_b[b] * F_b(selected)
        if score > best_score:
            best_score, best_set, best_binding = score, selected, b

    answer = reader(question, best_set)
    return answer, best_set, best_binding
```

### D.3 Greedy Solver

```python
def greedy_submodular(F, candidates, budget):
    selected = []
    while len(selected) < budget:
        best_doc = None
        best_gain = 0.0
        base = F(selected)
        for doc in candidates:
            if doc in selected:
                continue
            gain = F(selected + [doc]) - base
            if gain > best_gain:
                best_doc = doc
                best_gain = gain
        if best_doc is None or best_gain <= 0.0:
            break
        selected.append(best_doc)
    return selected
```

## E. Theory Contract

### E.1 Theorem 1: Pool Expansion

Fix the enumerated binding set `B_q` with nonnegative prior `p(b)` and `sum_b p(b) = 1`. For every binding `b` and requirement `i`, assume `phi_{i,b}(d) in [0,1]` is precomputed independently of the selected set `S`.

Define:

```text
C_{i,b}(S) = 1 - prod_{d in S}(1 - phi_{i,b}(d))
```

Then:

```text
F_mix(S) = sum_b p(b) sum_i pi_i C_{i,b}(S)
```

is normalized monotone submodular.

Proof sketch:

For `S subset T` and `x notin T`:

```text
Delta_x C_{i,b}(S)
  = phi_{i,b}(x) * prod_{d in S}(1 - phi_{i,b}(d))
  >= phi_{i,b}(x) * prod_{d in T}(1 - phi_{i,b}(d))
  = Delta_x C_{i,b}(T)
```

So each `C_{i,b}` is monotone submodular. Nonnegative weighted sums preserve monotone submodularity.

### E.2 Corollary

The greedy algorithm for:

```text
max_{|S| <= B} F_mix(S)
```

achieves the standard `(1 - 1/e)` approximation.

### E.3 Theorem 2: Final Composition

For each fixed binding `b`:

```text
F_b(S) = sum_i pi_i C_{i,b}(S)
```

is monotone submodular, and greedy under budget `k` achieves `(1 - 1/e)`.

System-level claim:

```text
F_{b*}(S*) >= (1 - 1/e) * max_{b in B_q} max_{|S| <= k} F_b(S)
```

over the enumerated binding set.

Do not claim global optimality over all possible bindings.

### E.4 What Not To Claim

Do not claim:

- adaptive submodularity;
- end-to-end differentiability;
- that the objective is a theoretical proxy for reader utility;
- that DAPG must beat PropRAG as a retriever for the paper to be valid.

Recommended method disclosure:

```text
We use a frozen binding posterior to preserve standard submodularity guarantees.
A dynamic-posterior variant where phi_{i,b}(d) is recomputed after each greedy
step is explored empirically in the appendix; we do not claim approximation
guarantees for this variant.
```

## F. Related-Work Differentiation

The most dangerous neighbor is PropRAG. The paper must separate DAEC/DAPG by objective and output:

| Dimension | HippoRAG v2 | PropRAG | SetR | IRCoT / Self-Ask | GraphRAG | DAEC / DAPG |
|---|---|---|---|---|---|---|
| Retrieval unit | fact triple | proposition path | passage | passage | entity community | proposition witness |
| Query model | full query | full query | requirements | decomposed subqueries | full query | typed demand graph |
| Variable binding | implicit graph connectivity | implicit path | none | implicit sequential reading | none | explicit enumeration + soft posterior |
| Objective | PPR ranking | path score | trained set selection | iterative reading | community summarization | binding-conditioned noisy-OR coverage |
| Pool to final | same ranker | same beam | separated | N/A | N/A | mixture pool + coherent final |
| Output | documents | proposition paths | passage set | generated answer | summary | reader-ready documents + binding |
| Reader protocol | fixed top-k | fixed top-k | fixed top-k | iterative | open context | fixed top-5 |
| Operators | none | none | none | implicit | none | comparison/aggregation typing |
| Guarantee | none | none | none | none | none | `(1 - 1/e)` over enumerated bindings |

One-line differentiation:

```text
PropRAG retrieves proposition paths; DAEC selects reader-ready document sets by
optimizing binding-conditioned demand coverage over a fixed budget.
```

## G. Experiment Plan

### G.1 Unified Protocol

Datasets:

- 2WikiMultihopQA
- HotpotQA
- MuSiQue

Reader:

- frozen Qwen3-8B no-think.

Final context:

- top-5 documents.

Metrics:

- EM
- F1
- Recall@5
- Recall@20
- Recall@100
- latency seconds/query
- online LLM calls
- token cost
- offline indexing cost

### G.2 Main Table: Retriever-Composer Factorial

Rows:

- Dense NV-Embed-v2 pool.
- HippoRAG v2 pool.
- PropRAG pool.
- DAPG pool.
- Oracle select@100 per pool.

Columns:

- top-k baseline.
- MMR.
- DPP.
- current DtC.
- DAEC.

Core paper sentences:

- `PropRAG pool + DAEC > PropRAG top-5`: composition helps even on the strongest proposition-level graph retriever.
- `DAPG + DAEC` vs `PropRAG top-5`: end-to-end instantiation competitiveness.
- Oracle gap recovery: how much of pool headroom DAEC closes.

### G.3 Ablations

Run at least on 2Wiki and MuSiQue full-test:

| Ablation | Purpose |
|---|---|
| `DAEC - binding`, set `beta = 1` | binding contribution |
| `DAEC - demand decomposition`, use query as one requirement | decomposition contribution |
| `DAEC - noisy-OR`, use linear sum | saturation/submodular coverage contribution |
| `DAEC - typing`, include operators in objective | operator typing contribution |
| `DAEC - coherent final`, use mixture objective for final top-5 | binding coherence contribution |
| `DAEC + dynamic binding refresh` | frozen vs dynamic empirical comparison |
| `DAPG - proposition graph`, doc graph only | proposition granularity contribution |
| `DAPG - diffusion`, dense top-100 instead | graph expansion contribution |
| hard beta vs soft beta | entity-linking posterior sensitivity |
| `M in {1, 3, 5, 10}` | binding enumeration sensitivity |
| `B in {50, 100, 200}` | pool budget sensitivity |

### G.4 Mechanism Analysis

Required analyses:

1. Oracle gap recovery:

```text
(Metric_DAEC - Metric_baseline) / (Metric_oracle@100 - Metric_baseline)
```

2. Anchor vs bridge recall@5 before and after DAEC.
3. Bridge depth distribution before and after DAEC.
4. Composition changed rate.
5. Binding recall@`M`.
6. Entity-linking error propagation: hard EL vs soft EL wrong-entity distractors.
7. Reader utility prefix curve for `S_1, ..., S_5`.
8. Support-completeness partitions:
   - baseline complete but wrong;
   - baseline incomplete;
   - fixed by DAEC.
9. Cost breakdown:
   - offline indexing GPU-hours;
   - per-query demand extraction;
   - binding enumeration and diffusion;
   - final composition;
   - reader.

### G.5 Cross-Protocol Appendix

IRCoT and Self-Ask should only appear in appendix or related work, not in the main same-protocol table.

If run, report:

- online LLM calls;
- token cost;
- latency;
- final EM/F1.

## H. Implementation Plan

### H.1 Existing Infrastructure

Reuse:

- `src/hipporag/HippoRAG.py`
- `src/hipporag/utils/config_utils.py`
- `scripts/eval_causal_qwen3.py`
- current DtC implementation as the Layer-1 DAEC approximation.
- `scripts/export_proprag_pool.py` for fixed PropRAG pool export.

### H.2 Proposed New Modules

```text
src/daec/
├── demand_graph.py
├── binding.py
├── proposition_graph.py
├── diffusion.py
├── witness.py
├── composer.py
├── reader_adapter.py
└── daec_pipeline.py
```

Responsibilities:

- `demand_graph.py`: LLM demand extraction.
- `binding.py`: frozen binding enumeration and priors.
- `proposition_graph.py`: tripartite graph construction.
- `diffusion.py`: absorbing walk / PPR-style hitting probabilities.
- `witness.py`: `omega` and `phi` computation.
- `composer.py`: mixture pool expansion and coherent final composition.
- `reader_adapter.py`: Qwen3-8B no-think reader interface.
- `daec_pipeline.py`: end-to-end orchestration.

### H.3 Fixed Hyperparameters

All main hyperparameters should be fixed across datasets:

| Hyperparameter | Value | Rationale |
|---|---:|---|
| `M`, binding enumeration | 5 | should cover most 2Wiki bindings |
| `B`, pool budget | 100 | matches HippoRAG / PropRAG protocol |
| `k`, final budget | 5 | fixed reader input |
| walk restart | 0.15 | standard PPR-like value |
| seed temperature | 0.1 | fixed softmax temperature |
| `K_seed` | 50 | seed set size |
| `alpha_fuzzy` | 0.7 | fixed fuzzy fallback |
| `p(b)` | uniform | avoids retriever-score entanglement |
| `pi_i` | uniform over retrieval demands | avoids per-requirement tuning |

No per-dataset tuning in the main result.

### H.4 Compute Budget

Approximate cost:

| Component | Estimated Cost |
|---|---:|
| 2Wiki proposition extraction | 5 GPU-min |
| MuSiQue proposition extraction | 1 GPU-hour |
| HotpotQA proposition extraction | 20 GPU-hours |
| entity linking, all datasets | 2 GPU-hours |
| proposition embedding, all datasets | 3 GPU-hours |
| per-query demand extraction | about 0.3s if cached |
| per-query binding/diffusion/composition | 1-3s |
| per-query reader | reader-service dependent |

### H.5 Caching

Cache:

- propositions per document;
- demand graph per query hash;
- graph assets per dataset.

Do not cache `phi` as a default. It is query-specific and should be recomputed unless profiling proves it dominates runtime.

## I. Three-Week Execution Roadmap

### Week 1: Layer 1, Retriever-Agnostic Composition

Purpose:

- lock the paper floor.
- test whether the composition claim survives outside HippoRAG's pool.

Tasks:

1. PropRAG full-test clean no-think on 2Wiki / HotpotQA / MuSiQue.
2. Oracle select@100 on dense, HippoRAG v2, and PropRAG pools.
3. Current DtC/DAEC composer on:
   - dense pool;
   - HippoRAG pool;
   - PropRAG pool.
4. Internal ablations on 2Wiki and MuSiQue:
   - `-binding`;
   - `-repair typing`;
   - `-rank prior`.

Week-1 gate:

| Condition | Interpretation |
|---|---|
| `PropRAG pool + DAEC` improves `PropRAG top-5` by `>= +2 F1` on at least 2/3 datasets | Layer-1 paper and retriever-agnostic claim are strong |
| PropRAG-pool oracle@100 has `>= +3 F1` headroom on at least 2 datasets | universal composition bottleneck is supported |
| internal ablations show `>= +1 F1` contribution per component | novelty has multiple legs |
| PropRAG pool does not improve but HippoRAG pool does | claim narrows to graph-pool-specific composition |
| dense and PropRAG pools both fail | method may be HippoRAG-specific and must be rethought |

### Week 2 First Half: Layer 2, Query-Local Proposition Graph

Purpose:

- test whether proposition-level DAEC beats document-level DAEC inside the same fixed top-100 pool.

Tasks:

1. Lazy proposition extraction/cache for pool documents.
2. Query-local tripartite graph construction.
3. Absorbing diffusion and three-factor witness.
4. Binding enumeration with `M = 5`.
5. Pilot-100 on 2Wiki, MuSiQue, and HotpotQA.

Layer-2 gate:

| Condition | Interpretation |
|---|---|
| proposition-level DAEC beats doc-level DtC on 2Wiki or MuSiQue | start Layer 3 |
| binding-error cases reduce by at least 30% | variable binding supports an independent section |
| both signals are weak | stop Layer 3 and write Layer-1 paper |
| proposition extraction coverage below 90% | fix extraction/entity linking before continuing |

### Week 2 Second Half + Week 3: Layer 3, Full Offline DAPG

Tasks:

1. Full-corpus proposition extraction.
2. Entity linking and graph build.
3. DAPG pool generation.
4. DAEC final composition and reader evaluation.
5. Full ablation table.
6. Mechanism analysis and cost table.

Week-3 gate:

| Condition | Submission Path |
|---|---|
| DAPG pool + DAEC beats PropRAG top-5 on at least 1 dataset, and PropRAG pool + DAEC beats PropRAG top-5 on at least 2 datasets | main-track target is viable |
| only PropRAG pool + DAEC beats PropRAG top-5 on at least 2 datasets | main may still be possible; Findings is safe |
| only HippoRAG pool improves | Findings with narrowed claim |
| no composition line improves | stop and debug |

## J. Reviewer Attack Plan

| Attack | Defense |
|---|---|
| "Just a HippoRAG add-on" | Main table includes PropRAG pool + DAEC and dense pool + DAEC |
| "PropRAG already does propositions" | Different objective and output: proposition paths vs reader-ready document set |
| "LLM decomposition is unfair" | Cost table; ablation giving decomposition to other composers |
| "Gains are from retriever, not composition" | Retriever-composer factorial table and oracle gap recovery |
| "Regex / threshold trick" | no sufficiency threshold gate; operator typing comes from demand graph |
| "Not end-to-end" | DAPG section and cost accounting |
| "Dataset-specific tuning" | single hyperparameter set across datasets |
| "Reader-specific overfitting" | cross-reader appendix |
| "Protocol mixing" | main table restricted to fixed top-5; cross-protocol only in appendix |
| "Submodular claim suspicious" | frozen binding theorem and explicit no adaptive-submodularity claim |
| "Entity linking breaks MuSiQue" | binding recall@M and soft-vs-hard EL ablation |
| "Why not adaptive submodular" | explain adaptive monotonicity is not guaranteed |
| "Graph underperforms vanilla RAG" | DAPG ablation and fallback to retriever-agnostic DAEC claim |

## K. Paper Structure

1. Introduction:
   - hero example;
   - oracle composition bottleneck;
   - DAEC and DAPG overview.
2. Related Work:
   - graph RAG;
   - set selection;
   - query decomposition;
   - utility-aware context construction.
3. Problem and Analysis:
   - formal setup;
   - oracle decomposition;
   - bridge bottleneck;
   - MMR/DPP negative result.
4. Method: DAEC:
   - demand graph;
   - frozen binding;
   - noisy-OR objective;
   - theorem and algorithm.
5. Method: DAPG:
   - tripartite proposition graph;
   - diffusion witness;
   - pool expansion.
6. Experiments:
   - fixed top-5 protocol;
   - main factorial table;
   - ablations;
   - mechanism analysis.
7. Discussion:
   - entity linking limits;
   - long-chain bindings;
   - DAPG vs PropRAG caveat.
8. Conclusion.

Appendix:

- proof details;
- full ablations;
- cross-reader transfer;
- dynamic binding variant;
- cross-protocol IRCoT/Self-Ask;
- failure cases;
- prompts;
- hyperparameter sensitivity.

## L. Current Week Action List

Layer 1 has been executed under the fixed-pool protocol. The completed sequence was:

1. PropRAG full-test clean no-think on all three datasets.
2. Export PropRAG fixed top-100 pools.
3. Oracle select@100 on PropRAG pool.
4. DtC/DAEC composer on PropRAG pool.
5. Dense top-100 pool export.
6. Oracle select@100 and DtC/DAEC composer on dense pool.
7. Internal ablations:
   - `-binding`;
   - `-repair typing`;
   - `-rank prior`.
8. Failure taxonomy over all six Layer-1 full1000 reports.
9. Paired bootstrap/sign-flip significance analysis.
10. Targeted cross-pool `-binding` ablations.

Historical implementation artifacts:

- `docs/daec_dapg_layered_plan_20260424.md`
- `scripts/export_proprag_pool.py`
- `scripts/export_dense_pool.py`
- `scripts/eval_causal_qwen3.py --external_pool_json`
- `scripts/summarize_layer1_results.py`
- `scripts/analyze_failure_taxonomy.py`
- `scripts/analyze_layer1_significance.py`
- `run_logs/run_layer1_proprag_pool_eval_20260424.sh`
- `run_logs/run_layer1_proprag_pool_ablation_20260424.sh`
- `run_logs/run_layer1_targeted_nobinding_20260424.sh`
- `run_logs/daec_layer1_status_20260424.txt`

Historical frozen state:

- No active Layer-1 DtC/DAEC evaluation jobs are expected to be running.
- Results are consolidated in `research_memory/emnlp_expand_then_compose/16_layer1_retriever_agnostic_composition_20260424.md`.
- Follow-up taxonomy, significance, and binding ablation results are consolidated in `research_memory/emnlp_expand_then_compose/17_layer1_followup_taxonomy_significance_20260424.md`.

## M. Original One-Sentence Commitments

- Original commitment: DAEC was the main contribution.
- Original commitment: DAPG was an instantiation.
- 2026-04-28 update: DAPG / local absorption is no longer the active method
  direction; use `docs/daec_l1_findings_decision_20260428.md` for current
  execution decisions.
- PropRAG is both a competitor and a substrate.
- Frozen binding is required for the theorem.
- Comparison and aggregation are answer operators, not retrieval demands.
- The main paper should not rely on threshold gates.
- The main experiment is a retriever-composer factorial table.
- The decisive row is `PropRAG pool + DAEC`.
