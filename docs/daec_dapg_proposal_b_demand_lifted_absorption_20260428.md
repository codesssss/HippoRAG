# Proposal B: DAEC-DAPG as Demand-Lifted Absorbing Evidence Assembly

Date: 2026-04-28

Status: archived negative proposal after Phase 2.5 hard gates.

**2026-04-28 decision update:** this proposal is no longer the active execution
plan. Phase 2.5 dense-only full1000 controls showed that direct NV-Embed question
cosine beats query absorption and demand-union absorption in every hop bucket;
multi-channel delayed aggregation also failed to beat channel-sum or single-query
controls. Do not implement Phase 3 / full DAPG from this document. The current
source of truth is:

- `docs/daec_l1_findings_decision_20260428.md`

This file is retained as the historical rationale and negative-ablation record
for the DAPG route.

Original intent: this document reframed the selector-only direction as a full
DAEC-DAPG train-free evidence assembly method. That intent is now rejected by
the Phase 2.5 hard gates. Keep the document only as historical design context
and as a record of the hypotheses that failed.

## Executive Position

The method should not be framed as a selector, reranker, generic GraphRAG
variant, or PPR variant.

The defensible core is:

```text
scalar document relevance
  -> demand-binding-document support tensor
  -> delayed submodular reader-budget projection
```

Existing graph RAG systems typically collapse a query into one graph relevance
signal:

```text
score(d | q)
```

DAEC-DAPG should instead estimate:

```text
phi[i,b,d] = P(document d supports demand i under binding b)
```

The central claim is not decomposition, graph construction, propagation, or
coverage selection in isolation. All have close prior art. The claim is that
multi-hop evidence assembly should preserve requirement-level support channels
through graph propagation and aggregate them only at the final reader-budget
projection stage.

## Original Non-Negotiable Framing

Historical proposed wording, now rejected as the active framing. Use only as a
record of the hypothesis that Phase 2.5 tested.

Original use:

```text
training-free query-to-answer evidence assembly
training-free inference-time evidence assembly
demand-lifted absorbing evidence graph
support tensor phi[i,b,d]
delayed aggregation
reader-budget projection
```

Original avoid:

```text
selector
reranker
graph boost
additive graph rerank
multi-channel PPR as the main novelty
we invented proposition graphs
we invented question decomposition
end-to-end trainable
```

If "end-to-end" is used, qualify it explicitly:

```text
end-to-end at inference time, not end-to-end trainable
```

## Relation To Current Code And Results

The current clean implementation:

```text
select_daec_noisyor_positions
phi = embedding similarity over bound requirements and documents
noisy-OR greedy projection
```

should be named:

```text
DAEC-L1
```

Role:

```text
document-level phi=sim ablation
diagnostic for demand coverage
not the final main method
```

Current full1000 DAEC-L1 results:

| Setting | Baseline EM/F1 | DAEC-L1 EM/F1 | Delta |
| --- | ---: | ---: | ---: |
| 2Wiki dense pool100 | 0.4550 / 0.4991 | 0.4980 / 0.5541 | +0.0430 / +0.0550 |
| Hotpot dense pool100 | 0.5950 / 0.7106 | 0.6120 / 0.7267 | +0.0170 / +0.0161 |
| MuSiQue dense pool100 | 0.2980 / 0.3896 | 0.2900 / 0.3818 | -0.0080 / -0.0078 |
| 2Wiki PropRAG pool100 | 0.5790 / 0.6503 | 0.6130 / 0.6828 | +0.0340 / +0.0325 |

Interpretation:

```text
DAEC-L1 validates demand-wise projection on 2Wiki/Hotpot/PropRAG pools,
but MuSiQue exposes the limitation of embedding-only phi.
```

Proposal B upgrades `phi` from document semantic similarity to graph-derived
absorbing support.

## Related-Work Collision Boundary

This direction has close neighbors. The paper must handle them explicitly.

| Direction | Representative pressure | Overlap | Defense |
| --- | --- | --- | --- |
| Query-level graph propagation | HippoRAG, PropRAG | graph diffusion / PPR / proposition paths | They output scalar `score(d)`. DAEC-DAPG outputs `phi[i,b,d]`. |
| Sub-question graph RAG | SubQRAG | sub-question decomposition plus graph retrieval | SubQRAG sequentially retrieves triples / graph memory. DAEC-DAPG estimates parallel/factorized demand support channels and delays aggregation. |
| Dependency-aware query resolution | PankRAG | dependency-aware subquestion planning / reranking | PankRAG reranks contexts using dependency information. DAEC-DAPG derives an absorption support tensor before projection. |
| Bridge-conditioned retrieval | BridgeRAG-style variants | later-hop evidence conditioned on bridge evidence | Closest neighbor conceptually, but no public implementation is available for controlled comparison. Mention briefly in related work; do not use as a main-table baseline. |
| Decomposition + rerank | QD-RAG, QDRAG, RT-RAG | atomic subqueries / reasoning trees / reranking | Decomposition is not claimed as novelty. The contribution is delayed aggregation over a support tensor. |
| Coverage context selection | S-RAG, PureCover | submodular / coverage selection | Noisy-OR alone is not novelty. It is the projection layer for graph-derived `phi`. |
| Path/subgraph retrieval | ToG-2, G-Retriever, PathRAG, HopRAG | graph path retrieval / subgraph evidence | DAEC-DAPG does not primarily retrieve one path or subgraph; it estimates demand-document support channels. |

Minimum mandatory baseline pressure:

```text
Dense
HippoRAG
PropRAG
DAEC-L1
DAEC-DAPG
same-substrate query-level PPR
same-substrate channel-sum propagation
same-substrate delayed noisy-OR projection
```

If SubQRAG / PankRAG / S-RAG / PureCover cannot be run cleanly, discuss them
directly in related work and include the closest implementable controlled
baselines.

## Main Contributions

### Contribution 1: Demand-Conditioned Absorption Tensor

Formulate multi-hop retrieval as estimating:

```text
phi[i,b,d]
```

instead of scalar:

```text
score(d | q)
```

Each channel is:

```text
c = (i,b)
```

where `i` is a retrieval-active demand and `b` is a binding candidate.

### Contribution 2: Delayed Aggregation By Submodular Projection

Do not aggregate channels during propagation.

Keep:

```text
phi[i,b,d]
```

until the final context-budget stage, then project by noisy-OR coverage.

This is the mathematical difference from any linear mixed-source graph ranker.

### Contribution 3: Controlled Same-Substrate Evaluation

Evaluate whether demand lifting itself matters, independent of graph substrate
or pool effects:

```text
same graph
same frontier
same source candidates
same reader
```

Compare:

```text
query-level PPR
mixed-source absorption
multi-channel + sum
multi-channel + delayed noisy-OR
```

If delayed noisy-OR does not win this controlled comparison, the method collapses
back into a selector / reranking story.

## Graph Objects

Separate two graphs.

Graph construction is a method component, but not the standalone novelty claim.
The paper should not say "we invent proposition/entity graphs." It should say:

```text
We construct an absorption-ready evidence graph whose transition operator is
designed to estimate demand-conditioned document support.
```

This keeps the method broader than a fixed-pool selector while avoiding a
collision with HippoRAG / PropRAG graph construction claims. HippoRAG and
PropRAG remain baselines; DAEC-DAPG may reuse comparable extraction ingredients,
but it must define its own directed absorption projection, sink design, hub
normalization, and demand-conditioned source grounding.

### Index Graph

Used for offline indexing and query-time frontier expansion.

May contain bidirectional edges:

```text
D <-> P
D <-> E
P <-> E
E <-> E
```

Node types:

```text
D: document / passage nodes
P: proposition nodes
E: entity nodes
```

### Absorption Graph

Used for support estimation.

Documents are sinks/readout nodes. They should have incoming edges and no
outgoing propagation edges.

Use transient nodes:

```text
U = E union P
```

and absorbing nodes:

```text
D = documents
```

Allowed main-method absorption edges:

```text
P -> E
E -> P
P -> D
```

Use `P -> D` as the main document sink edge: propositions are the evidence
carriers and documents absorb proposition support. Treat `E -> D` as an ablation,
not the default, because entity-document absorption can collapse back into the
entity-passage mechanism already used by HippoRAG / PropRAG.

Avoid main-method direct `P <-> P` shared-entity edges unless restricted to
same-document or adjacent-sentence proximity. General shared-entity `P <-> P`
can double-count the `P -> E -> P` route and amplify hubs. Put it in ablation.

Implementation note:

```text
The absorption graph is a directed projection of the index graph.
It does not require duplicating graph storage.
When constructing the transition matrix, document rows are masked to zero
and document self-absorption is represented by the lower-right identity block.
```

## Hub Leakage Control

Graph conductance should downweight high-frequency entities at the operator
level, not via post-hoc gates.

Recommended edge weights:

```text
w(P,E) = idf(E)
```

or:

```text
w(P,E) = 1 / sqrt(deg(E))
```

Then row-normalize the transient transition matrix.

Rationale:

```text
high-frequency entities such as "United States", "film", "person", "award"
should not dominate propagation.
```

This is a natural graph normalization, not a dataset-specific rule.

## Demand Graph

Use a frozen LLM to produce a demand graph:

```text
R_q = {r_1, ..., r_m}
```

Each requirement has:

```text
unit_id
subquery
role
depends_on
anchor_mentions
expected_answer_type
satisfiable_by
```

Retrieval-active roles:

```text
lookup
bridge
```

Operator roles:

```text
comparison
aggregation
inference-only answer operators
```

Operator roles do not enter the retrieval coverage objective. They can still
inform final answering instructions and dependency structure.

Default demand weights:

```text
pi_i = 1 / number_of_retrieval_active_demands
```

No per-dataset tuning.

## Binding Candidates

Bindings are latent assignments for dependency slots:

```text
b = {slot -> entity/title/proposition/document grounding}
```

Do not implement binding as:

```text
transition boost
edge mask
post-hoc graph bonus
```

Instead, binding enters the bound requirement and source construction.

Main budgets:

```text
M = 5       # candidates per slot
Bmax = 16  # max binding assignments for full DAPG
```

Historical DAEC-L1 experiments used a cap of 25. Full DAPG should default to
`Bmax=16` for compute control and report sensitivity over:

```text
Bmax in {16, 25}
```

If a requirement has no dependency:

```text
B_i = {empty binding}
```

Binding confidence must be represented explicitly if cross-binding comparisons
are made.

Recommended:

```text
p(b | q)
```

with an executable train-free definition.

For each dependency slot `sigma = (j -> i)`, let candidate entity/title `z`
come from upstream demand `r_j`. Define a slot grounding score:

```text
u_sigma(z) =
  max_{x in Dep_j(z)}
    support_j(x) * alias_score(z,x)
```

where:

```text
Dep_j(z): upstream propositions/documents/entities in the query frontier that mention or alias-match z
support_j(x): upstream support score for x from the unbound demand r_j prepass
alias_score(z,x): 1 for exact/title match, otherwise normalized alias/embedding match in [0,1]
```

`support_j(x)` must not be the raw baseline retriever rank or retriever score.
It is a local grounding confidence for the upstream demand, computed from the
same frozen semantic gate / absorption machinery used by the method. This keeps
`p(b | q)` from becoming a hidden retriever-proportional prior.

If the upstream prepass is unavailable in a local smoke, use:

```text
support_j(x) = kappa_{j,empty,x}
```

Normalize per slot without a temperature hyperparameter:

```text
p_sigma(z | q) =
  (epsilon + u_sigma(z)) /
  sum_{z' in Candidates(sigma)} (epsilon + u_sigma(z'))
```

with fixed:

```text
epsilon = 1e-6
```

For a full binding assignment:

```text
p(b | q) = prod_{(sigma,z) in b} p_sigma(z | q)
```

Binding enumeration truncates by this product:

```text
keep top-Bmax bindings by p(b | q)
renormalize p(b | q) over the retained binding set
```

Avoid internal per-binding normalization that makes wrong bindings look equally
confident.

## Source Construction

Avoid source-level weight soup:

```text
s = lambda_req s_req + lambda_anchor s_anchor + lambda_bind s_bind
```

Use a single bound requirement:

```text
tilde_r_c = Bind(r_i, b)
```

for channel:

```text
c = (i,b)
```

Define a single nonnegative grounding score:

```text
g(tilde_r_c, v) >= 0
```

for graph nodes `v`.

This score may combine exact entity linking, alias matching, and frozen embedding
retrieval internally, but the algorithm sees only one grounding likelihood.

Normalize:

```text
s_c(v) = g(tilde_r_c, v) / sum_u g(tilde_r_c, u)
```

Do not expose multiple source weights as main hyperparameters.

## Absorbing Propagation

Let transient nodes be:

```text
U = E union P
```

and absorbing documents be:

```text
D
```

The transition matrix is:

```text
T = [ Q  R
      0  I ]
```

where:

```text
Q: U -> U
R: U -> D
I: document self-absorption
```

For channel `c=(i,b)`, source is:

```text
s_c = (s_{c,U}, s_{c,D})
```

Finite-horizon absorption:

```text
a_c^(L) =
  s_{c,D}
  + sum_{t=0}^{L-1} s_{c,U} Q_alpha^t R_alpha
```

where `Q_alpha` and `R_alpha` implement restart-smoothed transient dynamics.

One acceptable definition:

```text
Q_alpha = (1-alpha) Q + alpha 1 s_{c,U}^T
R_alpha = (1-alpha) R + alpha 1 s_{c,D}^T
```

Use a fixed:

```text
alpha = 0.15
L = 2 or 3
```

Report sensitivity, but do not tune per dataset.

Important distinction:

```text
occupancy readout h^(L)(d) is not absorption probability.
absorption readout a_c^(L)(d) is the main method.
```

## Support Tensor

Define semantic gate:

```text
kappa_{c,d} = max(0, cos(z_{tilde_r_c}, z_d))
```

This ensures:

```text
0 <= kappa <= 1
```

Define:

```text
hit[c,d] = a_c^(L)(d)
```

Do not sum-to-one normalize `hit` per channel. If a demand has no support,
forced normalization amplifies noise and creates fake coverage.

Main support tensor:

```text
phi[c,d] = kappa_{c,d} * hit[c,d]
```

or, expanded:

```text
phi[i,b,d] = kappa_{i,b,d} * hit[i,b,d]
```

Guarantee:

```text
0 <= phi[i,b,d] <= 1
```

Main ablations:

```text
phi = hit
phi = kappa * hit
phi = kappa * hit with explicit binding prior in the binding-selection layer
```

Binding priors do not enter `phi` in the main method. This keeps `phi` as a
pure in-channel support probability and keeps `phi`-AUPRC / calibration metrics
interpretable.

## Reader-Budget Projection

For fixed binding `b`, requirement coverage:

```text
C_{i,b}(S) = 1 - prod_{d in S} (1 - phi[i,b,d])
```

Objective:

```text
F_b(S) = sum_i pi_i C_{i,b}(S)
```

Projection:

```text
S_b = argmax F_b(S)
      subject to budget(S) <= B_ctx
```

Budget can be:

```text
|S| <= k
```

or preferably:

```text
sum_{d in S} length(d) <= B_ctx
```

Token-budget projection is more fair across evidence granularities and aligns
with RAG context construction.

Greedy projection is justified because for fixed `b`, `F_b` is monotone
submodular when:

```text
0 <= phi[i,b,d] <= 1
pi_i >= 0
```

Marginal gain:

```text
Delta_d C_{i,b}(S)
  = phi[i,b,d] * prod_{e in S} (1 - phi[i,b,e])
```

If `A subseteq B`, then:

```text
prod_{e in B}(1 - phi[i,b,e])
  <= prod_{e in A}(1 - phi[i,b,e])
```

so diminishing returns holds.

For cardinality budget, classic greedy gives the standard `(1 - 1/e)`
approximation for fixed binding. For token budget, use the standard monotone
submodular knapsack greedy variant and report it as the reader-budget
projection layer.

## Binding Aggregation Options

### Main Option: Best Binding With Explicit Prior

Keep `phi` pure:

```text
phi[i,b,d] = kappa_{i,b,d} * hit[i,b,d]
```

For each binding, optimize the fixed-binding coverage objective:

```text
S_b = greedy(F_b)
```

Then select the binding by an explicit prior-weighted score:

```text
J_b = p(b | q) * F_b(S_b)
b* = argmax_b J_b
S* = S_b*
```

For fixed `b`, multiplying by `p(b | q)` does not affect the greedy set
selection, but it makes cross-binding selection explicit and auditable.

### Ablation: Binding-Marginalized Coverage

Alternative:

```text
C_i(S) =
  1 - prod_{b in B_i} prod_{d in S} (1 - p(b | q) * phi[i,b,d])
```

```text
F(S) = sum_i pi_i C_i(S)
```

This avoids explicit cross-binding argmax but may mix mutually incompatible
bindings in the same context. Use as ablation, not default. Keep `phi` pure in
this ablation as well; apply `p(b | q)` only inside the projection formula.

## Algorithm

```text
Algorithm: DAEC-DAPG

Input:
  question q
  index graph substrate G0
  frozen LLM decomposer
  frozen embedding model
  reader budget B_ctx or k
  binding budgets M, Bmax
  absorption horizon L

1. Extract demand graph R_q from q.
2. Keep retrieval-active demands R_q+.
3. Enumerate sparse binding candidates B_i for each demand.
4. Build query frontier over the index graph using bound requirements.
5. Convert the frontier into an absorption graph:
     transient nodes U = E union P
     document sinks D
     transition blocks Q and R
6. For each channel c=(i,b):
     tilde_r_c = Bind(r_i,b)
     s_c(v) = normalized grounding likelihood g(tilde_r_c,v)
     compute finite-horizon absorption hit[c,d]
     compute kappa[c,d]
     phi[c,d] = kappa[c,d] * hit[c,d]
7. Project phi to reader context by noisy-OR submodular greedy for each b.
8. Select b* by J_b = p(b|q) * F_b(S_b).
9. Run frozen reader on q and selected evidence S*.
```

## Key Mathematical Warning

Linear propagation alone is not enough novelty.

If `A` is a linear graph operator:

```text
sum_i pi_i A(s_i) = A(sum_i pi_i s_i)
```

Therefore:

```text
multi-channel propagation + immediate sum
```

is mathematically equivalent or near-equivalent to mixed-source query-level
propagation.

The key mechanism is:

```text
delayed aggregation:
  preserve phi[i,b,d] through propagation
  aggregate only via demand-wise noisy-OR projection
```

This equality and the delayed-aggregation response should appear in the main
method section of the paper, not only in ablations. The ablation verifies the
mechanism; the method section needs the warning to explain why batched PPR is
not the contribution.

## Required Ablations

### 1. Channel Collapse Test

Same graph, same frontier, same source candidates, same reader:

| Variant | Formula / behavior | Purpose |
| --- | --- | --- |
| Query-level PPR | scalar source `s_q(v)=g(q,v)` | standard graph retrieval on the same graph |
| Mixed-source absorption | `A(sum_i pi_i s_i^0)` | decompose demands, mix unbound sources before absorption |
| Multi-channel + sum | `score(d)=sum_i pi_i sum_b p(b|q) phi[i,b,d]` | full channels but immediate scalar aggregation |
| Multi-channel + noisy-OR | delayed aggregation over `phi[i,b,d]` | main mechanism |

Operational definitions:

```text
Query-level PPR:
  source = g(q,v), no decomposition, no binding.
  rank documents by absorption / PPR score.

Mixed-source absorption:
  sources s_i^0 come from unbound demands Bind(r_i, empty).
  mix source vectors before applying the graph operator:
    s_mix = sum_i pi_i s_i^0
  rank documents by A(s_mix).

Multi-channel + sum:
  use the same channels as the full method, including binding candidates and
  p(b|q), but collapse phi to one scalar document score before selection.

Multi-channel + noisy-OR:
  use the same phi as multi-channel + sum, but preserve channels until
  reader-budget projection.
```

If mixed-source absorption and multi-channel + sum differ numerically, report
why. The graph operator may not be exactly linear because grounding, sparse
frontier truncation, or binding enumeration introduces normalization. The paper
should still show that both early-aggregation variants are weaker than delayed
noisy-OR projection.

Expected result:

```text
multi-channel + noisy-OR > multi-channel + sum ~= mixed-source absorption
```

If this does not hold, the method is likely just graph retrieval plus selection.

### 2. Absorption vs Occupancy

| Variant | Readout | Purpose |
| --- | --- | --- |
| PPR occupancy | `h^(L)(d)` or stationary score | ordinary graph relevance |
| Absorbing hit | `a_c^(L)(d)` | document-as-evidence-sink operator |
| Absorbing hit + semantic gate | `kappa * a_c^(L)(d)` | main support tensor |

Expected result:

```text
absorbing hit should improve support calibration and long-hop support completeness.
```

### 3. Oracle Diagnostics

| Oracle | Purpose |
| --- | --- |
| Oracle demand graph | quantify decomposition bottleneck |
| Oracle binding | quantify binding enumeration bottleneck |
| Oracle support docs | quantify reader bottleneck |

If oracle support docs do not improve EM/F1, then the bottleneck is reader-side,
and DAEC-DAPG may improve Recall without improving answer quality.

### 4. Binding Diagnostics

Report:

```text
binding_count > 1 ratio
non-empty binding ratio
binding prior entropy
binding correctness if gold chain entities are available
by dataset
by hop bucket
by correct/wrong outcome
```

If binding is rare in MuSiQue 4-hop errors, emphasize demand-lifted absorption
rather than binding-lifted retrieval.

## Evaluation Metrics

Do not rely on EM/F1 alone.

Primary evidence assembly metrics:

```text
Support-Complete@k:
  1[all gold support docs are in selected context]

Support Recall@k:
  |S_k intersect G| / |G|

Noise Rate@k:
  1 - |S_k intersect G| / |S_k|
```

Additional metrics:

```text
EM / F1
Recall@5 / Recall@20 / Recall@100
Hop-bucket EM/F1: 2-hop, 3-hop, 4-hop
phi-AUPRC using gold support docs as positives
binding accuracy when gold intermediate entities are recoverable
latency
index cost
online LLM call count
token-budget performance
```

Report full-corpus retrieval and local top-100 experiments separately.

Statistical reporting:

```text
use paired bootstrap confidence intervals for EM/F1 and support metrics
report hop-bucket deltas separately
report support metric deltas even when EM/F1 is flat
```

Do not average away the main risk bucket. MuSiQue 3/4-hop should be reported as
a separate diagnostic table in Phase 0, Phase 1, and Phase 2.

## Implementation Roadmap

### Phase 0: Diagnostics Before New Method Work

Phase 0 is a hard gate before Local DAPG or Full DAPG implementation.

Run on current full1000 outputs, especially MuSiQue:

```text
baseline correct?
DAEC-L1 correct?
baseline support complete?
DAEC-L1 support complete?
reader wrong despite support complete?
```

Categorize:

| Category | Interpretation | DAPG can help? |
| --- | --- | --- |
| baseline wrong, DAEC wrong, missing support | retrieval/composition bottleneck | yes |
| baseline right, DAEC wrong, support replaced | destructive projection | yes, with better phi |
| both wrong, support complete | reader bottleneck | unlikely |
| both right | no headroom | no |
| baseline wrong, DAEC right | already solved | no |

Also run binding nontriviality diagnostics.

Run oracle ceilings in Phase 0, not after method implementation:

```text
Oracle support docs:
  replace selected evidence with gold support docs under the same reader budget.

Oracle demand graph:
  use manually/gold-aligned demands for a small diagnostic subset.

Oracle binding:
  force gold intermediate entities when recoverable from support annotations.
```

Hard gate:

```text
If oracle support docs do not improve MuSiQue 3/4-hop EM/F1,
then the main bottleneck is reader-side and DAEC-DAPG should not be promoted
as an answer-quality method for that bucket.
```

Phase 0 deliverables:

```text
1. MuSiQue 3/4-hop failure taxonomy:
     retrieval/composition bottleneck vs destructive projection vs reader bottleneck.
2. Oracle support-doc ceiling under the same reader budget.
3. Oracle demand and oracle binding ceilings on a small diagnostic subset.
4. Binding nontriviality by dataset and hop bucket.
5. Reproducible-baseline audit:
     HippoRAG / PropRAG paths and versions used for comparison.
```

Recommended artifact paths:

```text
reports/dpathrag/daec_dapg_phase0_diagnostics_20260428.md
reports/dpathrag/daec_dapg_phase0_diagnostics_20260428.json
reports/dpathrag/daec_dapg_phase0_rows_20260428.jsonl
reports/dpathrag/daec_dapg_phase0_oracle_predictions_20260428.jsonl
```

Row-level JSONL contract:

```text
qid
dataset
hop_bucket
question
answer
gold_titles
baseline_selected_titles
daec_l1_selected_titles
baseline_em
baseline_f1
daec_l1_em
daec_l1_f1
baseline_support_recall
baseline_support_complete
daec_l1_support_recall
daec_l1_support_complete
reader_wrong_despite_support_complete
binding_count
non_empty_binding
selected_binding
selected_coverage
failure_category
```

Use the existing D-PathRAG metric names where possible:

```text
support_recall
support_complete
selected_gold_count
bridge_entity_recall
```

Add `noise_rate` as:

```text
noise_rate = 1 - selected_gold_count / max(1, selected_doc_count)
```

Bridge-conditioned retrieval is handled only in related work unless a public,
reproducible implementation becomes available. Do not include a weak controlled
fallback in the main paper.

### Phase 1: Local Absorption Smoke

Purpose:

```text
test whether graph-derived phi improves over cosine phi
```

Scope:

```text
MuSiQue limit=100
2Wiki limit=100 sanity
Hotpot limit=100 sanity
```

Build local index graph from top-100 docs:

```text
docs -> cached propositions -> entities -> absorption graph
```

Variants:

| Variant | phi |
| --- | --- |
| DAEC-L1 | `cos(bound_req, doc)` |
| Local occupancy | PPR/occupancy document readout |
| Local absorption | finite-horizon absorbing hit |
| Local absorption + kappa | `kappa * hit` |
| Local absorption + kappa + explicit binding prior | pure `phi=kappa*hit`, select binding by `J_b=p(b|q)F_b(S_b)` |

Success signal:

```text
MuSiQue 3/4-hop improves over DAEC-L1
2Wiki/Hotpot do not collapse
```

This phase is not the main method; it is operator validation.

### Phase 1.5: Pool Ceiling Diagnostic

Before investing in full-corpus graph retrieval, measure whether fixed top-100
pools can contain complete evidence chains:

```text
support_recall@100
support_complete@100
by dataset
by hop bucket
especially MuSiQue 3/4-hop
```

Decision rule:

```text
If MuSiQue 4-hop support_complete@100 is low, local top-100 experiments cannot
prove complete-chain recovery. Phase 3 must focus on full graph frontier
construction rather than better fixed-pool projection.
```

Recommended artifacts:

```text
reports/dpathrag/daec_dapg_phase1_local_absorption_20260428.md
reports/dpathrag/daec_dapg_phase1_local_absorption_20260428.json
reports/dpathrag/daec_dapg_phase1_rows_20260428.jsonl
```

Phase 1 report table:

```text
dataset
hop_bucket
variant
EM
F1
Support-Complete@k
Support Recall@k
Noise Rate@k
phi-AUPRC
latency_ms_per_query
```

Minimum decision rule:

```text
If local absorption does not improve support metrics over DAEC-L1 on MuSiQue
3/4-hop, do not implement full DAPG yet. First inspect whether the failure is
graph construction, source grounding, binding enumeration, or reader ceiling.
```

### Phase 2: Same-Substrate Mechanism Test

Use one graph substrate and compare:

```text
query-level PPR
mixed-source absorption
multi-channel + sum
multi-channel + noisy-OR
```

This is the most important reviewer-defense experiment.

Concrete implementation:

```text
Graph:
  use the same local or full absorption graph for all four variants.

Frontier:
  use the same candidate frontier before propagation.

Reader:
  use the same frozen reader and context budget.

Document budget:
  use the same top-k or token budget for all variants.
```

The Phase 2 table must report both answer metrics and evidence metrics:

```text
EM/F1
Support-Complete@k
Support Recall@k
Noise Rate@k
```

Recommended artifacts:

```text
reports/dpathrag/daec_dapg_phase2_same_substrate_20260428.md
reports/dpathrag/daec_dapg_phase2_same_substrate_20260428.json
reports/dpathrag/daec_dapg_phase2_rows_20260428.jsonl
```

Phase 2 row contract:

```text
qid
dataset
hop_bucket
variant
source_definition
aggregation_timing
selected_titles
selected_doc_count
support_recall
support_complete
selected_gold_count
noise_rate
em
f1
latency_ms
```

Mechanism decision rule:

```text
The paper-level claim requires:
  multi-channel + noisy-OR > multi-channel + sum
  multi-channel + noisy-OR > mixed-source absorption
on support metrics, especially MuSiQue 3/4-hop.

If the only gain is EM/F1 without support metric gain, treat it as reader/order
interaction and do not claim demand-lifted retrieval.

### Phase 2.5: Dense-Grounding Mechanism Test

Repeat Phase 2 with soft grounding:

```text
g(tilde_r_c, v) = max(0, cos(embed(tilde_r_c), embed(v)))
```

Purpose:

```text
The lexical smoke may make noisy-OR and channel-sum nearly identical because
phi is sparse and close to binary. Dense grounding creates partial-support
overlap where delayed noisy-OR should differ from early channel-sum.
```

Decision rule:

```text
If dense-grounded multi-channel noisy-OR still matches channel-sum, do not claim
delayed aggregation as a main contribution. Reframe as demand-lifted graph
retrieval / evidence tensor estimation.
```
```

### Phase 3: Full DAEC-DAPG Retrieval

Move beyond fixed pool:

```text
question
  -> demand graph
  -> query frontier over index graph
  -> absorption graph
  -> support tensor phi
  -> reader-budget projection
```

Main table:

```text
Dense
HippoRAG
PropRAG
DAEC-L1
DAEC-DAPG
```

Bridge-conditioned variants without public implementations should be discussed
briefly in related work, not used as mandatory baselines.

Datasets:

```text
2WikiMultihopQA
HotpotQA
MuSiQue
```

Recommended artifacts:

```text
reports/dpathrag/daec_dapg_phase3_full_retrieval_20260428.md
reports/dpathrag/daec_dapg_phase3_full_retrieval_20260428.json
reports/dpathrag/daec_dapg_phase3_rows_20260428.jsonl
```

Full-retrieval requirement:

```text
DAEC-DAPG must produce its candidate evidence from the index graph frontier and
absorption projection, not merely reorder a fixed dense/PropRAG top-100 pool.
```

Local top-100 experiments are allowed only as Phase 1 / Phase 2 operator
diagnostics.

## Executable Task Backlog

### P0-DIAG: Failure Taxonomy And Oracle Ceiling

Inputs:

```text
current full1000 baseline and DAEC-L1 predictions
gold support annotations
same frozen reader configuration
```

Outputs:

```text
Phase 0 diagnostics report
row-level taxonomy JSONL
oracle support-doc reader predictions
```

Gate:

```text
Proceed only if MuSiQue 3/4-hop has retrieval/composition headroom.
```

### P1-LOCAL: Local Absorption Operator

Inputs:

```text
top-100 pool documents
cached or newly extracted propositions/entities
DAEC-L1 demand/binding outputs
```

Outputs:

```text
local absorption phi variants
DAEC-L1 vs occupancy vs absorption vs absorption+kappa report
```

Gate:

```text
Support metrics improve over embedding-only phi on MuSiQue 3/4-hop.
```

### P2-MECH: Same-Substrate Mechanism Test

Inputs:

```text
one graph substrate
same frontier
same reader
same budget
```

Outputs:

```text
query-level PPR
mixed-source absorption
multi-channel + sum
multi-channel + noisy-OR
```

Gate:

```text
Delayed noisy-OR must beat early aggregation on support metrics.
```

### P15-POOL: Fixed-Pool Ceiling

Inputs:

```text
top-100 dense / PropRAG pool exports
gold support annotations in pool records
```

Outputs:

```text
support_recall@100 and support_complete@100 by dataset and hop bucket
```

Gate:

```text
If MuSiQue 4-hop support_complete@100 is low, treat Phase 1 complete=0 as a
pool-ceiling symptom and prioritize full graph frontier construction.
```

### P25-DENSE: Dense-Grounding Mechanism Test

Inputs:

```text
same local graph substrate as Phase 2
dense or TF-IDF fallback embeddings for demands, propositions, entities, docs
```

Outputs:

```text
query-level PPR
mixed-source absorption
multi-channel + sum
multi-channel + noisy-OR
```

Gate:

```text
Delayed noisy-OR must beat channel-sum under soft grounding.
```

### P3-FULL: Full Train-Free Retrieval

Inputs:

```text
offline index graph
demand graph
binding candidates
absorption operator
reader-budget projection
```

Outputs:

```text
full-corpus DAEC-DAPG retrieval and QA results
main comparison table against Dense / HippoRAG / PropRAG
```

Gate:

```text
Full method improves evidence assembly and does not rely on fixed-pool reranking.
```

## Go / No-Go

### Go

Proceed to full paper framing if:

```text
1. demand-lifted delayed aggregation beats query-level PPR on the same substrate
2. multi-channel + noisy-OR beats multi-channel + sum
3. absorbing hit beats or calibrates better than occupancy readout
4. MuSiQue 3/4-hop has retrieval/composition headroom, not only reader bottleneck
5. DAEC-DAPG improves support-complete@k and at least does not damage EM/F1
```

### No-Go

Do not promote DAEC-DAPG as main method if:

```text
1. gains come only from fixed-pool projection
2. demand-lifted propagation does not beat query-level graph retrieval
3. channel-sum performs the same as delayed noisy-OR
4. MuSiQue failures are mostly reader bottleneck
5. improvements require dataset-specific gates or graph boosts
```

In no-go case:

```text
DAEC-L1 remains a diagnostic / ablation,
not the main paper.
Downgrade the project to a limited fixed-pool composition / analysis paper,
or target a Findings-style contribution rather than a main-method claim.
```

## Reviewer Defense

### Is this just PropRAG?

No. PropRAG retrieves with query-level proposition paths and PPR, producing a
single document ranking:

```text
score(d)
```

DAEC-DAPG estimates:

```text
phi[i,b,d]
```

and delays aggregation until budgeted projection.

### Is this just HippoRAG?

No. HippoRAG uses query/fact/entity seeds and graph PPR to rank documents.
DAEC-DAPG estimates requirement-binding-specific absorption support.

### Is this just decomposition plus reranking?

No, if and only if the same-substrate tests show:

```text
demand-lifted delayed aggregation > query-level PPR
```

and:

```text
multi-channel noisy-OR > multi-channel sum
```

Otherwise the criticism is valid.

### Is noisy-OR the novelty?

No. Noisy-OR is the projection layer. The novelty is the graph-derived support
tensor and delayed aggregation.

### Is graph construction the novelty?

Generic proposition/entity graph construction is not claimed as new. The
contribution is the demand-conditioned Markov evidence graph used to estimate
`phi[i,b,d]`.

### Is it train-free?

Yes:

```text
frozen LLM decomposition
frozen embeddings
offline frozen extraction/indexing
train-free graph propagation
frozen reader
no parameter updates
no per-dataset tuning
```

## Remaining Design Questions For Review

Resolved decisions:

```text
1. Keep phi pure: phi[i,b,d] = kappa[i,b,d] * hit[i,b,d].
2. Use explicit prior-weighted best-binding as the main aggregation:
     J_b = p(b | q) * F_b(S_b).
3. Use P -> D as the main document sink edge.
4. Treat E -> D and binding-marginalized coverage as ablations.
5. Define p(b | q) by slot grounding confidence, top-Bmax truncation,
   and renormalization over retained bindings.
```

Still open:

1. Is finite horizon `L=2/3` enough, or should absorption use truncated
   personalized hitting probabilities until convergence?
2. Can local absorption smoke show a meaningful MuSiQue 3/4-hop signal before
   implementing full DAPG?
3. Does binding-marginalized coverage help evidence metrics, or does it mix
   mutually incompatible chains enough to hurt reader utility?

## Original One-Paragraph Paper Positioning

This was the intended positioning before the Phase 2.5 hard gates. It should not
be used as the current paper abstract.

Original wording:

We propose DAEC-DAPG, a training-free evidence assembly framework for multi-hop
RAG. Instead of collapsing a question into a single graph relevance signal,
DAEC-DAPG decomposes the query into atomic demands and latent bindings,
constructs demand-conditioned sources over a Markov evidence graph, and
estimates a document absorption tensor `phi[i,b,d]`. This tensor is then
compressed into a reader-facing context through a noisy-OR submodular projection,
preserving complementary evidence across demands under a fixed context budget.
The key mechanism is delayed aggregation: requirement-level support channels are
kept separate through graph propagation and aggregated only when selecting a
budgeted reader context.
