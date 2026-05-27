# Proposal D: BEEP-RAG as Bridge-Edge Evidence Path Retrieval

Date: 2026-04-28

Status: stopped after Day -1 gates on 2026-04-29. This is retained as an
exploratory record, not an active paper plan.

**2026-04-29 decision update:** do not implement BEEP smoke or full BEEP. The
BridgeRAG method check found strong overlap risk, and the MuSiQue 4-hop manual
hop-type audit failed the entity-bridge ceiling gate. The current source of
truth remains:

- `docs/daec_l1_findings_decision_20260428.md`

Current source-of-truth relationship:

- `docs/daec_l1_findings_decision_20260428.md` remains the active execution
  decision.
- `docs/arec_rag_proposal_c_train_free_evidence_closure_20260428.md` is a
  red-light exploratory record after generated obligations degraded SC@5.
- This document is Proposal D: a corpus-grounded, train-free bridge-edge
  retrieval hypothesis that did not pass Day -1 pre-smoke gates.

## 2026-04-29 Day -1 Outcome

Day -1A BridgeRAG method check:

- Report: `reports/dpathrag/beep_rag_bridgerag_method_check_20260429.md`
- Decision: `PASS_WITH_STRONG_OVERLAP_RISK`
- Finding: BridgeRAG v2 already includes top-1 dense bridge selection,
  LLM-generated SVO hop-2 queries, dual-entity ANN, LLM entity extraction,
  tripartite LLM judging over `(q,b,e1,e2,c)`, and PIT fusion.
- Implication: BEEP cannot claim broad bridge-conditioned retrieval novelty.

Day -1B MuSiQue 4-hop manual hop-type audit:

- Report: `reports/dpathrag/beep_rag_musique4hop_manual_hop_audit_20260429.md`
- Decision: `STOP_ENTITY_BRIDGE_CEILING_FAIL`
- Sample: first 10 MuSiQue 4-hop examples inside limit=100.

| Metric | Value |
|---|---:|
| Queries audited | 10 |
| Estimated hop transitions | 30 |
| Entity-bridge transitions | 14 |
| Entity-bridge transition rate | 0.4667 |
| Fully entity-bridge queries | 1 |
| Fully entity-bridge query rate | 0.1000 |

The audit fails both pre-registered thresholds:

```text
entity_bridge transition rate >= 0.60
fully entity-bridge queries >= 6/10
```

Decision:

```text
Stop BEEP as an entity-only retrieval mainline.
Do not implement oracle bridge smoke.
Do not implement non-oracle bridge quality smoke.
Do not implement bridge mention retrieval smoke.
Do not implement BridgeRAG-lite.
Return to DAEC-L1 Findings + negative study framing.
```

## Executive Position

BEEP-RAG is a training-free evidence-chain assembly method for multi-hop RAG.
It does not ask an LLM to generate demands, obligations, or pseudo-queries. It
also does not run PPR / absorption over graph nodes. Instead, it enumerates and
scores explicit document-entity bridge edges:

```text
d1 --e1--> d2 --e2--> d3 ...
```

The core shift is:

```text
node relevance
  -> bridge-edge likelihood
  -> evidence-chain posterior
  -> reader-budget chain-position projection
```

One-sentence thesis:

```text
BEEP-RAG retrieves not the individually most relevant documents, but the most
probable corpus-grounded evidence chains and then projects their hop positions
into a fixed reader context budget.
```

This line exists because the previous train-free retrieval mechanisms failed in
specific ways:

- DAEC-DAPG failed because demand splitting, multi-channel propagation, local
  absorption, and delayed noisy-OR did not beat strong dense controls.
- AREC-RAG failed because generated obligations leaked candidate answers,
  encouraged self-justifying verifier scores, and pushed gold support out of the
  projected context.
- DAEC-L1 remains positive only as a fixed-pool context-selection method.

BEEP is the remaining plausible retrieval-side hypothesis because its retrieval
target is grounded in corpus text:

```text
original question + bridge mention sentence
```

not in LLM-generated proof obligations or demand text.

## Non-Negotiable Framing

Do not frame BEEP as:

```text
the first bridge-conditioned retriever
the first path retrieval method
another GraphRAG diffusion method
a reader-feedback method
an answer-conditioned retrieval method
a generated sub-question method
```

Frame it as:

```text
explicit document-entity bridge-edge likelihood
normalized evidence-chain scoring over enumerated beams
latent chain-position projection into reader-budget context sets
training-free corpus-grounded multi-hop evidence assembly
```

The paper should not fight on "bridge-conditioned retrieval" as a broad novelty
claim. That space already has recent preprints. The defensible claim is narrower:

```text
Bridge-conditioned retrieval should be evaluated as evidence-chain-to-context
assembly, not only as later-hop passage ranking.
```

## Relation To Prior Work

This section should be written carefully. The strongest pressure is BridgeRAG.
It should not be treated as a peer-reviewed blocker, but it is also not safe to
dismiss it from the abstract alone. The v2 paper posted on 2026-04-28 already
contains a concrete bridge-conditioned pipeline: top-1 dense bridge selection,
LLM-generated SVO hop-2 queries, dual-entity ANN expansion, LLM entity
extraction, tripartite LLM judging over `(q,b,e1,e2,c)`, and PIT score fusion.

| Method line | Status / overlap | Required BEEP distinction |
| --- | --- | --- |
| BridgeRAG | Recent arXiv preprint, v2 dated 2026-04-28; no venue / journal reference shown on arXiv as of 2026-04-29. It reports retrieval-only R@5 under matched benchmark evaluation. Its method is not just a metric claim: it uses top-1 dense bridge selection, LLM SVO generation, dual-entity ANN, LLM entity extraction, tripartite LLM judging, and PIT fusion. | Treat as related / concurrent pressure. BEEP must not claim bridge-conditioned retrieval as the novelty. The distinction must be: no LLM-generated SVO queries or LLM judge, explicit document-entity bridge edges, multi-hop chain posterior beyond a two-hop tripartite reranker, reader-budget chain-position projection, and end-to-end/context metrics rather than only R@5. |
| PathRAG | Retrieves key relational paths from a graph and converts paths into text for prompting | BEEP does not do flow-based graph pruning or path prompting; it scores document-entity bridge edges and projects document chains into the reader context. |
| HopRAG / graph expansion RAG | Passage-graph or logic-aware neighbor expansion | BEEP avoids node-mass expansion; it keeps explicit paths and scores each edge locally. |
| SetR / set selection RAG | ACL 2025 set-wise passage selection; shows RAG needs context sets, not only independent ranking | BEEP's final step is also set selection, but the signal comes from latent chain positions rather than CoT-derived information requirements. |
| DAEC-L1 | Fixed-pool demand-aware projection | BEEP must show full bridge expansion adds missing support beyond fixed-pool projection; otherwise it collapses back into selector territory. |
| AREC-RAG | Answer-conditioned proof obligations and verifier closure | BEEP avoids generated obligations and answer leakage by using corpus mention sentences as retrieval conditions. |

BridgeRAG should be cited, but not over-weighted. A precise defensive sentence:

```text
BridgeRAG is a recent arXiv preprint on bridge-conditioned retrieval. It
combines LLM-generated SVO expansion, dual-entity ANN, and an LLM tripartite
judge, and reports retrieval-only R@5. BEEP asks a different question: whether
explicit corpus entity edges can form multi-hop evidence-chain posteriors whose
chain positions improve fixed-budget reader contexts, measured by
Support-Complete@k and final reader EM/F1.
```

If a later BridgeRAG version adds reader-budget chain composition or final reader
EM/F1, BEEP should not proceed as a main method unless the entity-edge chain
posterior gives a clear empirical advantage in a matched setting.

References to verify during writing:

- BridgeRAG: `https://arxiv.org/abs/2604.03384`
- PathRAG: `https://arxiv.org/abs/2502.14902`
- SetR: `https://aclanthology.org/2025.acl-long.861/`
- HopRAG: `https://arxiv.org/abs/2502.12442`
- S-Path-RAG: `https://arxiv.org/abs/2603.23512`

## Core Hypothesis

Multi-hop QA often fails because the selected documents are individually
relevant but do not form a connected evidence chain. The missing object is not a
better scalar document score:

```text
score(d | q)
```

but a structured chain:

```text
gamma = (d1, e1, d2, e2, ..., e_{h-1}, dh)
```

where:

- `d_t` is a document / passage;
- `e_t` is a bridge entity or anchor that appears in both adjacent documents;
- `c(d_t, e_t)` is the bridge mention sentence or local mention window in
  document `d_t`.

BEEP's hypothesis:

```text
Bridge mention sentences from retrieved corpus documents provide cleaner
next-hop retrieval conditions than LLM-generated demands, obligations, or
sub-questions.
```

This should be tested before any full method investment.

## Offline Substrate

Build a lightweight document-entity index:

```text
G0 = (D, E, M)
```

where:

- `D` = documents / passages;
- `E` = canonical entities or mention anchors;
- `M(d,e)` = entity `e` appears in document `d`;
- `I(e) = {d : e in d}` = inverted list for entity `e`;
- `c(d,e)` = mention sentence or local window for entity `e` in document `d`.

Do not build:

- proposition graph;
- community summaries;
- PPR transition graph;
- LLM-generated pseudo-query graph;
- reader-side graph context.

Entity information weight:

```text
idf(e) = log(|D| / (1 + |I(e)|))
```

The physical meaning is simple: frequent entities are weak bridges; rare
entities are more specific bridges. In implementation, `idf(e)` is a feature in
a log-linear score, not a magical boost.

## Query-Time Chain Scoring

### Initial Document Distribution

Use the frozen dense retriever to get initial candidates:

```text
C0 = Ret(q, top-N)
```

Convert dense scores into an initial distribution over `C0`:

```text
p0(d | q) = softmax_{d in C0}(tau_0 * sim(q,d))
```

This is only the chain start distribution. It is not the final ranking.

### Document-To-Bridge Entity Likelihood

For a partial chain ending at document `d`, enumerate entity anchors in `d`.
Score each bridge entity with:

```text
s_e(d,q) = tau_c * sim(q, c(d,e)) + lambda_idf * log(idf(e) + eps)
```

and normalize locally:

```text
p(e | d,q) = softmax_{e in E(d)} s_e(d,q)
```

Default smoke values:

```text
tau_c = 1.0
lambda_idf = 1.0
eps = 1e-6
```

This is still train-free. The parameters are fixed global scaling constants, not
dataset-specific learned weights.

### Bridge-To-Next Document Likelihood

Given bridge entity `e`, the next candidate documents come from the inverted
list `I(e)`.

Use the original question plus the corpus bridge mention sentence as the
retrieval condition:

```text
Q_bridge = [q; c(d,e)]
```

Score next documents:

```text
s_d(d' | e,d,q) = tau_n * sim(Q_bridge, d')
```

and normalize within a truncated inverted-list frontier:

```text
p(d' | e,d,q) = softmax_{d' in TopL(I(e), Q_bridge)} s_d(d' | e,d,q)
```

The top-`L` truncation is an engineering frontier, not the conceptual mechanism.
It prevents high-frequency entities from exploding runtime.

### Evidence-Chain Score

A chain of length `h`:

```text
gamma = (d1, e1, d2, e2, ..., e_{h-1}, dh)
```

has log score:

```text
log S(gamma | q)
  = log p0(d1 | q)
    + sum_{t=1}^{h-1} [
        log p(e_t | d_t,q)
        + log p(d_{t+1} | e_t,d_t,q)
      ]
```

For smoke, test both:

```text
raw_product_score = log S(gamma | q)
length_normalized_score = log S(gamma | q) / max(1, h-1)
```

Do not overclaim calibrated probability unless scores are explicitly normalized
over the retained beam:

```text
P_B(gamma | q) = exp(score(gamma)) / sum_{gamma' in Gamma_B(q)} exp(score(gamma'))
```

Use `P_B` in projection and diagnostics. Call it an enumerated-beam posterior,
not a full-corpus posterior.

## Chain Beam Search

Algorithm:

```text
Input:
  question q
  dense retriever Ret
  document-entity index G0=(D,E,M)
  initial candidate size N
  entity branch size R
  inverted-list next-doc size L
  chain beam size B
  chain length H

1. C0 = Ret(q, top-N)

2. Initialize beams:
   for each d1 in C0:
       chain = [d1]
       score = log p0(d1 | q)

3. For t = 1 ... H-1:
   for each partial chain ending at d:
       enumerate top-R bridge entities e by p(e | d,q)
       for each e:
           retrieve top-L next documents d' from I(e)
           extend chain with (e, d')
           score += log p(e | d,q) + log p(d' | e,d,q)
   keep top-B chains

4. Normalize scores over final retained beam Gamma_B(q).

Output:
  Gamma_B(q)
```

Default smoke grid:

```text
N in {20, 100}
R in {5, 10}
L in {20, 50}
B in {50, 100}
H in {2, 3, 4, 5}
```

Do not tune this grid against final answer metrics. Use it to diagnose whether
the primitive has an oracle upper bound.

## Chain-Position Projection

The reader budget is still `k=5`. BEEP should not simply take all documents from
the top chain; that can duplicate titles and under-cover alternative hops.

Given retained chains:

```text
Gamma_B(q) = {gamma_1, ..., gamma_B}
```

Define the marginal probability that document `d` occupies chain position
`ell`:

```text
u_ell(d) = sum_{gamma in Gamma_B(q)} P_B(gamma | q) * 1[d = d_ell(gamma)]
```

Define position coverage:

```text
C_ell(S) = 1 - product_{d in S} (1 - u_ell(d))
```

Select a reader context:

```text
F(S) = sum_{ell=1}^{H} C_ell(S)
S* = argmax_{|S| <= k} F(S)
```

Greedy projection is justified because each `C_ell` is a monotone submodular
coverage function when `0 <= u_ell(d) <= 1`, and non-negative sums preserve
submodularity. Under a cardinality budget, greedy has the standard `(1 - 1/e)`
approximation guarantee.

Important caveat:

```text
The theorem supports the projection step. It is not the main novelty.
```

The main novelty must be empirical: explicit bridge-edge chain scoring improves
missing support recovery and reader-budget evidence completeness.

## Why BEEP Is Different From Failed AREC

AREC failed because the retrieval target was generated from the model's own
candidate answer:

```text
candidate answer -> generated obligation -> verifier support
```

The observed failure path was:

```text
generated obligation
  -> answer leakage / self-justification
  -> verifier self-consistency
  -> gold support is projected out
```

BEEP avoids this loop:

```text
retrieved document text
  -> real entity mention sentence
  -> next-hop corpus retrieval
```

No generated answer, obligation, or sub-question is used as a retrieval target.

## Why BEEP Is Different From Failed DAPG

DAPG represented multi-hop support as demand-channel document scores:

```text
phi[i,b,d]
```

and then relied on local absorption / delayed aggregation. The experiments
showed that:

- raw dense query cosine beats absorption;
- demand union does not beat the original question;
- multi-channel delayed aggregation is weaker than single-channel controls.

BEEP does not split the question into generated demands and does not diffuse
node mass. It keeps explicit edge decisions:

```text
document -> bridge entity -> next document
```

and preserves the path until reader-budget projection.

## Main Claims If Smoke Passes

Use three claims only if the smoke gates pass:

1. **Bridge-edge chain posterior.**
   BEEP scores explicit document-entity bridge edges and composes them into an
   enumerated-beam evidence-chain posterior, avoiding graph diffusion and
   generated retrieval targets.

2. **Chain-position context projection.**
   BEEP projects latent chain positions into a fixed reader context budget using
   a monotone submodular coverage objective, selecting complementary documents
   rather than individually high-ranking documents.

3. **End-to-end evidence-chain evaluation.**
   BEEP evaluates bridge retrieval by Support-Complete@k, missing-support hit
   rate, and final reader EM/F1, not only retrieval R@5.

Do not claim:

- BEEP is the first bridge-conditioned retrieval method;
- BEEP is the first path retrieval method;
- BEEP beats BridgeRAG unless a matched baseline is implemented;
- BEEP is a GraphRAG diffusion method;
- BEEP solves multi-hop QA before reader EM/F1 is measured.

## Day -1 Pre-Smoke Gates

Before writing BEEP code, run two cheap gates. These are harder blockers than
the smoke scripts.

### Gate A: BridgeRAG Method Check

Read the latest BridgeRAG paper, not only the abstract. The current v2 method
check is:

```text
BridgeRAG v2 uses:
  top-1 dense bridge b
  LLM SVO hop-2 query generation
  LLM extraction of e1/e2 from b
  dual-entity ANN expansion
  tripartite LLM judge s(q,b,e1,e2,c)
  PIT fusion
  retrieval-only R@5 evaluation
```

Current interpretation:

```text
BridgeRAG overlaps strongly with broad bridge-conditioned retrieval.
BEEP still differs if it avoids LLM-generated SVO / LLM judging and instead
scores explicit document-entity-document chains with reader-budget projection.
```

Hard stop:

```text
If a newer BridgeRAG version contains explicit multi-hop chain enumeration,
chain-position projection, or final reader EM/F1 evaluation that matches BEEP's
intended claim, stop BEEP as a main method and return to DAEC-L1.
```

### Gate B: Manual Hop-Type Audit

Before building the entity index pipeline, manually inspect 10 MuSiQue 4-hop
queries. For each hop, label whether the transition is recoverable by an
entity-bridge edge:

```text
entity_bridge        # next support is identifiable from a co-occurring entity
relation_selection   # multiple entities occur; relation type is required
attribute_or_event   # date, property, event, or phrase anchor is required
comparison_operator  # largest, later, higher, etc.
not_entity_bridge
```

Decision:

```text
If fewer than 60% of hop transitions are entity_bridge, or fewer than 6/10
queries have all required transitions representable by entity bridges, skip
BEEP implementation. The entity-only oracle ceiling is likely too low.
```

Recommended artifacts:

```text
reports/dpathrag/beep_rag_musique4hop_manual_hop_audit_20260429.md
reports/dpathrag/beep_rag_musique4hop_manual_hop_audit_20260429.json
```

Do not "rescue" a failed entity-only audit with phrase anchors in the first
implementation. Phrase anchors reopen the AREC-style risk of inventing
retrieval targets unless they are strictly corpus-extracted and separately
validated.

## Minimum Smoke Plan

Do not implement full BEEP before these smokes.

### Smoke 1: Oracle Bridge Upper Bound

Dataset:

```text
MuSiQue limit=100, focusing on 3-hop and 4-hop buckets.
```

Construct oracle bridge entities from gold support documents:

```text
gold support docs -> shared / linking entities -> oracle bridge frontier
```

Compare:

```text
initial dense top5
oracle bridge fixed-pool top20 projection
oracle bridge fixed-pool top100 projection
oracle entity-occurrence filter over dense top100 -> top5
oracle bridge full expansion + chain projection
```

Metrics:

```text
Support Recall@5
Support-Complete@5
unique-title-normalized SC@5
passage-level SC@5 if doc ids are available
```

Go condition:

```text
oracle bridge full expansion improves SC@5 over initial dense top5 by >= +0.20
and improves SC@5 over oracle fixed-pool top20 projection by >= +0.08.
It must also beat oracle entity-occurrence filtering over dense top100 by a
meaningful margin; otherwise the method is only an oracle selector.
```

No-go condition:

```text
oracle bridge fixed-pool projection explains most of the gain,
or oracle full expansion does not beat initial dense top5 by a meaningful margin.
```

This is the same lesson learned from AREC. Never interpret full expansion gains
without a fixed-pool control.

Recommended artifacts:

```text
reports/dpathrag/beep_rag_oracle_bridge_musique_limit100_20260428.md
reports/dpathrag/beep_rag_oracle_bridge_musique_limit100_20260428.json
reports/dpathrag/beep_rag_oracle_bridge_musique_limit100_rows_20260428.jsonl
```

### Smoke 2: Non-Oracle Bridge Entity Quality

From dense top20 documents only, compute `p(e | d,q)` and inspect whether top
bridge entities overlap with gold-chain entities.

Metrics:

```text
BridgeHit@5
BridgeHit@10
entity-idf percentile
generic-entity rate
question-entity loop rate
sim(q, e_name) baseline rank
sim(q, c(d,e)) minus sim(q, e_name)
```

Baselines:

```text
entity frequency baseline
idf-only baseline
entity-name cosine baseline: sim(q, e_name)
question-sentence cosine-only baseline
random entity from top20 docs
```

Go condition:

```text
BridgeHit@10 on 3/4-hop is clearly above frequency and random baselines,
and top bridges are not dominated by generic entities such as country, film,
person, United States, or title-only repeats.
The mention-sentence term must add signal beyond sim(q, e_name); if the gap is
<5% relative in BridgeHit@10, do not claim bridge mention sentences as a core
mechanism.
```

Recommended artifacts:

```text
reports/dpathrag/beep_rag_bridge_entity_quality_musique_limit100_20260428.md
reports/dpathrag/beep_rag_bridge_entity_quality_musique_limit100_20260428.json
reports/dpathrag/beep_rag_bridge_entity_quality_musique_limit100_rows_20260428.jsonl
```

### Smoke 3: Bridge Mention Retrieval Hit Rate

Compare second-hop candidate retrieval:

```text
raw-question second retrieval
entity-name retrieval: [q; e]
bridge mention sentence retrieval: [q; c(d,e)]
oracle bridge mention retrieval
```

Metric:

```text
MissingHit@M = new candidates contain gold support missing from initial top5
```

Go condition:

```text
[q; c(d,e)] missing-support hit rate is meaningfully above raw-question second
retrieval under the same candidate budget.
[q; c(d,e)] must also beat [q; e] by a visible margin. If [q; e] explains most
of the gain, mention-window indexing is not a defensible mechanism.
```

Recommended artifacts:

```text
reports/dpathrag/beep_rag_bridge_retrieval_musique_limit100_20260428.md
reports/dpathrag/beep_rag_bridge_retrieval_musique_limit100_20260428.json
reports/dpathrag/beep_rag_bridge_retrieval_musique_limit100_rows_20260428.jsonl
```

## Formal Go / No-Go

### Go

Proceed to implementation beyond smoke only if all are true:

```text
1. Oracle bridge full expansion beats initial dense top5 SC@5 by >= +0.20.
2. Oracle bridge full expansion beats oracle fixed-pool top20 SC@5 by >= +0.08.
3. Oracle bridge full expansion beats oracle entity-occurrence top100 filtering.
4. Non-oracle BridgeHit@10 beats frequency / idf / random / sim(q,e) baselines.
5. Bridge mention retrieval beats raw-question second retrieval in MissingHit@M.
6. Bridge mention retrieval visibly beats entity-name retrieval [q;e].
7. Top bridge entities are interpretable and not dominated by generic hubs.
8. Chain-position marginal entropy is not near-zero; otherwise projection is
   just top-1-chain selection with mathematical decoration.
```

### Conditional Go

Proceed only as a selector / DAEC-L1 ablation if:

```text
1. Oracle bridge fixed-pool projection improves SC@5,
2. but full bridge expansion adds little or nothing beyond fixed-pool projection.
```

In this case, BEEP is not a retrieval paper. It is a chain-aware context
selection signal.

### No-Go

Stop BEEP if:

```text
1. Oracle entity bridges do not recover missing support.
2. Entity-only bridge upper bound is low on MuSiQue 3/4-hop.
3. Oracle entity-occurrence filtering explains most oracle full-expansion gain.
4. Bridge mention retrieval does not beat raw-question second retrieval.
5. Bridge mention retrieval does not beat [q;e].
6. Chain-position projection mainly selects duplicate-title passages.
7. Position marginals collapse to one chain in most examples.
8. Gains disappear under unique-title or passage-level support metrics.
```

If BEEP hits no-go, stop opening new retrieval mainlines and return to:

```text
DAEC-L1 as fixed-pool context selection
+ negative GraphRAG / demand / reader-feedback study.
```

## Full Experiment Plan If Smoke Passes

Datasets:

```text
MuSiQue
2WikiMultihopQA
HotpotQA
```

Primary metrics:

```text
Answer EM/F1
Support Recall@5
Support-Complete@5
Hop-bucket F1 / SC@5
MissingHit@M
unique-title-normalized SC@5
latency / query
index size
```

Main baselines:

```text
Dense top-k
Dense top100 -> top5 by dense score
Raw-question second retrieval
Entity expansion only
Bridge mention sentence retrieval
Node-mass PPR / absorption from existing DAEC-DAPG diagnostics
DAEC-L1 fixed-pool projection
Chain posterior without position projection
BEEP full: chain posterior + chain-position projection
```

Reviewer-defense baseline:

```text
BridgeRAG-lite scorer:
  score(q, bridge_document_or_sentence, candidate_document)
```

Do a lightweight version earlier than the full experiment if the BridgeRAG
method check shows overlap in the exact tested setting. It does not need to
match every BridgeRAG detail initially; the goal is to know whether BEEP's
entity-edge posterior is doing more than a bridge-conditioned reranker.

Minimum BridgeRAG-lite control:

```text
same initial dense bridge document
same candidate frontier as BEEP
score candidates with [q; bridge_doc_or_sentence; candidate_doc]
project / select top5 under the same reader budget
```

Decision:

```text
If BridgeRAG-lite is within ~0.02 SC@5 / Recall@5 of BEEP in the smoke setting,
BEEP should be downgraded to Findings/internal-tool risk unless it has a clear
cost or interpretability advantage.
```

## Diagnostics

Report these diagnostics for every BEEP smoke / full run:

### Bridge Entity Diagnostics

```text
top bridge entities per query
idf distribution
generic entity rate
question-entity loop rate
gold-chain entity hit
```

### Chain Diagnostics

```text
top chain text view: d1 title -> e1 -> d2 title -> ...
chain length distribution
duplicate title count
unique title count
position marginal entropy
```

### Retrieval Diagnostics

```text
new gold outside initial top5
new gold outside initial top20
MissingHit@M vs raw-question second retrieval
bridge mention retrieval examples
```

### Projection Diagnostics

```text
selected chain positions covered
selected support titles
support lost relative to initial top5
fixed-pool vs full-expansion delta
```

The fixed-pool delta is mandatory. AREC failed because the apparent full-method
gain was mostly explained by fixed-pool projection.

## Main Risks

### Risk 1: BridgeRAG Similarity

BEEP is close enough to BridgeRAG that reviewer pressure is guaranteed.

Defense:

- BridgeRAG is currently a recent arXiv preprint, not a peer-reviewed published
  blocker.
- Its v2 method already includes concrete bridge-conditioned reranking with LLM
  SVO generation, dual-entity ANN, tripartite LLM judging, and PIT fusion.
- Its evaluation is still retrieval-only R@5, not fixed reader-budget
  Support-Complete@k or final reader EM/F1.
- BEEP should evaluate reader-budget context composition and final EM/F1.
- Add BridgeRAG-lite as soon as BEEP passes the Day -1 gates and the tested
  setting overlaps BridgeRAG's two-hop bridge-conditioned reranking. It can wait
  only if the oracle entity-bridge audit is already red.

Do not attack authorship or writing style in the paper. The scientific critique
is sufficient:

```text
not peer reviewed yet; retrieval-only metric; no final reader EM/F1; no
explicit chain-position reader-budget projection in v2.
```

### Risk 2: Entity Extraction / Canonicalization

Alias errors can break chains.

Mitigation:

- Start with the existing entity index if available.
- Report entity extraction coverage on gold support docs.
- Keep oracle bridge smoke separate from non-oracle bridge selection.
- If entity-only oracle upper bound is low, consider phrase anchors only as a
  separate follow-up, not in the first implementation.

### Risk 3: MuSiQue Hops Are Not Always Entity Bridges

Some hops may be events, dates, attributes, or descriptions. Entity-only bridges
may be too narrow.

Decision:

```text
If entity-only oracle bridge upper bound is low, stop BEEP rather than patching
with generated phrase obligations.
```

Phrase anchors are allowed only if extracted from corpus text, not generated by
the reader.

### Risk 4: Duplicate Titles Inflate Support Metrics

MuSiQue can contain multiple support passages with the same title. A path method
can look better at title-level than at passage-level.

Mitigation:

- Report title-level and passage-level metrics when IDs are available.
- Report unique-title-normalized SC@5.
- Penalize duplicate-title selection in diagnostics before turning it into a
  formal objective.

### Risk 5: It Still Becomes A Selector

If fixed-pool top20 projection explains most of the gain, BEEP is not a new
retrieval method.

Decision:

```text
Fold it into DAEC-L1 as a chain-aware selector ablation.
Do not pitch it as BEEP-RAG main method.
```

## Implementation Notes

Minimum modules if implemented:

```text
src/dpathrag/beep/
  entity_index.py       # document-entity mentions, inverted lists, mention windows
  bridge_scoring.py     # p(e|d,q), p(d'|e,d,q)
  chain_search.py       # beam enumeration and normalization
  projection.py         # chain-position coverage and greedy selection
  smoke.py              # shared row builders and metrics
```

Minimum scripts:

```text
scripts/beep_rag_oracle_bridge_smoke.py
scripts/beep_rag_bridge_entity_quality.py
scripts/beep_rag_bridge_retrieval_smoke.py
scripts/beep_rag_smoke_summary.py
```

Minimum tests:

```text
tests/dpathrag/test_beep_core.py
tests/dpathrag/test_beep_scripts.py
```

Do not run full1000 before smoke gates pass.

Immediate execution order:

```text
Day -1A: BridgeRAG method check against the current arXiv version.
Day -1B: manual MuSiQue 4-hop hop-type audit on 10 queries.
Stop if either Day -1 gate is red.
Day 0: oracle bridge upper bound with fixed-pool and entity-occurrence controls.
Day 1: non-oracle bridge entity quality with sim(q,e) baseline.
Day 2: bridge mention retrieval vs raw question and [q;e].
Only after these: consider full BEEP and BridgeRAG-lite.
```

## Reviewer-Facing Abstract Draft If It Works

```text
Multi-hop RAG is often evaluated as passage ranking, but reader performance
depends on whether the final context forms a complete evidence chain. We propose
BEEP-RAG, a training-free bridge-edge evidence path retrieval method. BEEP builds
a lightweight document-entity index, scores explicit document-entity-document
bridge edges using corpus mention sentences, composes these edges into an
enumerated evidence-chain posterior, and projects latent chain positions into a
fixed reader context budget with a submodular coverage objective. Unlike
generated-demand or reader-feedback methods, BEEP uses only corpus-grounded
bridge mentions as next-hop retrieval conditions. We evaluate not only retrieval
R@k but Support-Complete@k and final reader EM/F1, showing when explicit
bridge-edge chains recover missing multi-hop support beyond strong dense
retrievers and fixed-pool selectors.
```

## One-Sentence Current Decision

BEEP has enough paper shape to justify a half-day to one-day oracle bridge smoke,
but not enough evidence for full implementation. If oracle bridge expansion does
not beat both initial dense top5 and fixed-pool projection, stop new retrieval
mainlines and return to DAEC-L1 plus the negative GraphRAG study.
