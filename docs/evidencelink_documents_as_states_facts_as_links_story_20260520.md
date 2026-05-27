# EvLink Story Memo: Documents as States, Facts as Links

Date: 2026-05-20

Status: internal story memo only. This file records the current framing for discussion with the advisor. It does not modify the paper text.

## Core Claim

EvLink is built around a deliberately simple view:

> Documents are retrieval states; facts are evidence links.

In retrieval-first multi-hop QA, the system ultimately needs source-grounded textual evidence. Therefore, the query-time retrieval state should stay aligned with that evidence interface: documents or passages. OpenIE facts remain essential, but their role is not to replace documents as the final retrieval state. Instead, facts serve as source-grounded transition certificates that explain why one document should lead to another.

This is an interface-aligned design, not a universal knowledge representation claim. We do not argue that document graphs are always better knowledge graphs. Fine-grained entity, fact, proposition, and summary graphs are valuable for representing corpus structure. Our narrower claim is that, when the task is retrieval-first QA over source passages, the retrieval state should remain close to the evidence unit, while extracted facts should justify transitions between those evidence units.

## Why Propose This View?

Recent GraphRAG systems often make the graph state increasingly fine-grained:

- entities or phrases;
- OpenIE facts or triples;
- propositions;
- hypergraph units;
- summary nodes;
- mined reasoning chains.

This direction is natural for knowledge representation, because finer units expose more internal structure. But retrieval-first QA has a different interface: the system must return source-grounded textual evidence. In common QA settings, this evidence is a passage, chunk, paragraph, or document snippet. Benchmarks make this interface explicit through passage-level support labels, and deployed RAG systems expose the same interface through retrieved contexts and citations.

The concern is therefore not that fine-grained graphs are wrong. The concern is that graph state and evidence state can become misaligned. A system may know which entity, fact, proposition, or summary is salient, but it still has to decide which source passages should be returned as evidence. That readout can affect whether bridge evidence is preserved.

EvLink takes the opposite design pressure seriously: keep the query-time state simple and evidence-aligned. Let documents remain the moving state. Use facts not as the final objects to retrieve, but as certificates for moving from one document to another.

## What Problem Does This Target?

The target problem is not simply weak graph propagation. It is evidence discovery under conditional relevance.

In multi-hop QA, the hardest evidence is often not directly similar to the original question. A bridge document can become necessary only after another document reveals an entity or relation.

Example:

```text
Question: Where was A's spouse born?
Document 1: A married B.
Document 2: B was born in Paris.
```

Document 2 may not be strongly similar to the original question if `B` is not mentioned in the query. Its relevance is conditional: it is licensed by the fact in Document 1. A retrieval system that ranks passages independently by query similarity can underweight Document 2. A graph system can expose the bridge entity, but it still needs a reliable way to turn that bridge into source evidence.

The problem can be stated as:

> Multi-hop retrieval must recover documents whose relevance is activated by facts in other documents, not only by direct question-document similarity.

This creates two recurring failure modes:

1. Bridge underweighting: necessary bridge or answer-side passages receive low direct relevance and fall below the returned evidence set.

2. Evidence crowding: the retrieved evidence concentrates around one entity neighborhood, while other entities or relations required by the question remain uncovered.

EvLink's document-state view targets both failures by making document-to-document evidence transitions explicit.

## How Does the View Solve the Problem?

EvLink changes the division of labor between documents and facts.

### 1. Documents Stay as States

The query-time graph moves over documents/passages. Each admitted node is already a candidate source evidence unit. This keeps retrieval decisions aligned with the final evidence interface.

This reduces readout ambiguity: the system does not first rank a collection of internal facts/entities and then decide which passage should represent them. It directly admits and ranks source passages as evidence states.

### 2. Facts Become Links

OpenIE facts and source spans certify transitions between documents.

Instead of using a fact only as a graph node to be ranked, EvLink uses it to justify an edge:

```text
Document A --[source-grounded fact / endpoint / relation evidence]--> Document B
```

The edge means:

> A source-grounded fact in one document makes another document a plausible next-hop evidence candidate.

This turns OpenIE from a purely representational layer into a retrieval mechanism for conditional bridge discovery.

### 3. Retrieval Becomes Evidence Movement

Dense and lexical retrieval supply reliable entry documents. Query anchors provide symbolic entry points. From these seeds, EvLink traverses a compact query-local document graph along fact-certified evidence links.

The traversal is deliberately bounded and local. The goal is not to search a huge global graph exhaustively. The goal is to admit documents that become necessary through source-grounded evidence transitions.

### 4. Evidence Assembly Uses Documents Directly

Once candidate evidence documents are admitted, coverage-aware composition can operate directly on source passages. It asks which documents cover unresolved question-side evidence needs, rather than first mapping fine-grained graph units back to passages.

This makes the full retrieval process:

```text
entry documents
  -> fact-certified document transitions
  -> query-local evidence candidates
  -> coverage-oriented evidence assembly
```

## Why This Is "Simple" Rather Than Weak

The simplicity is not that the system ignores facts. It is that it assigns facts a different role.

Common direction:

```text
documents -> entities/facts/propositions as graph states -> graph search -> source-text readout
```

EvLink direction:

```text
documents remain graph states -> facts certify links -> retrieval moves over source evidence
```

This is a simpler query-time state, not a simpler source signal. The system still uses OpenIE, endpoint normalization, source spans, and relation evidence. But these signals justify document transitions instead of becoming the final retrieval state.

The slogan is:

> Simple states, evidential links.

Or:

> Keep retrieval at the evidence unit; use facts to license movement between evidence units.

## Empirical Signals Supporting the Story

The strongest support comes from the bridge-heavy 2WikiMultiHopQA setting.

Against dense entry retrieval:

| Dataset | R@5 Gain | All@5 Gain | F1 Gain | Support Gain |
|---|---:|---:|---:|---:|
| HotpotQA | +3.1 | +6.2 | +2.6 | 7.2% |
| 2WikiMultiHopQA | +20.4 | +40.2 | +14.2 | 45.1% |
| MuSiQue | +5.3 | +8.1 | +2.7 | 19.9% |

The 2Wiki result is central to the story. It shows that EvLink is not merely reordering already obvious neighbors. For 45.1% of 2Wiki queries, the method retrieves strictly more gold supporting passages in the top evidence set than dense entry retrieval.

The component ablations also support the facts-as-links claim:

| Variant | 2Wiki R@5 | 2Wiki F1 | 2Wiki All@5 |
|---|---:|---:|---:|
| Full EvLink | 96.2 | 74.7 | 89.5 |
| w/o evidence-need mining | 92.8 | 71.1 | 81.1 |
| w/o Evidence-Coverage-Aware Retrieval Optimization | 93.5 | 72.8 | 82.6 |
| w/o evidence-linked transitions | 93.9 | 73.0 | 82.0 |

The final row is especially important: replacing evidence-linked transitions with phrase-source-only links keeps the same broader pipeline but removes the fact-certified transition substrate. The 2Wiki All@5 drop from 89.5 to 82.0 and F1 drop from 74.7 to 73.0 support the claim that facts-as-links are not cosmetic.

## Boundary of the Claim

This memo preserves the current framing, but the claim should remain scoped:

- We do not claim document graphs are universally better knowledge representations.
- We do not claim fine-grained graph states are useless.
- We do not claim source-grounded fact extraction is free or errorless.
- We do claim that, for retrieval-first multi-hop QA, document-level retrieval states with fact-certified transitions are a strong interface-aligned design.

The key tradeoff is:

```text
less ambiguity in graph-unit-to-passage readout
vs.
greater dependence on OpenIE edge quality
```

This tradeoff is acceptable for the current method because the experiments show large gains where conditionally relevant bridge evidence is most important, especially on 2WikiMultiHopQA.

## Advisor-Facing Short Answer

If asked "why propose this view, what problem does it target, and how does it solve it?", the compact answer is:

> We propose "documents as states, facts as links" because retrieval-first multi-hop QA ultimately needs source passages as evidence, while many GraphRAG designs make the query-time state more fine-grained than the returned evidence. The targeted problem is conditional bridge discovery: some passages are not directly similar to the question, but become necessary after another passage exposes an entity or relation. EvLink addresses this by keeping documents as the retrieval states and using OpenIE facts/source spans as transition certificates. Retrieval therefore moves directly over source evidence, while facts explain why one document should lead to another.

Chinese version:

> 我们提出 "documents as states, facts as links"，是因为 retrieval-first multi-hop QA 最终需要返回源文本证据，而很多 GraphRAG 的 query-time 状态比最终证据更细。我们针对的问题是条件桥接证据发现：有些文档并不直接像问题，而是在另一篇文档揭示某个实体或关系后才变得必要。EvLink 的解决方式是让文档保持为检索状态，用 OpenIE facts/source spans 作为文档跳转的证明。这样检索直接在源文本证据上移动，同时 facts 解释为什么一篇文档应该引出另一篇文档。
