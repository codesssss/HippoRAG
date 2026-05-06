# DAEC-LLM Review TODO - 2026-05-03

This is the active TODO list for hardening the current paper-facing
`DAEC-LLM + CTL` line after cross-pool full1000 results. It is intentionally a
task list, not a new method formulation.

Priority update, 2026-05-06:

- Canonical execution order is now recorded in
  `research_memory/emnlp_expand_then_compose/39_submission_critical_path_20260506.md`.
- Current blockers are:
  1. full1000 `IRCoT-style (local)`;
  2. full1000 `LLM-direct-select`;
  3. full1000-only CI and cost tables;
  4. current-version PropRAG full1000 `nobinding` refresh.
- Do not mix limit100 and full1000 in the main table. Limit100 results are
  appendix/pilot diagnostics.
- Do not start optional leakage, answer-masking, or new verifier experiments
  before the four blockers above are closed.

## Current Method Boundary

Paper-safe claim:

```text
DAEC is a fixed-pool evidence composer for title-indexed multi-hop QA. It
projects a retriever-produced candidate pool into a reader-ready top-k evidence
set using demand decomposition, LLM entity extraction, canonical title linking
(CTL), and noisy-OR coverage selection.
```

Do not claim:

- general-purpose arbitrary RAG reranking;
- preservation of PropRAG's original online LLM-free retrieval profile after
  attaching DAEC-LLM;
- CTL reliability solely because it is deterministic;
- a strong theoretical guarantee for noisy-OR beyond an interpretable coverage
  objective;
- pure evidence support modeling without answer-spotting controls.

## P0: Cost-Normalized Evaluation

Reviewer risk:

```text
DAEC is train-free but query-time LLM-heavy. When attached to PropRAG, it may
break PropRAG's online LLM-free efficiency story.
```

Tasks:

- [ ] Extract selector-only online cost for each method:
  - LLM calls/query;
  - prompt tokens/query;
  - completion tokens/query;
  - wall-clock selector latency/query.
- [ ] Extract end-to-end QA cost:
  - selector latency/query;
  - reader latency/query;
  - total latency/query.
- [ ] Build a cost/performance table with:
  - Top-5 baseline;
  - DAEC-HC;
  - DAEC-LLM raw substring;
  - DAEC-LLM + CTL;
  - SetR@20;
  - SetR@100 compressed;
  - optional windowed SetR if full1000 completes in time.
- [ ] Report selection-only and end-to-end cost separately.
- [ ] Use paper wording:

```text
DAEC-LLM is not compute-free. It is an optional evidence-composition layer that
trades additional online LLM cost for improved fixed-pool evidence quality.
```

Gate:

- [ ] Decide whether DAEC-LLM should be presented as an accuracy-oriented
  composer and DAEC-HC as the low-cost variant.

## P0: Answer-Spotting / Leakage Controls

Reviewer risk:

```text
The LLM extractor prompt asks for entities that could answer the question, so
DAEC may be exploiting answer spotting or the extractor LLM's latent answer
priors rather than selecting evidence.
```

Tasks:

- [ ] Run `demand-only extraction` control on limit=100:
  - give the extractor a decomposed demand, not the full question;
  - keep CTL and noisy-OR unchanged;
  - compare EM/F1/R@5 against DAEC-LLM + CTL.
- [ ] Run `doc-only entity extraction + demand matching` control on limit=100:
  - extract named entities from the passage without the question;
  - apply deterministic CTL and existing demand/title matching afterward;
  - measure performance drop.
- [ ] Run `answer masking` control if implementation cost is reasonable:
  - mask gold answer strings in selector/extractor inputs only;
  - do not mask reader input;
  - stratify by answer in title, answer in passage only, and answer absent.
- [ ] If controls are too expensive for full1000, run limit=100 first and only
  scale the strongest control.

Gate:

- [ ] If demand-only extraction preserves most gains, use it as the safer
  paper-facing extractor.
- [ ] If answer masking removes most gains, shrink the claim and explicitly
  state that DAEC-LLM benefits from answer-bearing entity identification.

## P1: CTL Reliability Audit

Reviewer risk:

```text
Deterministic linking is cleaner than raw substring, but deterministic does not
mean reliable. Parenthetical stripping and multi-token alias containment can
still create false positive links.
```

Tasks:

- [ ] Sample 100-200 CTL links from full1000 outputs across pools/datasets.
- [ ] Label each sampled link as correct / ambiguous / incorrect.
- [ ] Stratify by CTL match type:
  - exact normalized title;
  - parenthetical disambiguation removal;
  - multi-token contiguous alias containment.
- [ ] Record typical false positives:
  - disambiguation collapse;
  - alias collision;
  - title variant;
  - missing title in pool.
- [ ] Sample 100 empty extractor responses and label:
  - truly no linkable answer-bearing entity;
  - passage irrelevant;
  - missed linkable entity;
  - model/prompt failure.
- [ ] Produce an audit table:

```text
Match type | Precision | Recall proxy | Typical error
```

Gate:

- [ ] If multi-token alias precision is weak, demote it to ablation or restrict
  its use as a tie-break instead of a primary link.
- [ ] Keep the claim boundary: CTL assumes a title-indexed multi-hop QA
  substrate and is not a universal linker for arbitrary untitled chunks.

## P1: Significance and Gain/Regression Diagnostics

Reviewer risk:

```text
MuSiQue gains are modest and gain/regression counts can be close to tied.
```

Tasks:

- [ ] Run paired bootstrap significance tests for DAEC-LLM + CTL vs same-pool
  top-5 baseline on all 9 pool x dataset settings.
- [ ] Report MuSiQue cautiously:
  - modest gains;
  - higher hop count and sparser bridge structure;
  - not the strongest evidence for the method.
- [ ] Compute gain / regression / same counts for EM and F1 deltas.
- [ ] Break down gains by support-recall change:
  - support added;
  - duplicate removed;
  - distractor replaced;
  - regression from anchor removal.

Gate:

- [ ] Do not overclaim MuSiQue without significance support.

## P2: Path Consistency / Anchor Guard Sanity

Reviewer risk:

```text
Demand coverage alone may select individually relevant documents that do not
form a coherent multi-hop reasoning path.
```

Do not implement a weighted four-term objective as the main method. Avoid:

```text
lambda_1 DemandCoverage + lambda_2 BridgeConnectivity - lambda_3 Redundancy
+ lambda_4 AnchorPreservation
```

because it reintroduces a weighted heuristic stack.

Allowed low-complexity probes:

- [ ] Connectivity as deterministic tie-break:
  - primary objective remains noisy-OR demand coverage;
  - if coverage gains tie or are near-tied, prefer a document whose title is
    linked from an already selected document, or that links to an already
    selected title.
- [ ] Post-hoc connectivity audit:
  - measure whether selected evidence sets form title-link connected components;
  - compare EM/F1 for connected vs disconnected selections.
- [ ] Anchor preservation guard:
  - mark top-1/top-2 retriever documents as anchors;
  - protect anchors if they support at least one demand under existing DAEC
    support signals;
  - only replace duplicate or unsupported documents.

Gate:

- [ ] Run limit=100 before any full1000 path-consistency change.
- [ ] Promote only if gain/regression improves without adding tuned weights.

## P2: Paper Wording Updates

Tasks:

- [ ] Replace broad "generalizes across RAG" language with:

```text
generalizes across retrieval substrates under a shared title-indexed multi-hop
QA candidate-pool substrate
```

- [ ] Weaken noisy-OR theory language:
  - call it an interpretable coverage objective;
  - mention submodular intuition only if assumptions are explicitly stated;
  - do not oversell approximation guarantees.
- [ ] Add limitations:
  - query-time LLM cost;
  - title-indexed substrate assumption;
  - LLM extraction failures and empty responses;
  - modest MuSiQue gains;
  - possible answer-bearing entity bias.

## Immediate Run Order

1. Wait for SetR full1000 results to finish enough for cost/performance
   comparison.
2. Build cost-normalized table.
3. Run demand-only extraction control at limit=100.
4. Run CTL false-positive and empty-response audit.
5. Only then test connectivity tie-break / anchor guard at limit=100.
