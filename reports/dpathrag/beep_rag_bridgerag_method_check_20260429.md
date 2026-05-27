# BEEP Day -1A: BridgeRAG Method Check

- Date: `2026-04-29`
- Decision: `PASS_WITH_STRONG_OVERLAP_RISK`
- Source checked: `https://arxiv.org/abs/2604.03384`
- Local PDF inspected: arXiv `2604.03384v2`, dated `2026-04-28`

## Finding

BridgeRAG is not just an abstract-level retrieval-only claim. The v2 method
contains a concrete bridge-conditioned retrieval pipeline:

```text
top-1 dense bridge b
LLM SVO hop-2 query generation
LLM extraction of e1/e2 from bridge b
dual-entity ANN expansion
tripartite LLM judge s(q,b,e1,e2,c)
PIT score fusion
retrieval-only R@5 evaluation
```

This means BEEP cannot claim broad bridge-conditioned retrieval novelty.

## BEEP Remaining Differentiation

BEEP remains distinct only if it can show all of the following:

```text
no LLM-generated SVO queries
no LLM tripartite judge
explicit document-entity-document edge scoring
multi-hop chain posterior rather than two-hop reranking
chain-position projection into fixed reader budget
Support-Complete@5 and reader EM/F1, not only R@5
```

## Decision

This gate does not kill BEEP by itself, because BridgeRAG v2 does not appear to
evaluate fixed reader-budget chain-position projection or final reader EM/F1.

However, it raises the reviewer-overlap risk to high. If BEEP passes the entity
bridge ceiling audit and later smoke tests, a BridgeRAG-lite control must be run
before any paper claim.

