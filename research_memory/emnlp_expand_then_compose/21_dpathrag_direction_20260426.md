# D-PathRAG Direction - 2026-04-26

## Final Direction

Stop expanding DAEC-vs-SetR as a main-track claim. SetR-windowed is a strong baseline and makes the "SetR cannot handle widened pools" framing unsafe.

Stop BSGS as a main route. Keep BSGS and QBF only as negative diagnostics.

New route:

> **D-PathRAG: fixed-pool, autoregressive, differentiable evidence-path selector.**

## Core Thesis

BSGS failed because node marginals cannot represent joint evidence coherence. D-PathRAG directly models an ordered evidence path:

```text
q_theta(d_1, ..., d_k | q, C_q)
```

The candidate pool remains fixed. The method is end-to-end selection over a fixed pool, not full-corpus end-to-end retrieval.

## Why This Is Different From Prior Work

Do not claim novelty from ST-Gumbel alone. Stochastic RAG and Gumbel Reranking are close prior work.

The intended novelty boundary is:

```text
autoregressive evidence path
per-step state-conditioned listwise re-encoding
DAEC binding features + warm start as multi-hop inductive bias
answer-loss-only E2E fine-tune with drift diagnostics
```

The decisive comparison is:

```text
D-PathRAG full vs unordered ST-Gumbel / Gumbel Reranking-style mask
```

If AR path does not beat unordered mask on support metrics, stop.

## Execution Rule

Do not implement the full four-week system immediately. First do a 10-day kill-or-continue pilot:

1. 2Wiki split/cache audit.
2. FiD/Flan reader baseline.
3. Selector warm start.
4. Small E2E pilot against unordered ST-Gumbel.

## Current Local Finding

Local 2Wiki file:

```text
reproduce/dataset/2wikimultihopqa.json
```

has 1000 examples and fields:

```text
_id
type
question
context
entity_ids
supporting_facts
evidences
answer
evidences_id
answer_id
```

This is sufficient for smoke/audit but likely not a full training split.

## Canonical Spec

See:

```text
docs/dpathrag_method_and_pilot_plan_20260426.md
```

## Implemented Baseline

Implemented and verified on 2026-04-26:

```text
scripts/dpathrag_audit_2wiki.py
scripts/dpathrag_build_cache.py
src/dpathrag/{data,cache,metrics,gumbel,selector,io}.py
tests/dpathrag/
```

Generated:

```text
reports/dpathrag/2wiki_split_audit.json
reports/dpathrag/2wiki_split_audit.md
data/dpathrag/cache/2wiki_dense_pool100_smoke.jsonl
data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl
```

Audit result:

```text
local 2Wiki file = 1000-example eval subset
train/dev/test not present locally
Dense pool aligned with 0 question mismatches
PropRAG pool aligned with 0 question mismatches
Dense support-complete@100 = 0.706
PropRAG support-complete@100 = 0.968
```

Cache safety:

```text
candidate.gold_support is a label for warm-start/eval only
candidate.features excludes gold_support to avoid training leakage
```

Next:

```text
Acquire/prepare official 2Wiki train/dev before real E2E selector training.
Use local 1000 only for smoke validation until then.
```
