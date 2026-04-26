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

## Borrowed Ideas To Keep

Must use:

```text
SetR: direct baseline; it owns the generic set-selection framing.
Stochastic RAG: unordered ST-Gumbel baseline.
Gumbel Reranking: document-wise top-k mask baseline.
G-Reasoner: negative-section framing for learned vs assumed graph edge semantics.
```

Recommended implementation details:

```text
Inject a query token h_q into every per-step ListTransformer pass.
Shuffle ambiguous gold support order per epoch during warm start.
Track Stage 2 selector drift with KL > 0.05 as meaningful movement.
Track warm-start top-1 overlap; healthy range is 60-90%.
Track selector gradient norm in the first 100 steps.
Use 2Wiki pilot -> HotpotQA sanity -> MuSiQue stress test.
```

Paper framing:

```text
Use "stateful evidence accumulation" as the intro story.
Do not use cognitive-memory metaphor as a technical claim.
Do not claim novelty from ST-Gumbel alone.
Do not add supporting-fact auxiliary loss in Stage 2.
Do not use hand-weighted DAEC residual prior as the main method.
```

Source anchors:

```text
SetR: https://aclanthology.org/2025.acl-long.861/
Stochastic RAG: https://arxiv.org/abs/2405.02816
Gumbel Reranking: https://aclanthology.org/2025.acl-long.354/
G-Reasoner: https://arxiv.org/abs/2509.24276
```

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

## Data Separation Rule

Keep legacy 1000-example reproduction protocol unchanged:

```text
reproduce/dataset/2wikimultihopqa.json
reproduce/dataset/2wikimultihopqa_corpus.json
```

Full 2Wiki train/dev/test for D-PathRAG lives under:

```text
data/dpathrag/full_2wiki/
```

Use HF mirror:

```text
HF_ENDPOINT=https://hf-mirror.com .venv-hipporag/bin/python scripts/dpathrag_prepare_2wiki_full.py
```

Prepared on 2026-04-26:

```text
train = 167454
validation = 12576
test = 12576
full corpus unique docs = 430225
full corpus unique titles = 398354
dedupe key = title+text
```

Do not assume title uniqueness in the full corpus.
