# D-PathRAG Method and Pilot Plan

Date: 2026-04-26

## Decision

Stop BSGS as a main route and keep it only as a negative diagnostic.

New main research direction:

> **D-PathRAG: Differentiable Evidence Path Selection for Multi-Hop RAG**

D-PathRAG is **not** full-corpus end-to-end retrieval. It is end-to-end evidence selection over a fixed candidate pool.

## Motivation

BSGS failed because node marginals are the wrong state variable for multi-hop evidence composition. The oracle diagnostics showed that even after oracle pool exposure, oracle binding rewrites, and oracle likelihoods, BSGS still failed to recover coherent full support sets and produced high duplicate-title concentration.

The lesson is:

```text
node marginal belief ≠ joint evidence coherence
```

D-PathRAG replaces node marginals with an autoregressive evidence path latent variable:

```text
pi = (d_1, d_2, ..., d_k)
S(pi) = {d_1, ..., d_k}
```

The selector learns:

```text
q_theta(pi | q, C_q)
```

where `C_q` is a fixed candidate pool from dense retrieval / DAEC / PropRAG.

## Method Boundary

### Main Method

- Fixed candidate pool.
- Autoregressive no-replacement evidence-path selector.
- Per-step conditioned listwise re-encoding.
- ST-GumbelTop1 hard-forward / soft-backward selection.
- DAEC features and imitation warm start as inductive bias.
- FiD/Flan-style differentiable reader with document-level mask.
- Stage 2 objective is pure answer NLL.

### Explicit Non-Goals

- Do not continue full BSGS.
- Do not use node-marginal graph propagation as the main state.
- Do not train Qwen3/vLLM end-to-end.
- Do not add supporting-fact auxiliary loss to the final objective.
- Do not use manually weighted `DAEC_score + neural_score` as the main scoring rule.
- Do not call the method full end-to-end retrieval.

## Architecture

### Candidate Pool

For each question `q`, build:

```text
C_q = {d_1, ..., d_n}
```

Pilot defaults:

```text
dataset: 2WikiMultiHopQA
n: 20 for smoke, 40 for main pilot
k: 4 or 5 selected documents
candidate pool: dense top-n and/or existing DAEC/PropRAG pool top-n
```

### DAEC Feature Cache

Each candidate document gets a compact feature vector:

```text
rank / retriever score
DAEC per-demand coverage score(s)
binding / entity overlap features
DAEC selected indicator
gold support indicator if available
title metadata
```

DAEC features are observations and warm-start targets, not hand-weighted logit priors.

### Autoregressive Selector

At each step `t`:

```text
H^(t) = ListTransformer(h_1, ..., h_n ; r_{t-1}, selected_mask)
logits_t = f_theta(H^(t), r_{t-1}) + no_replacement_mask
z_t = ST-GumbelTop1(logits_t, tau)
r_t = GRU(r_{t-1}, sum_i z_{t,i} H_i^(t))
```

Main variant:

```text
D-PathRAG-full = per-step conditioned listwise re-encoding
```

Ablation:

```text
D-PathRAG-lite = one-shot listwise encoding + GRU state
```

### Differentiable Reader

Use a HuggingFace FiD/Flan-T5-style reader, not vLLM.

Document-level mask:

```text
cross_attention_logit_i += log(z_i + eps)
```

Training:

```text
z_i: relaxed ST mask
inference: hard selected mask
```

## Training

### Stage 0: Cache

Build an offline cache:

```text
candidate docs
candidate token ids
gold support labels/path
DAEC features
DAEC selected pseudo-labels
metadata
```

Output target:

```text
cache/dpathrag/2wiki_pilot_train.pt
cache/dpathrag/2wiki_pilot_dev.pt
```

### Stage 1: Warm Start

Train selector on:

```text
gold support path if ordered path is available
gold support set with canonical ordering otherwise
DAEC selected path only when gold supervision is unavailable
```

Early stop on support metrics, not selector NLL alone:

```text
support-complete@k
bridge recall
duplicate-title rate
path partial match
```

### Stage 2: Pure E2E Fine-Tune

Optimize only answer NLL:

```text
L = -log p(answer | q, C_q, z_ST)
```

No supporting-fact auxiliary loss in the final objective.

Required drift diagnostics:

```text
KL/JS from warm-start selector
top-k overlap with warm-start selector
gradient norm into selector
selector entropy
support-complete@k
bridge recall
answer EM/F1
duplicate-title rate
```

If answer F1 improves but support metrics degrade, mark as reward shortcut, not a method success.

## Baselines

### First Pilot Baselines

Run these before any large matrix:

```text
Dense top-k + same reader
DAEC + same reader
Unordered ST-Gumbel top-k selector
D-PathRAG-full
Gold support + same reader
```

### Paper Baselines

Add only after pilot passes:

```text
Supervised selector + same reader
Stochastic RAG-style unordered ST-Gumbel-top-k
Gumbel Reranking-style document-wise top-k attention mask
D-PathRAG w/o DAEC features
D-PathRAG w/o DAEC warm start
D-PathRAG frozen-listwise ablation
Qwen3 hard-reader eval
```

## Metrics

Answer:

```text
Answer EM
Answer F1
```

Evidence:

```text
support-complete@k
supporting paragraph recall
supporting fact F1
bridge entity recall
evidence path recall
```

Shortcut / collapse:

```text
duplicate-title rate
unique-title rate
answer-in-context-but-fail rate
selector entropy
selector KL/JS from warm start
gradient norm into selector
```

## 10-Day Kill-or-Continue Pilot

### Day 1-2: Split + Cache Audit

Tasks:

- Audit current 2Wiki data source.
- Confirm train/dev/test availability.
- Confirm support fields and evidence path fields.
- Confirm existing candidate pool compatibility.
- Build a small pilot cache schema.

Kill gate:

```text
No usable support/path labels and no train split available => do not start E2E training yet.
```

### Day 3-4: Reader Baseline

Run:

```text
Dense top-k + FiD/Flan reader
DAEC top-k + FiD/Flan reader
Gold support + FiD/Flan reader
```

Kill gate:

```text
gold support + reader <= dense top-k + reader + 3 F1 points
=> reader/data format is not strong enough; stop and fix reader pipeline.
```

### Day 5-6: Selector Warm Start

Run:

```text
D-PathRAG selector warm-started on gold support path/set
unordered selector warm-start baseline
```

Kill gate:

```text
warm-start support-complete@k cannot beat dense/DAEC selection
=> selector/path architecture is wrong; stop before E2E.
```

### Day 7-10: E2E Pilot

Run:

```text
D-PathRAG-full
unordered ST-Gumbel baseline
```

Continue gate:

```text
D-PathRAG support-complete@k > unordered ST-Gumbel + 3 pp
AND answer F1 not lower than DAEC + same reader
AND selector KL/JS from warm-start is non-zero
```

If support metrics do not improve, stop D-PathRAG.

## Engineering Layout

New package:

```text
src/dpathrag/
  __init__.py
  data.py
  cache.py
  features.py
  selector.py
  gumbel.py
  fid_reader.py
  train_warmstart.py
  train_e2e.py
  metrics.py
  baselines.py
  eval.py
```

Scripts:

```text
scripts/dpathrag_audit_2wiki.py
scripts/dpathrag_build_cache.py
scripts/dpathrag_train_reader.py
scripts/dpathrag_train_warmstart.py
scripts/dpathrag_train_e2e.py
scripts/dpathrag_run_baselines.py
scripts/dpathrag_eval.py
scripts/dpathrag_make_report.py
```

Reports:

```text
reports/dpathrag/2wiki_split_audit.md
reports/dpathrag/week1_reader_baseline.md
reports/dpathrag/week2_selector_warmstart.md
reports/dpathrag/week3_e2e_pilot.md
reports/dpathrag/week4_ablation.md
```

## Current Local Constraint

The current repo contains:

```text
reproduce/dataset/2wikimultihopqa.json
reproduce/dataset/2wikimultihopqa_corpus.json
```

The local file has 1000 examples and appears to be an evaluation subset, not a full train/dev/test release. It includes:

```text
supporting_facts
evidences
evidences_id
context
answer
```

This is enough for smoke/cache audit and small evaluation. It is not enough for the full D-PathRAG training plan unless a train split is added or downloaded.

## Implemented Baseline Artifacts

Implemented and verified on 2026-04-26:

```text
scripts/dpathrag_audit_2wiki.py
scripts/dpathrag_build_cache.py
src/dpathrag/{data,cache,metrics,gumbel,selector,io}.py
tests/dpathrag/
```

Generated reports:

```text
reports/dpathrag/2wiki_split_audit.json
reports/dpathrag/2wiki_split_audit.md
```

Generated smoke caches:

```text
data/dpathrag/cache/2wiki_dense_pool100_smoke.jsonl
data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl
```

Current audit result:

```text
local 2Wiki samples: 1000
local corpus docs: 6119
split status: local_eval_subset_only
Dense pool question mismatches: 0
PropRAG pool question mismatches: 0
Dense support-complete@100: 0.706
PropRAG support-complete@100: 0.968
```

Important cache boundary:

```text
gold_support is stored as a warm-start/evaluation label only.
It is not included in candidate input features to avoid leakage.
```

Next implementation step:

```text
Prepare the official 2Wiki train/dev split, then train Stage 1 selector warm start.
Until then, use the local 1000-example subset only for smoke/cache validation.
```

## Full 2Wiki Data Policy

The existing 1000-example protocol remains:

```text
reproduce/dataset/2wikimultihopqa.json
reproduce/dataset/2wikimultihopqa_corpus.json
```

Full HuggingFace train/dev/test data must be kept separate:

```text
data/dpathrag/full_2wiki/
```

Prepare it with:

```text
HF_ENDPOINT=https://hf-mirror.com .venv-hipporag/bin/python scripts/dpathrag_prepare_2wiki_full.py
```

This writes normalized split files:

```text
data/dpathrag/full_2wiki/2wikimultihopqa_train.json
data/dpathrag/full_2wiki/2wikimultihopqa_validation.json
data/dpathrag/full_2wiki/2wikimultihopqa_test.json
data/dpathrag/full_2wiki/2wikimultihopqa_full_corpus.json
data/dpathrag/full_2wiki/manifest.json
```

Prepared on 2026-04-26 via `HF_ENDPOINT=https://hf-mirror.com`:

```text
train rows: 167454
validation rows: 12576
test rows: 12576
full corpus unique docs: 430225
full corpus unique titles: 398354
corpus dedupe key: title+text
```

The title-only corpus merge was explicitly avoided because the full release
contains same-title contexts with different text.  D-PathRAG full-corpus
experiments should use the `idx` field in `2wikimultihopqa_full_corpus.json`
rather than assuming title uniqueness.
