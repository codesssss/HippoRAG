# Path A: SetR-Aware Widened-Pool DAEC Plan - 2026-04-26

## Decision

Proceed with **Path A**:

> Reframe DAEC from generic set selection to fixed-pool evidence composition in widened GraphRAG pools.

The core paper should no longer claim "we select collectively comprehensive evidence sets" as the main novelty, because SetR already occupies that framing. The sharper contribution is:

> SetR studies set selection over top-20 retrieval contexts. DAEC studies dependency-bound composition over widened GraphRAG / dense pools, where required bridge evidence is often buried beyond top-20 and generic set selection is brittle.

## SetR Audit

Local repo:

- `/mnt/nvme/code/SetR`
- latest local commit: `d1ac78b Fix paper title`

Findings:

- Official prompts are available in `generate_data.py`.
- Main prompt of interest: `selection_IRI`.
- README states evaluation is `TBD`.
- No complete answer-reader evaluation script is provided.
- `generate_data.py` defaults to contexts and uses `contexts[:20]` for `contexts`-style input.
- `convert_rankify.py` hard-codes `rank < 21` and fallback over `range(20)`.
- Therefore official SetR code is not directly top-50/top-100 compatible.

Implication:

- We can fairly implement a **SetR-style IRI selector** using their released prompts.
- We should explicitly document that the official repo is top-20/evaluation-incomplete, and that our widened-pool comparison uses an adapter with the same prompt semantics and same reader protocol.

## New Claim Boundary

### Primary Claim

Widened GraphRAG pools expose composition headroom that top-20 set selection does not solve.

Evidence needed:

- support/bridge depth distribution showing required evidence beyond top-20;
- oracle@100 headroom across Dense and PropRAG pools;
- DAEC top-50/top-100 gap recovery exceeding SetR-style selection.

### Secondary Claim

Dependency binding is load-bearing for widened-pool composition.

Evidence needed:

- no-binding drop across multiple dataset/pool pairs;
- no-decomposition/global-query ablation;
- bridge-vs-anchor subset split showing the binding effect concentrates on bridge-heavy cases.

### Anti-Claim To Defeat

DAEC is just SetR with another prompt/scoring function.

Defense:

- direct SetR-style baseline on identical pools;
- K-scaling comparison at K=20/50/100;
- mechanism ablation that SetR does not have: dependency binding and repairability.

## Must-Run Blocks

### B0: SetR Adapter

Build:

- pool-to-SetR JSONL converter;
- SetR selection prompt runner or direct prompt integration into current Qwen endpoint;
- output parser for `### Final Selection: [2] [1]`;
- fallback to original rank order when parsing fails;
- selected top-5 evidence handoff to the same Qwen3-8B reader used by DAEC.

Do not train SetR. The goal is SetR-style inference baseline, not reproducing their fine-tuned checkpoint unless checkpoint is available and cheap.

### B1: Bridge Depth Diagnostic

Compute for each dataset/pool:

- gold support paragraph rank distribution;
- bridge/final-hop support rank where decomposition allows;
- fraction with any support beyond top-20;
- fraction with all support within top-20/top-50/top-100;
- median / p75 / p90 support depth.

This becomes the intro motivation figure.

### B2: K-Scaling Main Comparison

Run:

- pools: Dense top100, PropRAG top100;
- datasets: 2Wiki, HotpotQA, MuSiQue;
- K: 20, 50, 100;
- systems: baseline rank order, SetR-style IRI, DAEC, oracle select@K.

Metrics:

- Answer EM/F1;
- support recall;
- gap recovery;
- latency / LLM calls.

Main success condition:

- DAEC has stronger top-100 gap recovery than SetR-style IRI, especially on 2Wiki and dense pools.

### B3: Mechanism Ablations

Run or consolidate:

- full DAEC;
- no binding;
- no decomposition / global query;
- gold decomposition if available;
- LLM decomposition;
- bridge vs anchor question split.

This is the novelty isolation table.

### B4: Negative Controls

Use existing:

- QBF negative result;
- BSGS oracle-slot MuSiQue-1000 negative result;
- BSGS MuSiQue-200 oracle pool/binding/likelihood diagnostic.

No more BSGS engineering.

## Run Order

1. Adapter first: SetR input/output compatibility smoke on 5 examples.
2. Bridge-depth diagnostic from existing pool exports.
3. 2Wiki pilot100 K-scaling: SetR-style vs DAEC at K=20/50/100.
4. If pilot shows DAEC > SetR at widened K, launch full 3 dataset x 2 pool x 3 K matrix.
5. Add no-decomposition/global-query and bridge/anchor split.
6. Convert results into paper tables and update related work.

## Stop / Go Gates

### Gate 1: SetR Pilot

If SetR-style IRI matches or beats DAEC at K=100 on 2Wiki pilot100:

- stop the main-track Path A expansion;
- either narrow to a mechanism paper around binding or target Findings.

### Gate 2: Bridge Depth

If most gold supports are already in top-20:

- weakened widened-pool motivation;
- paper should avoid claiming SetR missed a major regime.

### Gate 3: Binding

If no-decomposition/global-query and no-binding are close to full DAEC:

- DAEC novelty collapses to scoring/reranking;
- do not frame binding as main contribution.

## Current Priority

Next implementation task:

```text
Build SetR adapter and run 2Wiki smoke5/pilot100 over existing Dense or PropRAG top100 pool.
```

Implemented files:

- `scripts/export_setr_inputs.py`
- `scripts/run_setr_style_selector.py`
- `scripts/apply_setr_selection_to_pool.py`
- `scripts/analyze_pool_support_depth.py`
- `tests/test_setr_adapter.py`

Keep all SetR-specific logic additive. Do not modify the frozen DAEC main path unless adding a clean selector hook is unavoidable.

## Implementation Status - 2026-04-26

Implemented additive SetR adapter utilities:

- `scripts/export_setr_inputs.py`: converts Dense/PropRAG external-pool JSON into SetR-style selector JSONL with configurable `pool_k` and `doc_max_chars`.
- `scripts/run_setr_style_selector.py`: runs the official SetR `selection_IRI` prompt against an OpenAI-compatible endpoint, with resumable JSONL output and a mock-rank mode for smoke tests.
- `scripts/apply_setr_selection_to_pool.py`: parses SetR-style selections and frontloads selected positions in an external-pool JSON so existing `eval_causal_qwen3.py --setwise_selector none` can evaluate it.
- `scripts/analyze_pool_support_depth.py`: computes support-rank depth, beyond-top20 rate, query coverage at K=20/50/100, and hop-level support ranks when decomposition is available.
- `tests/test_setr_adapter.py`: covers parser filtering, pool reordering, SetR input export, and support-depth matching.

Validation:

- `pytest tests/test_setr_adapter.py`: `5 passed`.
- `py_compile` passed for all four new scripts.
- Mock smoke passed on `run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json` with `limit=5`.
- Reordered mock pool loaded successfully through `scripts/eval_causal_qwen3.py --retrieval_only true`; external-pool alignment was exact (`500/500` exact doc matches).

Next run:

```text
Run real SetR-style Qwen selection on 2Wiki pilot100 for K=20/50/100, then evaluate selected pools with the existing reader protocol.
```
