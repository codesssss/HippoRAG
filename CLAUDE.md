# HippoRAG Project Guide

## Overview

HippoRAG is a multi-hop QA retrieval system.

Current branch roles:
- Default development branch: `feature/pcrs-rag-v1`
- Paper-facing mainline: `bridge_beam + set_closure + pathcore_guard + reserve3 + dedup`

This means current engineering work should default to the `feature/pcrs-rag-v1` branch, while paper claims should continue to treat the simple `pathcore_guard` stack as the frozen stable line until the new branch clears its promotion gates.

## Current Research Memory

For the active EMNLP paper direction, use the files under `research_memory/emnlp_expand_then_compose/` as the canonical source of truth before proposing new ideas or experiments.

Read order:
1. `research_memory/emnlp_expand_then_compose/00_north_star.md`
2. `research_memory/emnlp_expand_then_compose/01_claim_ledger.md`
3. `research_memory/emnlp_expand_then_compose/04_result_registry.md`
4. `research_memory/emnlp_expand_then_compose/03_experiment_board.md`
5. `research_memory/emnlp_expand_then_compose/02_decision_log.md`

Rules:
- Do not reframe the current paper as causal retrieval.
- Do not reframe the current paper as pointwise reranking.
- If the main framing changes, update the decision log first.
- If a claim becomes supported or contradicted, update the claim ledger.

## Current Best Configuration

The validated best config on 2Wiki-100 (EM=0.46, F1=0.5188, Recall@5=0.845):

```
causal_engine_version = v2
causal_v2_base_retrieval_mode = legacy_fact_graph
causal_enabled = false
causal_context_max_items = 0
```

This uses the original HippoRAG fact-graph PPR retrieval with properly aligned embedding assets. No V2 "algorithm enhancements" are active.

## Hardware & Model Services

**Server**: 8x NVIDIA H100 80GB HBM3, Intel Xeon Platinum 8480+ (224 cores), 2TB RAM

| Service | Model | Port | GPU | Serving |
|---|---|---|---|---|
| LLM (QA reader + extraction) | Qwen3-8B (`/mnt/nvme/Qwen3-8B`) | `http://localhost:8039/v1` | GPU 0,2 (TP=2) | VLLM, served-model-name=`qwen3-8b` |
| Embedding (dense passage) | Qwen3-Embedding-8B (`/mnt/nvme/Qwen3-Embedding-8B`) | `http://localhost:8018/v1/embeddings` | GPU 6 (TP=1) | VLLM, served-model-name=`/mnt/nvme/Qwen3-Embedding-8B` |
| Embedding (legacy fact PPR) | NV-Embed-v2 | Local (transformers) | Auto-loaded on demand | HF cache: `/mnt/nvme/hf/models--nvidia--NV-Embed-v2` |

Other services on this machine (not HippoRAG, do not touch):
- `qwen3-8b-train` on port 8043 (GPU 3,4 TP=2) - R-HAN training
- `qwen3-32b-judge` on port 8045 (GPU 5,7 TP=2) - R-HAN judge
- Ollama on port 11434 (llama2, deepseek-r1)

## Embedding Models

Two embedding models are in use. Do NOT mix their embedding spaces.

| Purpose | Model | Notes |
|---|---|---|
| Dense passage retrieval | Qwen3-Embedding-8B (VLLM) | Used for query-to-passage matching |
| Legacy fact/entity matching (PPR seeds) | NV-Embed-v2 | Used for query-to-fact matching when `base_retrieval_mode=legacy_fact_graph` |
| V2 general graph entities | Qwen3-Embedding-8B | V2 extraction uses this; Step 0 index cache built with it |

When the V2 path reuses legacy assets, it automatically instantiates a separate NV-Embed-v2 model for fact retrieval (`_active_embedding_model_for_fact_retrieval()`). Do not bypass this alignment.

## Proven Negative Directions (Do NOT Re-implement)

These have been experimentally disproven on 2Wiki/HotpotQA. They were removed in commit `1ad3f44`.

1. **Candidate injection** - Using relation graph to inject new documents into the dense top-K candidate set. Introduced distractors, Recall@5 dropped from 0.845 to 0.805, EM dropped from 0.43 to 0.37.

2. **Reader-side graph context injection** - Serializing relation chains and appending them to the QA prompt as "Causal Graph Context". In 12/12 hurt cases, reader was misled into outputting "can't determine" or wrong Yes/No. `causal_context_max_items` defaults to 0 for this reason.

3. **Additive boost rerank** - Adding a fixed score bonus to documents found via graph traversal. Cannot move documents from rank 15+ into top-5; the score gap is too large for additive correction.

4. **Semantic intent router (anchor bank)** - Embedding-similarity router classifying queries as causal vs standard. On 2Wiki, 100% of queries classified as standard (max causal score 0.396). Multi-hop QA datasets are fact-chain tasks, not causal reasoning tasks.

5. **Causal-only extraction (cause/enable/prevent)** - These three relation types do not cover the bridge relations in multi-hop QA (director_of, parent_of, born_in, etc.).

6. **General relation graph PPR** (commit `74dee61`) - Built V2 general extraction graph (8 typed relations, 19.5k entities, 12.6k edges) and ran PPR-based retrieval. Initial result was catastrophic (Recall@5 0.603 vs baseline 0.845). Applied 3-patch fix: (A) seed cap to 12 + reset budget entity=0.6/passage=0.4, (B) degree-aware hub decay `1/sqrt(log2(deg_s+1)*log2(deg_t+1))`, (C) related_to weight sweep 0.0/0.1/0.3. Best result: Recall@5=0.695, Recall@2=0.610. Still 0.150 below legacy baseline. Root causes: V2 entity resolution too noisy (many near-duplicate entities), entity-passage edge topology fundamentally different from legacy fact-graph's curated triples.

## Active Development Direction

For the retrieval backbone, the aligned `legacy_fact_graph` path remains the strongest validated base retriever.

The current research direction is not another causal or general-graph retrieval variant. The active EMNLP direction is evidence composition over a widened candidate pool:

- Core framing: hard multi-hop failures come from incomplete evidence sets, not only from bad pointwise scores
- Working story: `Expand-then-Compose`
- Immediate method target: simple setwise or bridge-aware evidence selection from a larger top-K pool
- Main analysis dataset: `2WikiMultihopQA`
- Cross-dataset ceiling checks: `HotpotQA`, `MuSiQue`

Current branch split:
- Paper mainline stays on the simple `pathcore_guard` selector stack
- Default development work proceeds on `feature/pcrs-rag-v1`
- `requirement_beam` is the active experimental selector on that branch
- Do not present `requirement_beam` as the paper default unless it clears the explicit validation gates recorded under `research_memory/emnlp_expand_then_compose/`

Retrieval-side negative results remain important, but now serve as motivation for the pivot rather than an unfinished algorithm branch.

## Code Structure

Key files:
- `src/hipporag/HippoRAG.py` - Main retrieval pipeline. `retrieve_v2()` is the V2 entry point, `_get_v2_base_retrieval()` routes between dense/legacy_fact_graph/general_relation_graph.
- `src/hipporag/causal_v2.py` - V2 extraction engine (`CausalV2Engine`). Supports `graph_mode=causal` and `graph_mode=general`.
- `src/hipporag/utils/config_utils.py` - All configuration fields (`BaseConfig` dataclass).
- `scripts/eval_causal_qwen3.py` - Evaluation harness for running experiments.

## Build & Test

```bash
# Syntax check
python -m py_compile src/hipporag/causal_v2.py
python -m py_compile src/hipporag/HippoRAG.py

# Run tests (no pytest in venv, use direct execution)
.venv-hipporag/bin/python tests/test_causal_v2.py
.venv-hipporag/bin/python tests/test_causal_utils.py
```

## Experiment Baselines

Always compare against these baselines on 2Wiki-100:

| Config | Recall@5 | EM | F1 |
|---|---:|---:|---:|
| Original HippoRAG baseline | 0.845 | 0.43 | 0.5005 |
| Aligned legacy_fact_graph_v2 (current best) | 0.845 | 0.46 | 0.5188 |

On HotpotQA-100:

| Config | Recall@5 | EM | F1 |
|---|---:|---:|---:|
| Original HippoRAG baseline | 0.885 | 0.54 | 0.6608 |
| Aligned legacy_fact_graph_v2 | 0.925 | 0.57 | 0.6828 |
