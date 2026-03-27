# HippoRAG Project Guide

## Overview

HippoRAG is a multi-hop QA retrieval system. The current work is on branch `planner-hipporag-integration`, focused on improving retrieval accuracy (Recall@K, EM, F1) on multi-hop QA benchmarks (2WikiMultiHopQA, HotpotQA).

## Current Best Configuration

The validated best config on 2Wiki-100 (EM=0.46, F1=0.5188, Recall@5=0.845):

```
causal_engine_version = v2
causal_v2_base_retrieval_mode = legacy_fact_graph
causal_enabled = false
causal_context_max_items = 0
```

This uses the original HippoRAG fact-graph PPR retrieval with properly aligned embedding assets. No V2 "algorithm enhancements" are active.

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

## Active Development Direction

**General relation graph PPR**: Use the V2 general extraction (8 typed relations) to build an igraph for PPR-based retrieval. This is the one remaining "algorithm innovation" direction not yet proven negative.

Key constraints for implementation:
- Hybrid seed generation (NER overlap + dense doc entities + embedding top-K), not raw query-to-entity embedding
- Anti-hub weight decay (`1/log2(chunk_count+1)`), hub entities like "William the Silent" (degree 642) will corrupt PPR otherwise
- `related_to` edge downweighting (47.7% of edges are `related_to`, too fuzzy for full weight)
- Accept undirected PPR for v1 (existing `run_ppr()` uses `directed=False`)

Step 0 index cache: `outputs_step0_general_2wikimultihopqa/qwen3-8b_VLLM__mnt_nvme_Qwen3-Embedding-8B/causal_v2/`

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
