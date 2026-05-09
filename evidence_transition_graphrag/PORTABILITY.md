# Portability Guide

This directory is now a portable overlay for a HippoRAG-style repository.

## What Is Included

| Path | Purpose |
| --- | --- |
| `contract.py` | Public method name and clean-contract flags |
| `run_fresh_e2e.py` | Public runner entry |
| `check_portable_install.py` | Verifies that the overlay and target dependencies are present |
| `portable_overlay/run_query_grounded_sto_fresh_e2e.py` | Fresh indexing + retrieval runner |
| `portable_overlay/evidence_transition_top5_qa_adapter.py` | Registers the method with the target QA runner without overwriting it |
| `portable_overlay/source_authorized_vocab_strict_retrieval/` | Candidate generation, E2E retrieval boundary, QA export |
| `portable_overlay/src/agsto/` | STO evidence graph construction and query-local graph retrieval |
| `portable_overlay/agsto/` | Compatibility import package for `src.agsto` |
| `install_overlay.py` | Copies the method package and overlay into a target repo |

## Target Repository Assumptions

The target repository must already be a working HippoRAG-style checkout with:

| Required target dependency | Why |
| --- | --- |
| `src/hipporag/` | Embedding store, OpenIE, LLM config, HippoRAG utilities |
| benchmark data under `reproduce/dataset/` | The fresh runner builds reports from raw datasets |
| Python environment with the normal HippoRAG dependencies | OpenIE, embeddings, reader, numpy/pandas/scipy/etc. |
| optional `run_transition_top5_qa.py` | Needed only if using `--run-qa` in the target repo; the overlay does not overwrite it |

## Install Into Another Repo

From this repo:

```bash
python evidence_transition_graphrag/install_overlay.py --target /path/to/target/HippoRAG
```

Or copy the `evidence_transition_graphrag/` directory to the target repo, then run:

```bash
python evidence_transition_graphrag/install_overlay.py --target .
```

Verify the install:

```bash
python evidence_transition_graphrag/check_portable_install.py --target /path/to/target/HippoRAG
```

## Run Retrieval

```bash
python -m evidence_transition_graphrag.run_fresh_e2e \
  --datasets 2wikimultihopqa,musique,hotpotqa \
  --output-root run_logs/evidence_transition_gpt4omini_limit100 \
  --max-queries 100 \
  --llm-name gpt-4o-mini \
  --llm-base-url https://yunwu.ai/v1 \
  --embedding-name nvidia/NV-Embed-v2 \
  --embedding-base-url http://localhost:8019/v1/embeddings
```

Add `--run-qa` only after the target repo's original `run_transition_top5_qa.py`
is available. The installed `evidence_transition_top5_qa_adapter.py` registers
`evidence_transition_graphrag` with that target QA runner at runtime.

## Current Verified Limit-100 Result

| Dataset | R@5 | EM | F1 |
| --- | ---: | ---: | ---: |
| 2Wiki | 0.9075 | 0.5700 | 0.6447 |
| MuSiQue | 0.7225 | 0.3900 | 0.4686 |
| HotpotQA | 0.9500 | 0.6000 | 0.7230 |

These numbers were produced from fresh GPT-4o-mini OpenIE and NV-Embed-v2
artifacts in this checkout.
