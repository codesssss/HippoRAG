# Evidence Transition GraphRAG v3 Variable-Flow

This folder keeps the variable-flow Evidence Transition branch isolated from
the frozen ET v1 line and the ETV2 channel-direct diagnostic.

## Boundary

| Item | Setting |
| --- | --- |
| Folder | `evidence_transition_graphragv3_variable_flow` |
| Method line | `evidence_transition_graphragv3_variable_flow` |
| Required switch | `--enable-variable-flow-traversal` |
| Forbidden switch | `--ablation-query-supported-object-handoff` |

The entrypoint automatically enables variable-flow traversal and rejects the
old query-supported object-handoff ablation.  To avoid changing frozen shared
ET files, this package carries a local copy of the source-authorized pipeline
and AG-STO graph modules needed by this method.

## Example

```bash
env PYTHONPATH=. /mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  evidence_transition_graphragv3_variable_flow/run_fresh_e2e.py \
  --output-root run_logs/evidence_transition_graphragv3_variable_flow_limit100 \
  --reuse-current-fresh-index \
  --max-queries 100 \
  --datasets 2wikimultihopqa,musique,hotpotqa
```

The wrapper automatically adds `--enable-variable-flow-traversal`.

## Reader-only QA

Use the package-local QA runner for ETv3 reports:

```bash
env PYTHONPATH=. /mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  evidence_transition_graphragv3_variable_flow/run_reader_qa.py \
  --retrieval-json run_logs/evidence_transition_graphragv3_variable_flow_limit100/musique/reports/musique_evidence_transition_graphragv3_variable_flow_retrieval.json \
  --max-queries 100 \
  --qa-top-k 5 \
  --llm-name qwen3-8b-train \
  --llm-base-url http://localhost:8043/v1 \
  --embedding-name VLLM/nvidia/NV-Embed-v2 \
  --embedding-base-url http://localhost:8019/v1/embeddings \
  --output-json run_logs/evidence_transition_graphragv3_variable_flow_limit100/reports/evidence_transition_graphragv3_variable_flow_reader_only_qa.json \
  --output-md run_logs/evidence_transition_graphragv3_variable_flow_limit100/reports/evidence_transition_graphragv3_variable_flow_reader_only_qa.md
```

This runner directly reads ETv3 `retrieved_doc_indices_top5`, loads the minimal
query/gold metadata, and calls the standard HippoRAG reader metric path. It does
not require an SFB variant, does not re-index the corpus, and does not run
triple or causal relation extraction.
