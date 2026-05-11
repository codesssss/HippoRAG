# Evidence Transition GraphRAG v4 Fact-Witnessed STO

This folder is an isolated V4 line. It does not modify the V3 package.

## Boundary

| Item | Setting |
| --- | --- |
| Folder | `evidence_transition_graphragv4_fact_witnessed_sto` |
| Method | `evidence_transition_graphragv4_fact_witnessed_sto` |
| Default runner | `query_grounded_sto_clean_mainline_v4` |
| Retrieval unit | document |
| Fact unit policy | OpenIE facts define/witness STO document edges; they are not ranked as reader items |
| Proposition retrieval | no |
| Dense policy | query-local graph entry only |
| Source-prior guard | no in the paper-facing mainline |
| Certificate closure selection | no in the paper-facing mainline |
| Forbidden switch | `--ablation-query-supported-object-handoff` |

## Algorithm

| Stage | Logic |
| --- | --- |
| 1. Indexing | Build fresh OpenIE and passage embedding artifacts from raw corpus. |
| 2. Entry | Use dense/textual retrieval only to seed a query-local STO document graph. |
| 3. Graph construction | Build STO document edges from title, endpoint, role transition, and variable-flow evidence. |
| 4. Fact witness | An STO edge can be promoted only when its local edge sample is backed by OpenIE fact unit ids. |
| 5. Readout | Select top5 by fact-witnessed branch readout over the query-local STO graph. |
| 6. QA | Feed fixed top5 documents to the normal reader-only QA path. |

This is not PropRAG-style proposition retrieval. The final retrieval object is
still a document context; OpenIE facts only witness document-to-document STO
transitions.

## Query Grounding Contract

| Rule | Meaning |
| --- | --- |
| Exact mention | If a normalized endpoint phrase appears contiguously in the query and has a named surface cue, it can be a symbolic STO root. |
| Non-contiguous lexical match | If an endpoint is only matched by token subset overlap, it can become a symbolic root only when it is also a corpus title endpoint. |
| Relation words | Relation / answer-type words guide textual entry and edge ordering, but should not become standalone graph roots. |

This is a graph-construction contract, not a score. It prevents phrases like
`american defeat` from acting as entity roots while preserving exact mentions
and title entities such as `treaty of versailles`.

## Mainline Readout Contract

| Rule | Meaning |
| --- | --- |
| Base readout | Use fact-witnessed STO document transitions, not proposition ranking or weighted score fusion. |
| Multi-branch case | When the query-local graph exposes confirmed symbolic roots, give each branch a transition witness before fallback ordering. |
| Chain case | If the branch condition is not structurally confirmed, use the fact-witnessed source-prior order inside the local graph. |
| Symbolic-only root precision | In multi-anchor queries, a symbolic-only root may seed graph admission but cannot promote frontier docs unless also supported by textual entry. Single-anchor symbolic roots may still promote fact-witnessed frontier docs. |
| Guard policy | Do not run source-prior guard in the paper-facing mainline. |
| Certificate policy | Do not use source-text certificate closure as a selection stage in the paper-facing mainline. |

This is not dataset routing. It is a graph-shape readout over the query-local
STO graph. Dense/textual retrieval opens the graph; it does not protect final
top5 by a separate guard.

## Readout Policies

| Policy | Status | Logic |
| --- | --- | --- |
| `clean_mainline` | default / paper-facing | Dense/textual entry -> query-local STO graph -> fact-witnessed branch readout. No source-prior guard, no certificate-closure selection. |
| `fact_witnessed` | legacy ablation | Guarded fact-witnessed source-prior readout kept for reproduction. |
| `branch_balanced` | legacy ablation | Guarded form of the positive branch-balanced readout. |
| `fact_witnessed_path_cover` | legacy negative ablation | Tested as endpoint/path cover on MuSiQue limit=100; it reduced R@5. |
| `transition_valid_closure` | legacy negative ablation | Producer-consumer transition-valid closure; lower than current mainline. |
| `root_balanced_transition` | legacy negative ablation | Global root balancing; helps 2Wiki but hurts MuSiQue/HotpotQA. |
| `transition_closure` | legacy negative ablation | Pure closure without source-prior protection; sharply reduced MuSiQue R@5. |

The legacy policies are kept for ablation/reproduction only. Do not use them as
the reported V4 result unless a later revision changes the result profile and
updates this note.

## Example

```bash
env PYTHONPATH=. /mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  evidence_transition_graphragv4_fact_witnessed_sto/run_fresh_e2e.py \
  --output-root run_logs/evidence_transition_graphragv4_fact_witnessed_sto_limit100 \
  --reuse-current-fresh-index \
  --max-queries 100 \
  --datasets 2wikimultihopqa,musique,hotpotqa
```

The wrapper automatically enables variable-flow traversal and rejects the old
query-supported same-object diagnostic ablation.

## Reader-only QA

Use the package-local QA runner for V4 reports:

```bash
env PYTHONPATH=. /mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  evidence_transition_graphragv4_fact_witnessed_sto/run_reader_qa.py \
  --retrieval-reports run_logs/evidence_transition_graphragv4_fact_witnessed_sto_limit100/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json \
  --max-queries 100 \
  --qa-top-k 5 \
  --llm-name qwen3-8b-train \
  --llm-base-url http://localhost:8043/v1 \
  --max-new-tokens 400 \
  --embedding-name nvidia/NV-Embed-v2 \
  --embedding-base-url http://localhost:8019/v1/embeddings \
  --output-json run_logs/evidence_transition_graphragv4_fact_witnessed_sto_limit100/reports/evidence_transition_graphragv4_fact_witnessed_sto_reader_only_qa.json \
  --output-md run_logs/evidence_transition_graphragv4_fact_witnessed_sto_limit100/reports/evidence_transition_graphragv4_fact_witnessed_sto_reader_only_qa.md
```

The QA runner directly reads `retrieved_doc_indices_top5`. It does not require
an SFB variant, does not re-index the corpus, and does not run causal relation
extraction.
