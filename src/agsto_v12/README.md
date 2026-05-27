# AG-STO v12

Standalone package for the clean AG-STO v12 mainline:

```text
AG-STO: Anchor-Guided Source-Title-OpenIE Evidence Set Retrieval
```

This folder packages the method-facing retrieval code only. It is intentionally
separate from historical diagnostics, native score-chasing scripts,
support_fusion experiments, and paper/report generation utilities.

## Boundary

| Item                              | Included |
| --------------------------------- | -------- |
| Source/Title/OpenIE graph index   | yes      |
| Stable anchor graph entry         | yes      |
| STO proposal lanes                | yes      |
| Graph-constrained evidence set    | yes      |
| v12 graph-obligated completion    | yes      |
| Native proposal fallback          | yes      |
| support_fusion                    | no       |
| answer/gold-aware diagnostics     | no       |
| local PPR bridge variants         | no       |
| historical analysis scripts       | no       |

## Package Contents

| File                      | Role                                                       |
| ------------------------- | ---------------------------------------------------------- |
| `config.py`               | Frozen AG-STO v12 config; default `policy="graph"`         |
| `index.py`                | Source/Title/OpenIE graph construction                     |
| `lexical.py`              | BM25-style lexical scoring over STO doc tokens             |
| `ranking.py`              | Ranking and proposal-consensus helpers                     |
| `scoring.py`              | Graph-constrained evidence-set objective                   |
| `completion.py`           | Conservative graph-obligated completion                    |
| `proposal_roles.py`       | Structured metadata for proposal lanes                     |
| `proposals.py`            | Package-owned native STO proposal generation               |
| `selector.py`             | Evidence-set selection and final retrieval assembly        |
| `retriever.py`            | Public `AGSTORetriever` facade                             |
| `build_openie.py`         | Raw corpus -> AG-STO-compatible OpenIE JSON adapter        |
| `evaluate_cached.py`      | Cached transition-report evaluator for frozen reproduction |
| `evaluate_from_openie.py` | Native OpenIE -> STO graph -> AG-STO retrieval evaluator   |
| `evaluate_from_corpus.py` | Raw corpus -> OpenIE -> AG-STO retrieval evaluator         |

## Algorithm Flow

### 1. Graph Construction

Input documents should contain:

```text
idx
passage
extracted_triples
```

`build_corpus_unit_index(openie_docs)` builds a lightweight STO substrate:

```text
passage -> source_span units
passage -> openie_fact units
unit -> informative endpoints
endpoint -> docs
```

Two documents are connected at retrieval time when they share a low-degree
informative endpoint. The edge weight is the sum of endpoint IDF values.

### 2. Retrieval

`AGSTORetriever.retrieve(...)` takes a query and optional proposal lanes:

```text
anchor_doc_indices
native_dense_doc_indices
bm25_doc_indices
specificity_doc_indices
endpoint_transition_doc_indices
hybrid_residual_doc_indices
query_conditioned_neighborhood
support_set_search
```

The selector builds a query-local candidate pool and selects a fixed-budget
evidence set using:

```text
score(S) = query_utility(S) + graph_support(S) - set_cost(S)
```

For AG-STO v12, `policy="graph"` applies one conservative graph-obligated
completion step after the initial evidence set is selected.

### 3. QA

This package stops at retrieval. The selected evidence set is exposed as:

```text
result["selected_evidence_set"]["doc_indices"]
result["retrieved_doc_indices"][:top_k]
```

The QA pipeline should load the corresponding passage texts and pass them to
the reader as ordinary evidence context. No gold labels, answer-aware verifier,
or support_fusion rule is used by this package.

## Cached Evaluation

Run the paper-facing cached-proposal retrieval evaluation with:

```bash
python -m agsto_v12.evaluate_cached \
  --transition-report outputs_full_sfb_supportfusion_gpt4omini_qwen_cleanbaseline_rebuild_20260425/reports/transition_component_retriever_full_native_denseanchor20_context10_support400.json \
  --datasets 2wikimultihopqa,musique,hotpotqa \
  --policy graph \
  --output-json outputs_agsto_v12/reports/agsto_v12_cached.json \
  --output-md outputs_agsto_v12/reports/agsto_v12_cached.md
```

This evaluator uses cached STO proposal lanes only. It does not run QA,
support_fusion, native proposal diagnostics, or local PPR bridge variants.

## Native OpenIE Evaluation

Run the fully package-owned retrieval path from OpenIE documents:

```bash
python -m agsto_v12.evaluate_from_openie \
  --openie-results path/to/openie_results_ner_gpt-4o-mini.json \
  --queries-json path/to/dataset_questions.json \
  --dataset 2wikimultihopqa \
  --policy graph \
  --max-queries 0 \
  --output-json outputs_agsto_v12/reports/agsto_v12_from_openie.json \
  --output-md outputs_agsto_v12/reports/agsto_v12_from_openie.md
```

This path builds the lightweight STO graph in memory from OpenIE docs, creates
native STO proposal lanes, and runs the same AG-STO v12 selector. It is useful
for portability and clean package execution. Cached-proposal evaluation remains
the frozen-result reproduction path.

The query JSON can be either:

```text
a list of rows
a mapping with rows
a mapping with datasets[].rows
```

Gold labels are optional. If present, retrieval metrics use `gold_doc_indices`.
For raw HippoRAG-style query rows, supporting paragraph indices are inferred
from `paragraphs[].idx` where `paragraphs[].is_supporting` is true.

## Raw Corpus Evaluation

Run from raw corpus JSON by first building an AG-STO-compatible OpenIE cache:

```bash
python -m agsto_v12.evaluate_from_corpus \
  --corpus-json path/to/dataset_corpus.json \
  --queries-json path/to/dataset_questions.json \
  --dataset 2wikimultihopqa \
  --openie-mode llm \
  --llm-base-url https://api.openai.com/v1 \
  --llm-model gpt-4o-mini \
  --api-key-file /path/to/api_key_file \
  --policy graph \
  --max-queries 0 \
  --output-json outputs_agsto_v12/reports/agsto_v12_from_corpus.json \
  --output-md outputs_agsto_v12/reports/agsto_v12_from_corpus.md
```

The raw corpus format is the HippoRAG corpus format:

```json
[
  {"idx": 0, "title": "Passage Title", "text": "Passage text..."}
]
```

For dependency-free smoke tests, use:

```bash
python -m agsto_v12.evaluate_from_corpus \
  --corpus-json path/to/dataset_corpus.json \
  --queries-json path/to/dataset_questions.json \
  --openie-mode heuristic \
  --output-json outputs_agsto_v12/reports/agsto_v12_from_corpus_smoke.json \
  --output-md outputs_agsto_v12/reports/agsto_v12_from_corpus_smoke.md
```

`heuristic` mode is only a portability smoke path. Use `llm` mode for real
OpenIE extraction and performance-facing runs. API keys are read from
`--api-key-file`, `--api-key`, or `OPENAI_API_KEY`; the package does not print
the key.

## Minimal Example

```python
from agsto_v12 import AGSTOConfig, AGSTORetriever

docs = [
    {
        "idx": "0",
        "passage": "Alice\nAlice was born in Paris.",
        "extracted_triples": [("Alice", "was born in", "Paris")],
    },
    {
        "idx": "1",
        "passage": "Paris\nParis is in France.",
        "extracted_triples": [("Paris", "is in", "France")],
    },
]

retriever = AGSTORetriever.from_openie_docs(
    docs,
    config=AGSTOConfig(evidence_set_size=2, retrieval_top_k=2),
)
result = retriever.retrieve(
    query="Where was Alice born?",
    anchor_doc_indices=[0],
    specificity_doc_indices=[1],
    query_conditioned_neighborhood={
        "selected_neighborhood_doc_indices": [0, 1],
        "retrieved_doc_indices": [0, 1],
    },
    support_set_search={
        "retrieved_doc_indices": [0, 1],
        "top_support_sets": [{"doc_indices": [0, 1], "score": 10.0}],
    },
)
print(result["selected_evidence_set"]["doc_indices"])
```

## Naming

Use this name in method-facing text:

```text
AG-STO: Anchor-Guided Source-Title-OpenIE Evidence Set Retrieval
```

Use `AG-STO v12` only as an internal implementation/checkpoint label.
