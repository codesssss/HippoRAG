# PropRAG / DAEC / DAEC-L1 / NEOCORRAG / HGRAG Limit-100 Protocol

Date: 2026-04-29

Status: active execution protocol for the current limit-100 aligned smoke.

Primary run directory:

```text
run_logs/prop_neocorr_daec_limit100_20260429/
```

Primary summary artifacts:

```text
reports/dpathrag/prop_neocorr_daec_limit100_20260429.md
reports/dpathrag/prop_neocorr_daec_limit100_20260429.json
```

NEOCORRAG native launcher:

```text
run_logs/launch_neocorrag_limit100_20260429.sh
```

HGRAG native launcher:

```text
run_logs/launch_hgrag_limit100_20260429.sh
```

Combined launcher used for the Prop-side rows:

```text
run_logs/launch_prop_neocorr_daec_limit100_20260429.sh
```

## Purpose

This protocol compares four rows on the first 100 questions of each dataset:

```text
PropRAG
PropRAG + DAEC
PropRAG + DAEC-L1
NEOCORRAG
HGRAG
```

The main goal is not to claim a perfectly identical implementation substrate
across systems. The goal is to separate two comparison regimes:

1. **Same-pool composition comparison.**
   PropRAG, PropRAG + DAEC, and PropRAG + DAEC-L1 all consume the same fixed
   PropRAG top-100 candidate pool.

2. **Native-system comparison.**
   NEOCORRAG and HGRAG are run through native or native-shaped indexing,
   retrieval, and QA pipelines, while aligning reader, embedding endpoint,
   datasets, query limit, and output metrics as much as practical.

The NEOCORRAG and HGRAG rows must therefore be labeled as native-system
baselines, not as exact same-pool selector baselines.

## Datasets

Datasets:

```text
2wikimultihopqa
hotpotqa
musique
```

Limit:

```text
limit = 100
```

The limit applies to QA queries. For NEOCORRAG native runs, this does not limit
corpus indexing. NEOCORRAG may still build or load corpus-level OpenIE, graph,
chunk embeddings, entity embeddings, and fact embeddings before answering the
first 100 questions.

## Metrics

Report these metrics for every row:

```text
Recall@5
Recall@20
EM
F1
```

Metric sources:

- PropRAG row: `overall_recomputed` from the HippoRAG external-pool evaluator.
- PropRAG + DAEC row: `setwise_selector_qa.selector_retrieval_metrics` and
  `setwise_selector_qa.selector_EM/F1`.
- PropRAG + DAEC-L1 row: same fields as the DAEC row.
- NEOCORRAG row: `overall_retrieval_result` and `overall_qa_results` saved by
  `scripts/run_neocorrag_aligned.py`.

## Runtime Alignment

Shared runtime settings where applicable:

```text
limit = 100
qa_top_k = 5
embedding endpoint = http://localhost:8019/v1/embeddings
embedding model = nvidia/NV-Embed-v2
reader API model = qwen3-8b-train
max_new_tokens = 2048
```

Dataset-to-port mapping:

```text
2wikimultihopqa -> http://localhost:8041/v1
hotpotqa        -> http://localhost:8042/v1
musique         -> http://localhost:8043/v1
```

The Prop-side evaluator is launched through:

```text
scripts/eval_causal_qwen3.py
```

The NEOCORRAG native wrapper is:

```text
scripts/run_neocorrag_aligned.py
```

The HGRAG native-shaped wrapper is:

```text
scripts/run_hgrag_aligned.py
```

## PropRAG Same-Pool Rows

All three Prop-side rows use the fixed exported PropRAG top-100 pool:

```text
run_logs/proprag_pool_exports_full1000_20260424/{dataset}_pool100.json
```

The external pool is consumed with:

```text
--external_pool_json <pool_json>
--external_pool_source_name proprag_clean_nothink_top100
--external_pool_strict_questions true
```

This makes the Prop-side rows valid same-pool comparisons.

### PropRAG

This row selects directly from the PropRAG top-100 pool with no setwise selector:

```text
--setwise_selector none
```

Interpretation:

```text
PropRAG top-100 pool -> top-k reader context
```

### PropRAG + DAEC

This row applies the original DAEC selector to the same PropRAG top-100 pool:

```text
--setwise_selector dtc_embed
--setwise_pool_k 100
--setwise_reserve_top_m 0
--dtc_rank_weight 0.2
--dtc_include_satisfiable_by true
--dtc_repairable_filter_enabled true
--dtc_satisfiable_by_policy binding_override
--dtc_enable_dependency_binding true
--dtc_binding_max_candidates 4
--dtc_binding_entity_hit_required true
```

Interpretation:

```text
PropRAG top-100 pool -> original DAEC fixed-pool composition -> reader context
```

### PropRAG + DAEC-L1

This row applies DAEC-L1 noisy-OR projection to the same PropRAG top-100 pool:

```text
--setwise_selector daec_noisyor
--setwise_pool_k 100
--dtc_decomposition_mode llm
--dtc_binding_max_candidates 5
```

Interpretation:

```text
PropRAG top-100 pool -> DAEC-L1 fixed-pool projection -> reader context
```

## NEOCORRAG Native Row

NEOCORRAG is run natively rather than forced to consume the PropRAG pool.

Command shape:

```text
CUDA_VISIBLE_DEVICES=<device> .venv-hipporag/bin/python scripts/run_neocorrag_aligned.py \
  --dataset <dataset> \
  --limit 100 \
  --neocorrag_root /mnt/nvme/code/NeocorRAG \
  --output_json run_logs/prop_neocorr_daec_limit100_20260429/<dataset>_neocorrag.json \
  --llm_name qwen3-8b-train \
  --llm_base_url http://localhost:<port>/v1 \
  --graph_llm_name qwen3-8b-train \
  --graph_llm_base_url http://localhost:<port>/v1 \
  --embedding_name nvidia/NV-Embed-v2 \
  --embedding_base_url http://localhost:8019/v1/embeddings \
  --reretrieval_llm_name /mnt/nvme/Qwen2.5-7B-Instruct \
  --reretrieval_embedding_name nvidia/NV-Embed-v2 \
  --generation_mode greedy \
  --k 1 \
  --qa_top_k 5 \
  --retrieval_top_k 100 \
  --embedding_batch_size 4 \
  --max_new_tokens 2048
```

NEOCORRAG output fields saved by the wrapper:

```text
overall_retrieval_result
overall_qa_results
records
rerank_log_preview
```

Interpretation:

```text
NEOCORRAG native index/retrieval/Reretrieval QA -> aligned metric JSON
```

The row is comparable as a native system under aligned endpoints and metrics. It
is not comparable as a same-pool reranker or selector.

Accepted NEOCORRAG rows must satisfy the wrapper config checks below:

```text
config.retrieval_top_k = 100
config.reretrieval_embedding_name = nvidia/NV-Embed-v2
config.no_think = true
```

Earlier diagnostic runs with `retrieval_top_k=300`, with
`reretrieval_embedding_name=bge-large-en-v1.5`, or with Qwen context-overflow
reader errors are excluded from the comparison table.

## HGRAG Native-Shaped Row

HGRAG is cloned under:

```text
/mnt/nvme/code/HGRAG
```

The wrapper preserves the HGRAG entity-hypergraph retrieval shape:

```text
query/corpus entity extraction
  -> entity-to-entity dense retrieval
  -> query-to-document dense retrieval
  -> entity-document hypergraph diffusion
  -> top-k document context
  -> QA
```

The adaptation replaces HGRAG's default local-model assumptions with the same
local endpoints used by the other rows:

```text
Qwen3-8B chat API     -> NER and QA
NV-Embed API endpoint -> entity/entity and query/document embeddings
```

Command shape:

```text
.venv-hipporag/bin/python scripts/run_hgrag_aligned.py \
  --dataset <dataset> \
  --limit 100 \
  --hgrag_root /mnt/nvme/code/HGRAG \
  --output_json run_logs/hgrag_aligned_limit100_20260429/<dataset>_hgrag.json \
  --llm_name qwen3-8b-train \
  --llm_base_url http://localhost:<port>/v1 \
  --embedding_name nvidia/NV-Embed-v2 \
  --embedding_base_url http://localhost:8019/v1/embeddings \
  --qa_top_k 5 \
  --recall_top_k 20 \
  --e2e_top_k 20 \
  --ent_topk 1 \
  --beta 0.5 \
  --step 2 \
  --hgraph_device cpu
```

Qwen3 `/no_think` is mandatory for both HGRAG NER and HGRAG QA. The wrapper
injects `/no_think` into user messages and strips any accidental `<think>` blocks
before parsing JSON entities or final answers.

The HGRAG row is comparable as a native-shaped HGRAG system under aligned local
model endpoints. It is not a same PropRAG-pool selector comparison.

## NEOCORRAG Wrapper Compatibility Patches

The wrapper intentionally avoids modifying the sibling NEOCORRAG repository.
Compatibility fixes are applied at runtime in:

```text
scripts/run_neocorrag_aligned.py
```

### Embedding Class Patch

NEOCORRAG's local NVEmbedV2 class in this checkout hard-codes a lab-local model
path. The wrapper replaces the embedding class with NEOCORRAG's
OpenAI-compatible embedding client so the run uses the same NV-Embed endpoint as
the Prop-side rows.

The wrapper also strips a trailing `/embeddings` path from the configured base
URL because the OpenAI client appends `/embeddings` itself.

### NER Entity Normalization Patch

Observed issue:

```text
NER failed for chunk ... Error: unhashable type: 'dict'
```

Cause:

Qwen-style OpenIE responses sometimes return entities as objects such as:

```json
{"name": "...", "type": "..."}
```

Native NEOCORRAG deduplicates with `dict.fromkeys(extracted_entities)`, which
cannot hash dict elements. The wrapper normalizes string, dict, and nested list
entities into a deduplicated list of strings.

Decision:

This is a parser compatibility fix, not a method change. It preserves valid
entities instead of dropping chunks after retries.

### DSPyFilter Template Patch

Observed issue:

```text
'DSPyFilter' object has no attribute 'message_template'
```

Cause:

The `DSPyFilter` class uses `self.message_template` in `llm_call`, but the
initializer in this checkout does not create the attribute.

Decision:

The wrapper initializes the intended default prompt state at runtime. Without
this fix, filter calls silently fall back after exceptions.

### Reretrieval No-Path Fallback Patch

Observed issue:

```text
TypeError: cannot unpack non-iterable NoneType object
```

Cause:

`get_graph_index_with_dynamic_search()` returns `None` when no valid paths are
found, while `Reretrieval_prediction()` expects:

```text
all_tries, all_entities = ...
```

Decision:

The wrapper converts no-path cases to:

```text
([], [])
```

This lets NEOCORRAG's existing empty-prediction/default-QA fallback handle the
case instead of crashing after retrieval has completed.

### Qwen3 `/no_think` Patch

Qwen3 `/no_think` is mandatory for NEOCORRAG OpenIE/query NER and QA calls. The
wrapper injects `/no_think` into the first user message and strips accidental
`<think>...</think>` blocks before downstream parsing or answer scoring.

### QA Prompt Budget Patch

Observed issue:

```text
Native Reretrieval QA can pass many retrieved documents and path strings into
the final reader prompt, exceeding the local Qwen3 context window.
```

Decision:

The wrapper keeps NEOCORRAG's native Reretrieval path signal but caps final QA
inputs before calling the reader:

```text
qa_top_k = 5
doc_total_chars = 9000
path_total_chars = 4500
doc_item_chars = 2200
path_item_chars = 900
```

Runs that hit context-overflow API errors are diagnostic only and are not valid
comparison rows.

## Error Handling Policy

### NER Parser Failures

NER parser failures are treated as hard protocol issues when they systematically
drop chunks. The `unhashable type: 'dict'` failure was fixed before accepting
NEOCORRAG results.

### Triple Parse Failures

Malformed triple outputs such as incomplete string literals are tracked but not
patched unless the rate becomes large.

Current rule:

```text
< 1% malformed triples:
  continue the run and report the count

1% to 5% malformed triples:
  inspect raw responses and consider parser hardening plus rerun

> 5% malformed triples:
  stop and treat the NEOCORRAG row as invalid until extractor stability is fixed
```

For the accepted 2Wiki NEOCORRAG `ret100 + NV-Embed Reretrieval` run, observed
malformed triple outputs were:

```text
6 parse failures
```

This is recorded as extraction noise, not a fatal protocol failure.

### Reretrieval No-Path Failures

No-path Reretrieval is not a fatal retrieval failure. It means the retrieved
subgraph has no valid constrained path for that query. The accepted behavior is
empty path prediction plus default QA fallback.

## Execution Semantics

Prop-side rows can run in parallel by dataset because they consume fixed pool
JSON files and write independent outputs.

NEOCORRAG is run sequentially by default:

```text
2wikimultihopqa -> hotpotqa -> musique
```

Reason:

NEOCORRAG may load local constrained-decoding and reretrieval models, and all
runs share the NV-Embed endpoint. Parallel NEOCORRAG runs are possible only if
the launcher prevents duplicate dataset execution and GPU/embedding-endpoint
contention is explicitly managed.

## Current Completed Prop-Side Numbers

These are the completed Prop-side limit-100 results from the current run.

| Dataset | Method | Recall@5 | Recall@20 | EM | F1 |
|---|---:|---:|---:|---:|---:|
| 2Wiki | PropRAG | 0.9350 | 0.9625 | 0.5800 | 0.6318 |
| 2Wiki | PropRAG + DAEC | 0.9525 | 0.9675 | 0.5800 | 0.6435 |
| 2Wiki | PropRAG + DAEC-L1 | 0.9375 | 0.9650 | 0.5700 | 0.6164 |
| HotpotQA | PropRAG | 0.9300 | 0.9950 | 0.5700 | 0.6912 |
| HotpotQA | PropRAG + DAEC | 0.9500 | 0.9950 | 0.5700 | 0.6912 |
| HotpotQA | PropRAG + DAEC-L1 | 0.9500 | 0.9950 | 0.6000 | 0.7079 |
| MuSiQue | PropRAG | 0.6775 | 0.8900 | 0.3800 | 0.4374 |
| MuSiQue | PropRAG + DAEC | 0.7175 | 0.8983 | 0.3900 | 0.4660 |
| MuSiQue | PropRAG + DAEC-L1 | 0.6600 | 0.9058 | 0.3700 | 0.4202 |

Interpretation of this table should stay inside the same-pool PropRAG regime
until the NEOCORRAG rows complete.

## Current Native-System Numbers

These rows are native-system baselines under aligned local endpoints. They are
not same-pool selector comparisons against PropRAG.

| Dataset | Method | Recall@5 | Recall@20 | EM | F1 | Diagnostics |
|---|---:|---:|---:|---:|---:|---|
| 2Wiki | HGRAG | 0.7650 | 0.8200 | 0.4600 | 0.4837 | qner_empty=0, cner_empty=3, qa_empty=0 |
| HotpotQA | HGRAG | 0.9450 | 0.9900 | 0.5800 | 0.7117 | qner_empty=0, cner_empty=5, qa_empty=0 |
| MuSiQue | HGRAG | 0.6733 | 0.8000 | 0.3600 | 0.4236 | qner_empty=0, cner_empty=10, qa_empty=0 |
| 2Wiki | NEOCORRAG | 0.8325 | 0.9175 | 0.5300 | 0.5655 | ret100, NV-Embed Reretrieval, triple_parse_errors=6, no_path_fallback=13 |
| HotpotQA | NEOCORRAG | pending | pending | pending | pending | running after 2Wiki |
| MuSiQue | NEOCORRAG | pending | pending | pending | pending | queued after HotpotQA |

## Reporting Rules

Final report must include:

1. A main table with Recall@5, Recall@20, EM, and F1.
2. A protocol note that Prop-side rows are same PropRAG-pool comparisons.
3. A protocol note that NEOCORRAG is native-system aligned, not same-pool.
4. NEOCORRAG parser/fallback compatibility patches.
5. HGRAG endpoint adaptation and `/no_think` enforcement.
6. Counts of NEOCORRAG NER failures and malformed triple outputs.
7. HGRAG QNER/CNER empty-entity counts and QA empty-answer count.
8. Any native-system row that fails before JSON generation should be marked failed,
   not missing.

## Decision Boundaries

Use these boundaries when interpreting the smoke:

```text
PropRAG + DAEC > PropRAG:
  evidence that original DAEC composition helps on the PropRAG pool.

PropRAG + DAEC-L1 > PropRAG:
  evidence that DAEC-L1 projection helps on the PropRAG pool.

PropRAG + DAEC or DAEC-L1 tied with PropRAG:
  selector details are not clearly adding over the PropRAG candidate order.

NEOCORRAG > Prop-side rows:
  native NEOCORRAG system is stronger under this aligned runtime, but not a
  same-pool selector conclusion.

NEOCORRAG < Prop-side rows:
  evidence against the native NEOCORRAG pipeline in this local setting, subject
  to wrapper patch notes and extraction-failure diagnostics.

HGRAG > Prop-side rows:
  native-shaped HGRAG hypergraph retrieval is stronger under aligned local
  endpoints, but not a same-pool selector conclusion.

HGRAG < Prop-side rows:
  evidence against the adapted HGRAG pipeline in this local setting, subject to
  endpoint adaptation and entity-extraction diagnostics.
```

Do not use this smoke alone to claim full benchmark superiority. It is a
limit-100 protocol sanity check and candidate-routing decision point.
