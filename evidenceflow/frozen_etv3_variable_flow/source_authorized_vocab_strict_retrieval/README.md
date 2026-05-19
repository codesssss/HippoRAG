# Source-Text Certificate GraphRAG

This folder is the standalone home for the source-authorized retrieval line.
The default runner is now the canonical no-role/no-hint method:
`source_text_certificate_graphrag`.

It exists because the report variant name
`graph_native_source_authorized_evidence_set_search` was overloaded.  The
high-score reports do **not** correspond to the plain dense-entry extraction in
`source_authorized_evidence_set_search_exact`.  They correspond to the
source-text evidence-set MCT v18 branch.

That historical branch remains useful as an ablation target, but it is not the
default paper-facing method because it uses hand-written role and hint
machinery.

## High-Score Contract

| item | value |
| --- | --- |
| Public report alias | `graph_native_source_authorized_evidence_set_search` |
| Legacy implementation branch | `graph_native_source_text_evidence_set_mct_v18` |
| Method object | `source_text_authorized_evidence_set` |
| Candidate entrance | `method_internal_evidence_frontier` |
| Composition | `source_text_query_chain_certified_evidence_set_mct_to_reader_facing_context` |

## Canonical Method Boundary

The canonical method constructs a method-internal source-text evidence frontier,
then selects a reader-facing evidence set whose insertions are authorized by
source-text certificates.

It explicitly excludes weighted score fusion, dataset routing, external
baseline frontiers, proposition support evidence, distillation, learned
rerankers, role/focal overrides, compact-reader dual outputs, relation-role
certificates, surface-role certificates, child-parent hints, broad-location
blocklists, and hand-written role vocabularies.

## Demand-Aware Completion Candidate

The tail-only canonical runner remains available as a conservative safe-mode
ablation.  The current candidate for upgrading it into a main method is:

`source_certified_evidence_set_completion`

It keeps source-text certificates as the admission requirement, but no longer
treats admission alone as the method.  The reader context is viewed as a fixed
budget evidence set.  Starting from the dense/local top-k, the method builds
retrieval-grounded evidence demands from selected source passages: if a
selected source has a canonical source-text certificate to a target that is
missing from the current reader set, that target represents a missing evidence
demand.  A candidate can enter only when it covers one of these active missing
demands, and the authorizing source remains in the reader set.

This path is implemented in:

| File | Role |
| --- | --- |
| `evidence_completion.py` | Demand-aware source-certified evidence-set completion. |
| `pipeline.py` | Exposes `source_certified_evidence_completion_pipeline_retrieve`. |
| `evaluate_report.py` | Exposes `--runner completion` for report replay. |

The selector is deliberately not a weighted utility function.  It applies hard
admission and completion conditions, considers eligible targets in the original
candidate order, preserves protected query/title anchors, and returns no-op
when no active missing demand can be covered.

## Query-Conditioned Active Graph Candidate

The active trust-region branch tests a stricter method change: do not expose all
source-certified edges to repair.  First filter the certificate graph into a
query-conditioned active graph, then allow a small trust-region repair only over
active edges.

| File | Role |
| --- | --- |
| `query_conditioning.py` | Extracts query surface signatures without LLM decomposition, role vocab, or propositions. |
| `active_certificate_graph.py` | Builds `G_cert(q)` by activating certificate edges before repair and suppressing same-source same-slot siblings. |
| `trust_region_repair.py` | Repairs reader top-k only over active certificate edges. |
| `pipeline.py` | Exposes `source_active_trust_region_pipeline_retrieve`. |
| `evaluate_report.py` | Exposes `--runner active_trust` for report replay. |

Limit=100 smoke on 2026-05-07 showed that active-edge filtering reduces harm
relative to unrestricted completion, but it is not yet a replacement for the
canonical clean runner:

| Dataset | active_trust R@5 | changed | gains | losses |
| --- | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | 0.9250 | 43 | 21 | 3 |
| musique | 0.7108 | 52 | 10 | 6 |
| hotpotqa | 0.9650 | 64 | 2 | 4 |

The main remaining issue is activation recall: pure query-surface activation
misses relation paraphrases such as author/written or spouse/married without
reintroducing hand-written role vocabularies.

## Current Status

This folder pins the correct contract, provides reproduction utilities, and now
contains a canonical end-to-end GraphRAG extraction of the source-authorized
line:

`dense/local candidate universe -> source-authorized candidate expansion -> source-text certificate graph -> source frontier -> conservative evidence-set search -> reader QA export`

The core graph primitive turns source-text authorization into an explicit graph
object:

| Component | Meaning |
| --- | --- |
| `EvidenceNode` | One candidate passage with title, text, and OpenIE triples. |
| `EvidenceCertificate` | One binary source -> target authorization edge. |
| `SourceTextCertificateGraph` | Query-local evidence graph whose edges are certificates, not scores. |
| `build_source_text_certificate_graph` | Builds certificate edges from source text, titles, and OpenIE endpoints. The default policy does not use query-visible role terms. |
| `SourceTextCandidateExpansion` | Source-bound endpoint text induction that can add local candidates before graph construction. |
| `SourceTextFrontier` | Method-internal frontier built from initial head, query-visible title mentions, and certificate closure. |
| `source_authorized_vocab_strict_pipeline_retrieve` | End-to-end runner for the paper-facing clean method. |
| `normalize.py` | Local deterministic text/title normalization, kept inside this package so the high-score line does not depend on the discarded low-score exact branch. |

The canonical certificate types are:

| Certificate | What It Means |
| --- | --- |
| `title_alias` | Source passage text explicitly mentions the target title. |
| `endpoint_title` | A source OpenIE endpoint matches the target title. |
| `source_title_endpoint` | The source title is one OpenIE endpoint and the opposite endpoint matches the target title. |

The legacy opt-in policy `legacy_role_hints` also exposes `relation_role`,
`surface_role`, and `directional_child_to_parent` certificates.  Those are kept
for historical reproduction and ablation only; they are not part of the
canonical clean method.

The package runners are:

| File | Role |
| --- | --- |
| `candidate_expansion.py` | Adds source-authorized candidates from seed-source endpoints when the target text certifies the endpoint. |
| `frontier.py` | Builds source frontier/search exposure without importing any external baseline frontier. |
| `retriever.py` | Selects top-k reader evidence by entry document, certificate reachability, then original candidate order. |
| `evidence_completion.py` | Completes a fixed-budget reader evidence set using active missing source-certified demands. |
| `pipeline.py` | Runs the full clean GraphRAG chain in one call. |
| `evaluate_report.py` | Replays the standalone runner from a per-query report and an OpenIE file. |
| `export_qa_transition_report.py` | Exports fixed top5 evidence into the reader QA evaluator format. |

This runner is intentionally not a score-fusion method.  Dense or HippoRAGv2
output may provide the local candidate universe, but final top-k movement is
authorized only by certificate graph reachability.

## Previous Vocab-Strict Checkpoint: 2026-05-06

The following numbers were produced before the canonical no-role/no-hint default
was introduced.  They are useful as a legacy vocab-strict reference, not as the
final clean-method claim.  Protocol: limit=100, `candidate_pool_k=200`,
runner=`pipeline`.

| Dataset | HippoRAGv2 R@5 | PropRAG-local R@5 | Historical Source-Authorized R@5 | Current Clean R@5 |
| --- | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | 0.8525 | 0.9150 | 0.9600 | 0.9350 |
| musique | 0.6950 | 0.7283 | 0.7358 | 0.7408 |
| hotpotqa | 0.9750 | 0.9500 | 0.9700 | 0.9800 |

Reader QA with local Qwen3-8B no-think on the same fixed top5 set, using
source-before-certificate-target context order:

| Dataset | R@5 | EM | F1 |
| --- | ---: | ---: | ---: |
| 2wikimultihopqa | 0.9350 | 0.6000 | 0.6577 |
| musique | 0.7408 | 0.4400 | 0.5140 |
| hotpotqa | 0.9800 | 0.6200 | 0.7338 |

Against PropRAG-local under the same limit=100 Qwen3-8B no-think protocol:

| Dataset | PropRAG R@5 | Current R@5 | PropRAG EM | Current EM | PropRAG F1 | Current F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | 0.9150 | 0.9350 | 0.5500 | 0.6000 | 0.6263 | 0.6577 |
| musique | 0.7283 | 0.7408 | 0.3700 | 0.4400 | 0.4694 | 0.5140 |
| hotpotqa | 0.9500 | 0.9800 | 0.5900 | 0.6200 | 0.7154 | 0.7338 |

The remaining gap is not a candidate-expansion parameter problem.  A paired
2Wiki analysis shows most historical-only wins already have endpoint
certificates, but directly promoting `source_title_endpoint` as reader evidence
hurts MuSiQue badly.  Therefore `source_title_endpoint` is kept as search
exposure/frontier semantics, not as a standalone reader replacement certificate.

The old `source_authorized_evidence_set_search_exact` folder is a low-score
plain dense-entry branch and should not be used as the paper-facing method.

## Reproduction Checkpoint: 2026-05-06

The high-score oracle is the historical report family:

`outputs_source_authorized_vocab_strict_limit100_20260505`

Directly running the current dirty `compare_graph_retrievers.py` with
`graph_native_source_text_evidence_set_mct_v18` matches the contract but does
not exactly match the oracle scores.  This means the method target is now
identified, but the current monolithic worktree is not a clean oracle.

| Dataset | HippoRAGv2 R@5 | Oracle R@5 | Direct v18 R@5 | Delta |
| --- | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | 0.8525 | 0.9600 | 0.9575 | -0.0025 |
| musique | 0.6950 | 0.7458 | 0.7350 | -0.0108 |
| hotpotqa | 0.9750 | 0.9700 | 0.9550 | -0.0150 |

The gap report is written to:

`run_logs/source_authorized_vocab_strict_repro_gap_20260506.md`

Do not use the current public alias branch in `compare_graph_retrievers.py` as
the extraction oracle.  In the current dirty file, that alias name routes to a
different source-authorized readout path than the historical high-score oracle.
Use the contract-checked historical reports plus direct v18 reproduction as the
migration target, and require parity against the oracle before claiming full
extraction.
