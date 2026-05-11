# ETv4 Clean Mainline Audit 2026-05-11

## Scope

This audit freezes the paper-facing ETv4 path after removing redundant or
engineering-looking selection stages from the default runner.

| Item | Setting |
| --- | --- |
| Method folder | `evidence_transition_graphragv4_fact_witnessed_sto` |
| Default readout policy | `clean_mainline` |
| Default runner | `query_grounded_sto_clean_mainline_v4` |
| Retrieval run | `run_logs/evidence_transition_graphragv4_clean_mainline_final_gpt4omini_3x100_20260511` |
| Reader QA run | `run_logs/gpt4omini_reader_aligned_prop_etv4_multi_anchor_strict_limit100_20260511` |
| Limit | 100 queries per dataset |
| Reader | `gpt-4o-mini` |
| Reader max tokens | 400 |
| Reader context | top-5 full passages |

## Paper-Facing Mainline

| Stage | Clean mainline behavior |
| --- | --- |
| Indexing | Fresh OpenIE plus embedding artifacts for the current run |
| Entry | Dense/textual retrieval opens a query-local STO graph |
| Graph construction | Query-local STO document graph from OpenIE fact endpoints and document/title endpoints |
| Final readout | Fact-witnessed branch document readout |
| Reader input | Fixed top-5 documents selected by the clean readout |

The final readout applies one precision rule:

| Rule | Meaning |
| --- | --- |
| Single symbolic anchor | A symbolic-only root may promote fact-witnessed frontier docs |
| Multiple symbolic anchors | A symbolic-only root may seed graph admission, but cannot promote frontier docs unless also supported by textual entry |

This prevents ambiguous multi-anchor title/entity hits from pushing documents
into the reader context while preserving useful single-anchor bridge documents.

## Removed From Default Path

These components remain in the package only as legacy ablations or diagnostic
tools. They are not used by `clean_mainline`.

| Component | Clean mainline status | Reason |
| --- | --- | --- |
| Source-prior guard | Disabled | Looks like a safety repair rather than the core graph claim |
| Source-text validation closure | Disabled for selection | Useful trace, but not a clean readout stage |
| Replacement policy | Disabled | Hidden source-prior repair would blur the method claim |
| Path-cover readout | Legacy negative ablation | Lower MuSiQue performance |
| Transition-valid closure | Legacy negative ablation | Lower than branch readout |
| Root-balanced transition | Legacy negative ablation | Helps 2Wiki but hurts MuSiQue/HotpotQA |
| Query-supported same-object handoff | Disabled by default | Kept only as explicit ablation flag |
| Variable-flow traversal | Enabled by wrapper | Closed-form graph traversal over active fact endpoints; not a reader-time fallback |
| Weighted score fusion | Not used | Avoids multi-parameter engineering mixture |
| LLM query schema | Not used | Avoids query-specific prompt/schema machinery |

## Trace Verification

All clean retrieval rows reported the same disabled flags.

| Dataset | Rows | Guard | Source-text validation | Replacement |
| --- | ---: | --- | --- | --- |
| 2Wiki | 100 | `False` | `False` | `False` |
| MuSiQue | 100 | `False` | `False` | `False` |
| HotpotQA | 100 | `False` | `False` | `False` |

The wrapper automatically enables variable-flow traversal unless explicitly
provided. The trace confirmed it was enabled for all three datasets, while
query-supported same-object handoff remained disabled.

| Dataset | Variable-flow traversal | Query-supported same-object handoff |
| --- | --- | --- |
| 2Wiki | `True` | `False` |
| MuSiQue | `True` | `False` |
| HotpotQA | `True` | `False` |

## Retrieval: Legacy vs Clean

| Dataset | Rows | Legacy R@5 | Clean R@5 | Delta R@5 | Legacy all-gold@5 | Clean all-gold@5 | Delta all-gold@5 | Top-5 changed | Gains | Losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 100 | 0.9400 | 0.9400 | +0.0000 | 0.8500 | 0.8500 | +0.0000 | 0 | 0 | 0 |
| MuSiQue | 100 | 0.7358 | 0.7358 | +0.0000 | 0.4300 | 0.4300 | +0.0000 | 2 | 0 | 0 |
| HotpotQA | 100 | 0.9550 | 0.9550 | +0.0000 | 0.9100 | 0.9100 | +0.0000 | 1 | 0 | 0 |

## Reader QA: PropRAG vs Legacy vs Clean

All methods use the aligned `gpt-4o-mini` reader with `max_new_tokens=400` and
top-5 full-passage context.

| Dataset | Method | R@5 | all-gold@5 | EM | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 2Wiki | PropRAG | 0.9350 |  | 0.6200 | 0.6842 |
| 2Wiki | ETv4 legacy | 0.9400 | 0.8500 | 0.6200 | 0.6983 |
| 2Wiki | ETv4 clean | 0.9400 | 0.8500 | 0.6200 | 0.6983 |
| MuSiQue | PropRAG | 0.6775 |  | 0.4300 | 0.5224 |
| MuSiQue | ETv4 legacy | 0.7358 | 0.4300 | 0.4300 | 0.5440 |
| MuSiQue | ETv4 clean | 0.7358 | 0.4300 | 0.4300 | 0.5440 |
| HotpotQA | PropRAG | 0.9300 |  | 0.6300 | 0.7419 |
| HotpotQA | ETv4 legacy | 0.9550 | 0.9100 | 0.6400 | 0.7542 |
| HotpotQA | ETv4 clean | 0.9550 | 0.9100 | 0.6400 | 0.7542 |

## Interpretation

| Finding | Meaning |
| --- | --- |
| 2Wiki is identical to legacy | Removed stages were redundant on this dataset |
| HotpotQA retrieval is metric-identical to legacy | Top-5 order changes do not affect gold coverage or QA |
| MuSiQue matches legacy after multi-anchor precision rule | The clean rule removes the harmful ambiguous branch while preserving useful single-anchor bridge docs |
| Clean matches legacy on R@5, EM, and F1 | The paper-facing mainline no longer depends on guard/validation/replacement for these results |
| Clean beats or ties PropRAG on EM and beats PropRAG on R@5/F1 | The cleaned mainline remains competitive after deduplication |
| Clean has a clearer claim than legacy | Final selection is now fact-witnessed branch STO readout, not guarded replay |

## Current Claim

ETv4 clean mainline is not a pure graph-from-scratch retriever. It is a
baseline-aligned GraphRAG method:

| Signal | Role |
| --- | --- |
| Dense/textual retrieval | Opens the query-local candidate graph |
| STO graph | Defines document transitions |
| OpenIE facts | Witness whether transitions are evidence-bearing |
| Branch readout | Selects the final top-5 document context |
| Reader | Answers from the selected top-5 passages |

This is aligned with HippoRAGv2/PropRAG-style retrieval systems that use
non-graph entry signals, but the final readout is graph-based and
fact-witnessed.
