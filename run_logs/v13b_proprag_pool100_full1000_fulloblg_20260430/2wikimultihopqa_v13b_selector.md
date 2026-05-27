# Obligation-Closed STO Local PPR

This is a diagnostic selector over existing candidate documents. It computes local PPR from query-seed facts over the STO fact graph and projects that mass onto closed STO units.

## Config

| parameter | value |
| --- | ---: |
| `max_queries` | 1000 |
| `evidence_set_size` | 5 |
| `max_path_edges` | 3 |
| `max_endpoint_degree` | 30 |
| `alpha` | 0.2 |
| `residual_epsilon` | 1e-06 |
| `query_indices` | [] |
| `head_traces_cache_paths` | {} |
| `query_obligation_cache_path` | /mnt/nvme/code/HippoRAG/run_logs/v13b_proprag_pool100_full1000_fulloblg_20260430/2wikimultihopqa_query_obligation_cache_full1000.json |
| `allow_lexical_fallback` | False |
| `allow_partial_obligation_grounding` | False |
| `allow_support_obligation_grounding` | True |
| `allow_source_span_obligation_grounding` | True |
| `allow_program_evidence_assembler` | True |
| `use_typed_retrieval_critical_obligations` | True |

## Metrics

| dataset | rows | R@5 | all-gold@5 | exact full | effective full | program feasible | changed | gains | losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | 1000 | 0.9067 | 0.7800 | 89 | 164 | 156 | 79 | 8 | 0 |

## Method Boundary

- Uses local PPR over the STO fact graph, not global HippoRAG v2 diffusion.
- Uses no gold labels, no QA outcomes, no SFB outputs, no answer-type rules, and no weighted late fusion.
- Still uses the source report's candidate document universe, so this is not yet a standalone retriever.
