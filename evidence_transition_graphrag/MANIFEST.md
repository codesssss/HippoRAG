# Portable Overlay Manifest

| Included path | Installed target path | Required |
| --- | --- | --- |
| `evidence_transition_graphrag/` | `evidence_transition_graphrag/` | yes |
| `portable_overlay/run_query_grounded_sto_fresh_e2e.py` | `run_query_grounded_sto_fresh_e2e.py` | yes |
| `portable_overlay/evidence_transition_top5_qa_adapter.py` | `evidence_transition_top5_qa_adapter.py` | yes |
| `portable_overlay/source_authorized_vocab_strict_retrieval/` | `source_authorized_vocab_strict_retrieval/` | yes |
| `portable_overlay/src/agsto/` | `src/agsto/` | yes, merged without deleting `src/hipporag` |
| `portable_overlay/agsto/` | `agsto/` | yes |

The overlay assumes the target already has the base HippoRAG implementation
under `src/hipporag/`.
