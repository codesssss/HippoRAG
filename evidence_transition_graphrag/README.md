# Evidence Transition GraphRAG

`evidence_transition_graphrag` is the current clean end-to-end GraphRAG line.

## Method Boundary

| Stage | Implementation |
| --- | --- |
| Indexing | fresh passage embeddings + fresh OpenIE |
| Corpus graph | global STO evidence graph |
| Query graph | query-local STO graph induction |
| Retrieval | graph-native top-5 reader context selection |
| QA | fixed top-5 passage reader |

## Clean Contract

| Constraint | Status |
| --- | --- |
| No weighted score fusion | yes |
| No dataset routing | yes |
| No external baseline frontier | yes |
| No LLM query compiler | yes |
| No SFB/support-fusion source report | yes |
| No V13B source report replay | yes |

## Compatibility

Older reports may still expose `query_grounded_sto_graphrag` or
`query_grounded_sto_doc_indices_top5`. Those names are kept as compatibility
aliases only. The public method name for new reports is
`evidence_transition_graphrag`.

## Portability

This package is not just the four top-level Python files. For use in another
HippoRAG-style checkout, install the bundled overlay:

```bash
python evidence_transition_graphrag/install_overlay.py --target /path/to/HippoRAG
```

See `PORTABILITY.md` for the included files and target-repo assumptions.
