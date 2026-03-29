# Claim Ledger

Last updated: 2026-03-29

Status values:
- `supported`
- `partial`
- `unsupported`
- `contradicted`

| ID | Claim | Status | Current Evidence | Gap | Next Action |
|---|---|---|---|---|---|
| C1 | Oracle evidence selection from a larger pool yields substantial headroom over the baseline reader input. | supported | `2Wiki-1000`: EM `0.436 -> 0.560` at `K=100`; `MuSiQue-1000`: EM `0.257 -> 0.436`; `HotpotQA-1000`: EM `0.563 -> 0.645`. See `04_result_registry.md` and the cross-dataset summary. | None for the ceiling claim. | Use this as the main ceiling table in the paper. |
| C2 | Hard multi-hop queries require much deeper support pools than easy queries. | supported | `2Wiki-1000`: `2-doc median depth = 3`, `4-doc median depth = 64`; `MuSiQue-1000`: `2-doc = 5`, `3-doc = 18`, `4-doc = 55`. | Need only visualization, not more evidence. | Turn this into one support-depth figure or table. |
| C3 | Larger candidate pools help hard queries more than easy queries. | supported | `2Wiki-1000`: `4-doc FS@20 = 0.0936`, `FS@100 = 0.2085` while `2-doc` is already `0.8327 -> 0.9190`; `MuSiQue-1000`: `4-doc FS@20 = 0.0904`, `FS@100 = 0.4096`; `HotpotQA` saturates early because it is all `2-doc`. | Need only a clean cross-dataset narrative. | Present Hotpot as the shallow contrast case. |
| C4 | The bottleneck is evidence composition rather than simple top-5 reordering. | supported | `2Wiki-1000`: reorder@20 `+0.062` vs select@100 `+0.124`; `MuSiQue-1000`: reorder@20 `+0.106` vs select@100 `+0.179`; `HotpotQA-1000`: reorder@20 `+0.068` vs select@100 `+0.082`. | Could still benefit from adding CE reranker into the same table. | Build one compact comparison table: baseline vs CE rerank vs oracle reorder vs oracle select. |
| C5 | Bridge docs are systematically harder to retrieve than anchor docs. | supported | `2Wiki bridge analysis`: anchor `Recall@20 = 0.9995`, bridge `Recall@20 = 0.5997`; anchor median depth `1.0`, bridge median depth `6.0`; bridge mean depth `29.3`, `P90 = 100.6`; `71.15%` of partial top-5 failures are bridgeable. | Mechanism is shown on `2Wiki`, but not yet on `MuSiQue`. | Optional: add a lighter bridge-style analysis on `MuSiQue` only if needed. |
| C6 | The structural gap generalizes to another hard multi-hop dataset. | supported | `MuSiQue-1000` reproduces the same depth hierarchy and larger ceiling gains than `HotpotQA`; `2-doc/3-doc/4-doc` support depths increase monotonically and `select@100` gives `+0.179 EM`. | No major gap. | Keep `MuSiQue` as the main generalization dataset and `HotpotQA` as the contrast dataset. |
| C7 | A simple non-oracle setwise selector can recover a meaningful fraction of the oracle gap. | unsupported | No non-oracle setwise method result yet. | This is the main method-side gap for the paper. | Implement and test one minimal heuristic baseline. |
| C8 | `Expand-then-Compose` is a better framing than "better causal / relation graph retrieval." | supported internally | Six failed retrieval-side experiments plus current oracle ceiling analysis point in the same direction. Existing negative evidence is recorded in `CLAUDE.md`. | Needs a clean paper narrative table. | Convert failures into one "why pointwise/retrieval-side fixes failed" table. |

