# D-PathRAG Selector K-Fold Summary

- Pool: `proprag`
- Cache: `data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl`
- Embeddings: `data/dpathrag/cache/selector_embedding_proprag_local1000_rp64.npz`
- Rows: `1000`
- Fold size: `200`
- Num folds: `5`
- Train/eval qid overlap: `0`

## Aggregate Selector Metrics

| Variant | Support Recall | Support Complete | Selected Gold | Bridge Entity Recall | Selection Overlap |
|---|---:|---:|---:|---:|---:|
| rank_topk | 0.9028 | 0.7720 | 2.2270 | 0.9365 | 1.0000 |
| model | 0.9233 | 0.8050 | 2.2660 | 0.9416 | 0.6192 |

## Delta

- Support-complete delta: `+0.0330`
- Support-recall delta: `+0.0205`
- Bridge-entity-recall delta: `+0.0051`

## Folds

| Fold | Eval Start | Train Rows | Eval Rows | Best Epoch | Rank Complete | Model Complete | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 800 | 200 | 4 | 0.8150 | 0.8150 | +0.0000 |
| 1 | 200 | 800 | 200 | 4 | 0.7750 | 0.8050 | +0.0300 |
| 2 | 400 | 800 | 200 | 7 | 0.7600 | 0.7950 | +0.0350 |
| 3 | 600 | 800 | 200 | 2 | 0.7600 | 0.8100 | +0.0500 |
| 4 | 800 | 800 | 200 | 4 | 0.7500 | 0.8000 | +0.0500 |
