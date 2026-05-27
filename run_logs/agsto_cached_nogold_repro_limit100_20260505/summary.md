# AG-STO Cached No-Gold Reproduction Limit100

This audit tests whether the strong cached-STO pool depends on row-level gold fields in the upstream per-query report.

## Procedure

1. Stripped row-level keys containing `gold` or `answer` from the first 100 rows of each per-query graph-compare report.
2. Re-ran `/mnt/nvme/zly/HippoRAG/evaluate_transition_component_retriever.py` with the same cached-STO settings:
   - `native_dense_anchor=true`
   - `native_dense_anchor_top_k=20`
   - `context_anchor_k=10`
   - `semantic_residual_weight=12.0`
   - `support_candidate_docs=400`
   - `support_set_size=2`
   - `support_beam_size=12`
3. Exported AG-STO cached pools from the no-gold transition report with `scripts/export_agsto_cached_pool.py`.
4. Recomputed title recall at export time using dataset gold from `reproduce/dataset`, not from the stripped transition rows.

## Removed Row Fields

For each dataset and for both `hipporag_v2` and `hippohead_qgate_lexbeam_topkfact_stabilityfallback_guarded_local_ppr_gamma_0.3`, the following keys were removed from 100 rows:

- `gold_answers`
- `gold_doc_indices`
- `gold_doc_titles`
- `gold_hit_ranks`
- `hidden_gold_doc_indices`
- `hidden_gold_doc_titles`

## Pool Recall Comparison

Old cached means the first 100 rows of `run_logs/agsto_cached_parity_full_20260505/pools/*_agsto_cached_graph_pool100_full.json`.

| Dataset | Old Cached R@5/R@20/R@100 | No-Gold Rebuilt R@5/R@20/R@100 |
|---|---:|---:|
| 2Wiki | 0.9025 / 0.9325 / 0.9625 | 0.9025 / 0.9325 / 0.9625 |
| HotpotQA | 0.9550 / 0.9800 / 0.9850 | 0.9550 / 0.9800 / 0.9850 |
| MuSiQue | 0.6817 / 0.8075 / 0.8758 | 0.6817 / 0.8067 / 0.8758 |

## Pool Overlap With Old Cached

| Dataset | Exact Pool Match | Exact Top-5 Match | Mean Jaccard@5 | Mean Jaccard@100 |
|---|---:|---:|---:|---:|
| 2Wiki | 0.9100 | 0.9800 | 0.9967 | 0.9989 |
| HotpotQA | 0.9600 | 0.9900 | 0.9967 | 0.9993 |
| MuSiQue | 0.8000 | 0.9700 | 0.9933 | 0.9965 |

## Conclusion

The cached-STO strength is reproducible after removing row-level gold/answer fields from the transition inputs. The strong pool is therefore not explained by direct use of `gold_doc_indices` or `gold_answers` inside the cached transition rows.

The remaining explanation is substrate/proposal quality:

- GPT-4o-mini OpenIE from the clean snapshot cache.
- Active native dense anchor reconstruction from Qwen3-Embedding-8B.
- Active semantic residual / hybrid residual proposal lanes.
- Larger support candidate budget.
- Candidate fill in the final pool export.

This does not prove current native AG-STO reproduces the cached pipeline; it shows that the historical cached transition-proposal path is likely gold-free at the row level and is the source of the strong STO+DAEC substrate.
