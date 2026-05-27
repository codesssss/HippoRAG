# PropRAG / DAEC / NEOCORRAG Limit-100 Aligned Smoke

Results dir: `/mnt/nvme/code/HippoRAG/run_logs/prop_neocorr_daec_limit100_20260429`

| Dataset | Method | Status | Recall@5 | Recall@20 | EM | F1 |
|---|---|---:|---:|---:|---:|---:|
| 2wikimultihopqa | PropRAG | done | 0.9350 | 0.9625 | 0.5800 | 0.6318 |
| 2wikimultihopqa | PropRAG + DAEC | done | 0.9525 | 0.9675 | 0.5800 | 0.6435 |
| 2wikimultihopqa | PropRAG + DAEC-L1 | done | 0.9375 | 0.9650 | 0.5700 | 0.6164 |
| 2wikimultihopqa | NEOCORRAG | missing | - | - | - | - |
| hotpotqa | PropRAG | done | 0.9300 | 0.9950 | 0.5700 | 0.6912 |
| hotpotqa | PropRAG + DAEC | done | 0.9500 | 0.9950 | 0.5700 | 0.6912 |
| hotpotqa | PropRAG + DAEC-L1 | done | 0.9500 | 0.9950 | 0.6000 | 0.7079 |
| hotpotqa | NEOCORRAG | missing | - | - | - | - |
| musique | PropRAG | done | 0.6775 | 0.8900 | 0.3800 | 0.4374 |
| musique | PropRAG + DAEC | done | 0.7175 | 0.8983 | 0.3900 | 0.4660 |
| musique | PropRAG + DAEC-L1 | done | 0.6600 | 0.9058 | 0.3700 | 0.4202 |
| musique | NEOCORRAG | missing | - | - | - | - |

Protocol notes:

- PropRAG rows use the fixed PropRAG top-100 pool exported under `run_logs/proprag_pool_exports_full1000_20260424/`.
- `PropRAG + DAEC` uses `eval_causal_qwen3.py --setwise_selector dtc_embed` with the prior original-DAEC PropRAG-pool flags.
- `PropRAG + DAEC-L1` uses `--setwise_selector daec_noisyor` on the same PropRAG pool.
- NEOCORRAG is run natively through `scripts/run_neocorrag_aligned.py`; retrieval Recall@5/20 is from NeocorRAG retrieval, EM/F1 from its QA output.
- All rows are `limit=100`, `qa_top_k=5`, NV-Embed endpoint `localhost:8019`, and Qwen3-8B API reader where applicable.
