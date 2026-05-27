# LLM-Direct-Select over PropRAG Pool100, Limit100

Date: 2026-05-06

## Protocol

- Pool: `run_logs/proprag_pool_exports_full1000_20260424/*_pool100.json`
- Selector model: `qwen3-8b-train`
- Reader: existing `eval_causal_qwen3.py` path with `qwen3-8b-train`, `qa_top_k=5`, `qa_doc_max_chars=2048`
- Save dir / embeddings: `outputs_step0_general_nvembed`, `VLLM/nvidia/NV-Embed-v2`
- LLM selector outputs a reordered external pool; evaluation uses `--setwise_selector none`
- `top5` and `daec` rows are recomputed from the first 100 query traces of `run_logs/layer1_proprag_pool_eval_fixed_20260424/*_proprag_pool_daec_oracle.json`

## Main Results

| Dataset | Variant | EM | F1 | R@5 | Parse | Avg prompt tok | Avg latency s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | top5 | 0.580 | 0.632 | 0.935 | - | - | - |
| 2Wiki | daec | 0.580 | 0.643 | 0.953 | - | - | - |
| 2Wiki | llm_title | 0.500 | 0.541 | 0.833 | 100/100 | 1324.9 | 0.268 |
| 2Wiki | llm_snippet128 | 0.520 | 0.579 | 0.887 | 100/100 | 4335.2 | 0.378 |
| HotpotQA | top5 | 0.570 | 0.691 | 0.930 | - | - | - |
| HotpotQA | daec | 0.570 | 0.691 | 0.950 | - | - | - |
| HotpotQA | llm_title | 0.540 | 0.647 | 0.795 | 100/100 | 1336.5 | 0.276 |
| HotpotQA | llm_snippet128 | 0.510 | 0.606 | 0.805 | 100/100 | 4904.1 | 0.392 |
| MuSiQue | top5 | 0.380 | 0.437 | 0.698 | - | - | - |
| MuSiQue | daec | 0.390 | 0.466 | 0.738 | - | - | - |
| MuSiQue | llm_title | 0.260 | 0.329 | 0.531 | 100/100 | 1268.8 | 0.274 |
| MuSiQue | llm_snippet128 | 0.280 | 0.354 | 0.533 | 100/100 | 4748.6 | 0.406 |

## Selector Behavior

| Dataset | Variant | Changed from top5 | Any selected rank >= 6 | Avg max selected rank |
|---|---:|---:|---:|---:|
| 2Wiki | llm_title | 66/100 | 66/100 | 36.9 |
| 2Wiki | llm_snippet128 | 75/100 | 72/100 | 39.2 |
| HotpotQA | llm_title | 84/100 | 82/100 | 39.2 |
| HotpotQA | llm_snippet128 | 82/100 | 78/100 | 28.7 |
| MuSiQue | llm_title | 94/100 | 93/100 | 46.8 |
| MuSiQue | llm_snippet128 | 97/100 | 96/100 | 35.0 |

## Takeaways

1. LLM-direct-select is not a stronger same-pool compositor under this protocol. Both title-only and snippet128 variants underperform top5 and DAEC on all three datasets.
2. The failure is not parsing. All six selector runs parsed successfully on 100/100 queries with no fallback fills.
3. The main failure mode is over-exploration. The LLM frequently replaces top-ranked PropRAG evidence with deep-pool candidates, causing large R@5 drops, especially on HotpotQA and MuSiQue.
4. This protects the DAEC machinery against the reviewer baseline "why not directly ask the LLM to select 5 docs from top100?" At least with qwen3-8b, direct selection is weaker than deterministic demand-aware composition.

## Artifacts

- Launcher: `run_logs/launch_llm_direct_select_proprag_limit100_20260506.sh`
- Selector script: `scripts/run_llm_direct_pool_selector.py`
- Output root: `run_logs/llm_direct_select_proprag_limit100_20260506/`
- Selected pools: `run_logs/llm_direct_select_proprag_limit100_20260506/selected_pools/`
- Caches: `run_logs/llm_direct_select_proprag_limit100_20260506/caches/`
