# Qwen3-8B vs Qwen3-32B Upstream Sensitivity

Protocol: GPT-4o-mini reader fixed; main3 datasets; 1000 queries per dataset. NeocorRAG 8B uses the clean beam k=3 saved-docs replay, so the older greedy k=1 8B Neocor run is excluded. HippoRAG/PropRAG 8B rows use pure pool exports, not DAEC selector outputs.

## Per-Dataset Results

| Method | Dataset | 8B R@5 | 32B R@5 | ΔR@5 | 8B EM | 32B EM | ΔEM | 8B F1 | 32B F1 | ΔF1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EvidenceFlow | 2Wiki | 90.15 | 96.23 | +6.08 | 63.30 | 65.90 | +2.60 | 71.64 | 74.68 | +3.04 |
| EvidenceFlow | HotpotQA | 95.05 | 96.50 | +1.45 | 61.80 | 62.80 | +1.00 | 74.64 | 75.57 | +0.93 |
| EvidenceFlow | MuSiQue | 71.84 | 73.51 | +1.67 | 36.10 | 37.20 | +1.10 | 47.00 | 48.91 | +1.91 |
| HippoRAG | 2Wiki | 83.13 | 82.75 | -0.38 | 56.00 | 56.70 | +0.70 | 62.64 | 63.43 | +0.79 |
| HippoRAG | HotpotQA | 92.30 | 92.60 | +0.30 | 59.60 | 59.30 | -0.30 | 72.89 | 72.51 | -0.38 |
| HippoRAG | MuSiQue | 70.11 | 70.96 | +0.85 | 33.20 | 34.90 | +1.70 | 44.30 | 46.54 | +2.24 |
| PropRAG | 2Wiki | 90.28 | 90.90 | +0.62 | 60.50 | 61.10 | +0.60 | 68.42 | 69.09 | +0.67 |
| PropRAG | HotpotQA | 95.00 | 95.10 | +0.10 | 60.70 | 61.80 | +1.10 | 74.26 | 75.10 | +0.84 |
| PropRAG | MuSiQue | 73.72 | 74.24 | +0.52 | 36.40 | 36.50 | +0.10 | 47.43 | 47.60 | +0.17 |
| HGRAG | 2Wiki | 76.92 | 76.65 | -0.27 | 55.20 | 54.50 | -0.70 | 61.02 | 60.17 | -0.85 |
| HGRAG | HotpotQA | 94.50 | 94.45 | -0.05 | 61.10 | 60.60 | -0.50 | 73.76 | 73.28 | -0.48 |
| HGRAG | MuSiQue | 69.52 | 69.49 | -0.03 | 36.10 | 36.80 | +0.70 | 47.57 | 48.06 | +0.49 |
| NeocorRAG | 2Wiki | 78.35 | 88.95 | +10.60 | 51.40 | 59.80 | +8.40 | 57.40 | 66.55 | +9.15 |
| NeocorRAG | HotpotQA | 93.50 | 95.05 | +1.55 | 59.10 | 62.10 | +3.00 | 72.44 | 74.76 | +2.32 |
| NeocorRAG | MuSiQue | 67.40 | 73.12 | +5.72 | 34.40 | 37.10 | +2.70 | 45.65 | 48.11 | +2.46 |

## Average Over Main3

| Method | Avg 8B R@5 | Avg 32B R@5 | Avg ΔR@5 | Avg 8B EM | Avg 32B EM | Avg ΔEM | Avg 8B F1 | Avg 32B F1 | Avg ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EvidenceFlow | 85.68 | 88.74 | +3.06 | 53.73 | 55.30 | +1.57 | 64.43 | 66.39 | +1.96 |
| HippoRAG | 81.85 | 82.10 | +0.26 | 49.60 | 50.30 | +0.70 | 59.94 | 60.83 | +0.88 |
| PropRAG | 86.33 | 86.75 | +0.41 | 52.53 | 53.13 | +0.60 | 63.37 | 63.93 | +0.56 |
| HGRAG | 80.31 | 80.20 | -0.12 | 50.80 | 50.63 | -0.17 | 60.78 | 60.50 | -0.28 |
| NeocorRAG | 79.75 | 85.71 | +5.96 | 48.30 | 53.00 | +4.70 | 58.50 | 63.14 | +4.64 |

## Source Run Roots
- 8B EvidenceFlow/HGRAG: `run_logs/qwen8b_upstream_gpt4omini_main3_20260518/`
- 8B pure HippoRAG/PropRAG: `run_logs/pure_hippo_prop_qwen8b_gpt4omini_main3_20260518/`
- 8B clean NeocorRAG beam k=3: `run_logs/neocor_8b_k3_gpt4omini_full1000_20260518/`
- 32B EvidenceFlow: `run_logs/all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514/`
- 32B HippoRAG: `run_logs/hipporag_qwen32b_valid_graph_top200_full1000_20260513_r2/`
- 32B PropRAG: `run_logs/baseline_qwen32b_graph_top200_gpt4omini_full_20260512/`
- 32B HGRAG: `run_logs/hgrag_main3_32b_no_think_gpt4omini_full1000_20260517/`
- 32B NeocorRAG: `run_logs/neocorrag_reader_gpt4omini_full1000_20260516/`
