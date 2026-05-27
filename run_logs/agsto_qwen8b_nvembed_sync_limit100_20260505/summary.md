# AG-STO Qwen3-8B + NV-Embed-v2 Sync Limit100 Summary

Date: 2026-05-05

## Protocol

- OpenIE substrate: `outputs_step0_general_nvembed_{dataset}/openie_results_ner_qwen3-8b.json`
- Chunk embeddings: `outputs_step0_general_nvembed_{dataset}/qwen3-8b_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet`
- Query embedding API: `http://localhost:8019/v1/embeddings`
- Served embedding model: `nvidia/NV-Embed-v2`
- Reader / DAEC binding model: `qwen3-8b-train`
- Selector: `daec_noisyor_llm`
- Binding title match: `wiki_title`
- Limit: first 100 queries

## Pool Recall, First 100

| Dataset | Substrate | R@5 | R@20 | R@100 |
|---|---|---:|---:|---:|
| 2Wiki | HippoRAG | 0.8150 | 0.9000 | 0.9425 |
| 2Wiki | ProPRAG | 0.9350 | 0.9625 | 0.9875 |
| 2Wiki | AG-STO Qwen3 old | 0.8375 | 0.8975 | 0.9125 |
| 2Wiki | AG-STO Qwen3 owned | 0.8575 | 0.9100 | 0.9200 |
| 2Wiki | AG-STO Qwen3+NV sync | 0.8850 | 0.9250 | 0.9700 |
| 2Wiki | AG-STO 4o-mini cached | 0.9025 | 0.9325 | 0.9625 |
| HotpotQA | HippoRAG | 0.9250 | 0.9950 | 1.0000 |
| HotpotQA | ProPRAG | 0.9300 | 0.9950 | 1.0000 |
| HotpotQA | AG-STO Qwen3 old | 0.8700 | 0.9650 | 0.9850 |
| HotpotQA | AG-STO Qwen3 owned | 0.8700 | 0.9750 | 0.9900 |
| HotpotQA | AG-STO Qwen3+NV sync | 0.9200 | 0.9700 | 0.9850 |
| HotpotQA | AG-STO 4o-mini cached | 0.9550 | 0.9800 | 0.9850 |
| MuSiQue | HippoRAG | 0.6700 | 0.8458 | 0.9558 |
| MuSiQue | ProPRAG | 0.6975 | 0.8900 | 0.9758 |
| MuSiQue | AG-STO Qwen3 old | 0.5042 | 0.6900 | 0.8033 |
| MuSiQue | AG-STO Qwen3 owned | 0.5125 | 0.6817 | 0.7983 |
| MuSiQue | AG-STO Qwen3+NV sync | 0.6383 | 0.7517 | 0.8308 |
| MuSiQue | AG-STO 4o-mini cached | 0.6817 | 0.8075 | 0.8758 |

## DAEC QA, First 100 / Limit100

Rows marked first100 are recomputed from the first 100 per-query selector traces of full1000 runs.

| Dataset | Method | EM | F1 | Selector R@5 | Top5 Base EM/F1 |
|---|---|---:|---:|---:|---:|
| 2Wiki | HippoRAG+DAEC | 0.540 | 0.6019 | 0.9025 | 0.510 / 0.5470 |
| 2Wiki | ProPRAG+DAEC | 0.590 | 0.6437 | 0.9400 | 0.580 / 0.6318 |
| 2Wiki | AG-STO Qwen3 old + DAEC | 0.500 | 0.5709 | 0.8850 | 0.480 / 0.5257 |
| 2Wiki | AG-STO Qwen3 owned + DAEC | 0.510 | 0.5671 | 0.9050 | 0.490 / 0.5353 |
| 2Wiki | AG-STO Qwen3+NV sync + DAEC | 0.540 | 0.6051 | 0.9350 | 0.490 / 0.5303 |
| 2Wiki | AG-STO 4o-mini cached + DAEC | 0.550 | 0.6241 | 0.9400 | 0.600 / 0.6482 |
| HotpotQA | HippoRAG+DAEC | 0.600 | 0.7062 | 0.9450 | 0.560 / 0.6712 |
| HotpotQA | ProPRAG+DAEC | 0.600 | 0.7120 | 0.9450 | 0.570 / 0.6912 |
| HotpotQA | AG-STO Qwen3 old + DAEC | 0.550 | 0.6604 | 0.9350 | 0.500 / 0.6088 |
| HotpotQA | AG-STO Qwen3 owned + DAEC | 0.570 | 0.6763 | 0.9400 | 0.470 / 0.5763 |
| HotpotQA | AG-STO Qwen3+NV sync + DAEC | 0.580 | 0.6838 | 0.9400 | 0.530 / 0.6463 |
| HotpotQA | AG-STO 4o-mini cached + DAEC | 0.610 | 0.7296 | 0.9700 | 0.580 / 0.7021 |
| MuSiQue | HippoRAG+DAEC | 0.340 | 0.4376 | 0.6933 | 0.300 / 0.3687 |
| MuSiQue | ProPRAG+DAEC | 0.390 | 0.4695 | 0.7425 | 0.380 / 0.4374 |
| MuSiQue | AG-STO Qwen3 old + DAEC | 0.250 | 0.3165 | 0.5875 | 0.190 / 0.2432 |
| MuSiQue | AG-STO Qwen3 owned + DAEC | 0.280 | 0.3538 | 0.6250 | 0.250 / 0.3097 |
| MuSiQue | AG-STO Qwen3+NV sync + DAEC | 0.320 | 0.3917 | 0.6425 | 0.310 / 0.3746 |
| MuSiQue | AG-STO 4o-mini cached + DAEC | 0.380 | 0.4333 | 0.7108 | 0.390 / 0.4408 |

## Average DAEC QA

| Method | Avg EM | Avg F1 | Avg Selector R@5 |
|---|---:|---:|---:|
| HippoRAG+DAEC | 0.493 | 0.5819 | 0.8469 |
| ProPRAG+DAEC | 0.527 | 0.6084 | 0.8758 |
| AG-STO Qwen3 old + DAEC | 0.433 | 0.5159 | 0.8025 |
| AG-STO Qwen3 owned + DAEC | 0.453 | 0.5324 | 0.8233 |
| AG-STO Qwen3+NV sync + DAEC | 0.480 | 0.5602 | 0.8392 |
| AG-STO 4o-mini cached + DAEC | 0.513 | 0.5957 | 0.8736 |

## Primary Comparison Against Bare Retrievers

This is the cleaner comparison if the claim is that AG-STO+DAEC is a train-free
evidence composer stacked on top of a graph substrate.  The compared baselines
are the bare top-5 reader outputs from HippoRAG and ProPRAG, without DAEC.

| Dataset | Bare HippoRAG EM/F1/R@5 | Bare ProPRAG EM/F1/R@5 | AG-STO Qwen3+NV sync + DAEC EM/F1/R@5 |
|---|---:|---:|---:|
| 2Wiki | 0.510 / 0.5470 / 0.8150 | 0.580 / 0.6318 / 0.9350 | 0.540 / 0.6051 / 0.9350 |
| HotpotQA | 0.560 / 0.6712 / 0.9250 | 0.570 / 0.6912 / 0.9300 | 0.580 / 0.6838 / 0.9400 |
| MuSiQue | 0.300 / 0.3687 / 0.6700 | 0.380 / 0.4374 / 0.6975 | 0.320 / 0.3917 / 0.6425 |
| Avg | 0.457 / 0.5290 / 0.8033 | 0.510 / 0.5868 / 0.8542 | 0.480 / 0.5602 / 0.8392 |

Delta of AG-STO Qwen3+NV sync + DAEC:

| Baseline | Avg EM Delta | Avg F1 Delta | Avg R@5 Delta |
|---|---:|---:|---:|
| vs bare HippoRAG | +0.023 | +0.0312 | +0.0359 |
| vs bare ProPRAG | -0.030 | -0.0266 | -0.0150 |

## Main Finding

Using Qwen3-8B OpenIE with NV-Embed-v2 dense anchor / semantic residual recovers much of the gap over prior native Qwen3 AG-STO:

- vs AG-STO Qwen3 old + DAEC: +0.047 EM / +0.0443 F1 / +0.0367 R@5 average
- vs AG-STO Qwen3 owned + DAEC: +0.027 EM / +0.0278 F1 / +0.0158 R@5 average

It does not fully recover the cached 4o-mini substrate:

- vs AG-STO 4o-mini cached + DAEC: -0.033 EM / -0.0355 F1 / -0.0344 R@5 average

Against bare retrievers, the synced AG-STO+DAEC beats bare HippoRAG on average
but does not beat bare ProPRAG on average.  It matches or exceeds bare ProPRAG
on HotpotQA, is competitive on 2Wiki recall but lower in QA, and still loses on
MuSiQue.  Interpretation: the sync fixes a real implementation/config gap,
especially on 2Wiki and HotpotQA.  The remaining loss is mostly MuSiQue and
likely comes from OpenIE/proposal substrate quality rather than DAEC itself.
