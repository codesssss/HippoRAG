# AG-STO Cached-Parity Full1000 Main Results

Date: 2026-05-05

This is the corrected main comparison for AG-STO.  The AG-STO rows use the cached
v12 transition report, not the owned-OpenIE/native-proposal ablation.

## Protocol

- Datasets: `musique`, `hotpotqa`, `2wikimultihopqa`
- Query count: 1000 per dataset
- Reader: `qwen3-8b-train`
- Qwen thinking mode: disabled with `/no_think` and `enable_thinking=false`
- Embedding server: `VLLM/nvidia/NV-Embed-v2`
- AG-STO proposal source: cached transition report
- AG-STO policy: `graph`
- AG-STO `max_queries`: `0`
- AG-STO external pool source: `agsto_cached_graph_pool100`
- Baselines: bare HippoRAG and bare ProPRAG from previous full1000 pool100 runs

Cached transition report:

```text
/mnt/nvme/zly/HippoRAG/outputs_full_sfb_supportfusion_gpt4omini_qwen_cleanbaseline_rebuild_20260425/reports/transition_component_retriever_full_native_denseanchor20_context10_support400.json
```

Important note: `run_logs/agsto_owned_openie_full_20260505` is not the main
AG-STO v12 cached-parity result.  It is an owned-OpenIE/native-proposal ablation.
The main AG-STO result is this directory:

```text
run_logs/agsto_cached_parity_full_20260505/
```

## Main Table

```text
Dataset          Method       Pool R@100     EM      F1  Records
---------------  -----------  ----------  -----  ------  -------
musique          HippoRAG         0.9382  0.311  0.3947     1000
musique          ProPRAG          0.9689  0.330  0.4266     1000
musique          AG-STO           0.8893  0.320  0.4129     1000
musique          AG-STO+DAEC      0.8893  0.346  0.4506     1000

hotpotqa         HippoRAG         0.9965  0.577  0.7010     1000
hotpotqa         ProPRAG          0.9990  0.595  0.7227     1000
hotpotqa         AG-STO           0.9895  0.608  0.7213     1000
hotpotqa         AG-STO+DAEC      0.9895  0.633  0.7496     1000

2wikimultihopqa  HippoRAG         0.9545  0.524  0.5850     1000
2wikimultihopqa  ProPRAG          0.9872  0.575  0.6457     1000
2wikimultihopqa  AG-STO           0.9822  0.605  0.6782     1000
2wikimultihopqa  AG-STO+DAEC      0.9822  0.623  0.7045     1000
---------------  -----------  ----------  -----  ------  -------
Avg              HippoRAG         0.9631  0.471  0.5602   1000x3
Avg              ProPRAG          0.9850  0.500  0.5983   1000x3
Avg              AG-STO           0.9537  0.511  0.6041   1000x3
Avg              AG-STO+DAEC      0.9537  0.534  0.6349   1000x3
```

## Delta Table

```text
Dataset          AG-STO+DAEC vs HippoRAG      AG-STO+DAEC vs ProPRAG
---------------  ---------------------------  --------------------------
musique          +0.035 EM / +0.0559 F1       +0.016 EM / +0.0240 F1
hotpotqa         +0.056 EM / +0.0486 F1       +0.038 EM / +0.0269 F1
2wikimultihopqa  +0.099 EM / +0.1195 F1       +0.048 EM / +0.0588 F1
---------------  ---------------------------  --------------------------
Avg              +0.063 EM / +0.0747 F1       +0.034 EM / +0.0366 F1
```

## AG-STO DAEC Gain

```text
Dataset          AG-STO Base EM/F1  AG-STO+DAEC EM/F1  Delta
---------------  -----------------  -----------------  -----------------
musique          0.320 / 0.4129     0.346 / 0.4506     +0.026 / +0.0377
hotpotqa         0.608 / 0.7213     0.633 / 0.7496     +0.025 / +0.0283
2wikimultihopqa  0.605 / 0.6782     0.623 / 0.7045     +0.018 / +0.0263
```

## Validation

All compared files use 1000 records per dataset.  For all rows:

- `question_mismatch_count = 0`
- `query_idx_mismatch_count = 0`
- `unmatched_doc_count = 0`

For all AG-STO+DAEC rows:

- `llm_binding_total_failures = 0`

AG-STO cached pool statistics:

```text
Dataset          Pool R@5  Pool R@20  Pool R@100  Mean Pool Size
---------------  -------  ---------  ----------  --------------
musique           0.6987     0.8287      0.8893           41.34
hotpotqa          0.9390     0.9815      0.9895           36.76
2wikimultihopqa   0.9155     0.9565      0.9822           43.40
```

## Source Files

AG-STO corrected cached-parity outputs:

```text
run_logs/agsto_cached_parity_full_20260505/evals/musique_agsto_cached_graph_pool100_base_full.json
run_logs/agsto_cached_parity_full_20260505/evals/musique_agsto_cached_graph_pool100_daec_llm_full.json
run_logs/agsto_cached_parity_full_20260505/evals/hotpotqa_agsto_cached_graph_pool100_base_full.json
run_logs/agsto_cached_parity_full_20260505/evals/hotpotqa_agsto_cached_graph_pool100_daec_llm_full.json
run_logs/agsto_cached_parity_full_20260505/evals/2wikimultihopqa_agsto_cached_graph_pool100_base_full.json
run_logs/agsto_cached_parity_full_20260505/evals/2wikimultihopqa_agsto_cached_graph_pool100_daec_llm_full.json
```

Baseline outputs:

```text
run_logs/daec_llm_wiki_title_dense_hipporag_full1000_20260503/*_hipporag_wiki_title_daec_llm_full1000.json
run_logs/daec_llm_wiki_title_proprag_full1000_20260503/*_proprag_wiki_title_daec_llm_full1000.json
```

Launcher:

```text
run_logs/launch_agsto_cached_parity_compare_full_20260505.sh
```
