# ETv4 Query-Local GraphRAG Full1000 Comparison

Date: 2026-05-13

## Scope

This note records the full1000 comparison between `ETV4`, `PropRAG`, and
`HippoRAG` under the aligned Qwen3-32B graph / GPT-4o-mini reader protocol.

ETV4 is treated here as an end-to-end GraphRAG pipeline with a method-owned
query-local document graph.  It should not be framed as a dense reranker.
Dense/textual retrieval is the query entry mechanism; graph construction,
traversal, and graph-constrained document selection are the method-owned stages.

Audit update:

- The `HippoRAG` rows in the current Qwen32B baseline export are invalid as
  HippoRAG graph results on all three datasets.  The exporter fell back to
  dense retrieval because no OpenIE results were available for the legacy
  fact-graph base retrieval path.
- Keep the HippoRAG rows below only as a dense-fallback audit artifact until a
  corrected HippoRAG graph rerun is completed.
- ETV4 and PropRAG rows are unaffected by this HippoRAG exporter issue.

## Protocol

| Item | Setting |
| --- | --- |
| Rows | full1000 per dataset |
| Datasets | `2wikimultihopqa`, `hotpotqa`, `musique` |
| Graph / OpenIE LLM | Qwen3-32B `/no_think` |
| ETV4 graph endpoint | `http://localhost:8045/v1` |
| Reader LLM | `gpt-4o-mini` |
| Reader endpoint | `https://yunwu.ai/v1` |
| Reader evidence budget | top5 full passages |
| Reader max tokens | `none` |
| Embedding | `VLLM/nvidia/NV-Embed-v2` |
| Embedding endpoint | `http://localhost:8019/v1/embeddings` |
| Candidate pool | top200 for baseline pools; ETV4 method-owned query-local candidate universe |

Metric protocol:

- Retrieval metrics in the master table are title-level for consistency with
  the HippoRAG and PropRAG pool exporters.
- For ETV4, `R@20/100/200` and `All@20/100/200` are computed by appending the
  remaining query-local candidate universe after the graph-selected top5.
- Reader QA always uses only the top5 documents for all methods.
- ETV4's native retrieval JSON also stores exact document-index recall.  That
  exact-doc metric is stricter on MuSiQue because many examples contain
  duplicate titles with multiple passages.

Paper table note:

```text
All retrieval metrics are computed at title level for consistency with
HippoRAG/PropRAG exports. For ETV4, R@20/100/200 and All@20/100/200 are
computed by appending the remaining query-local candidate universe after the
graph-selected top-5. Reader QA always uses the top-5 documents for all methods.
```

## Main Full1000 Table

```text
Dataset   | Method   |     R@5 |    R@20 |   R@100 |   R@200 |   All@5 |  All@20 |  All@100 |  All@200 |      EM |      F1
----------+----------+---------+---------+---------+---------+---------+---------+----------+----------+---------+--------
2Wiki     | HippoRAG |  0.7238 |  0.7990 |  0.8770 |  0.9120 |  0.4290 |  0.5580 |   0.7060 |   0.7690 |  0.5170 |  0.5619
2Wiki     | PropRAG  |  0.9090 |  0.9633 |  0.9910 |  0.9960 |  0.7800 |  0.9050 |   0.9740 |   0.9890 |  0.6110 |  0.6909
2Wiki     | ETV4     |  0.9350 |  0.9688 |  0.9925 |  0.9932 |  0.8260 |  0.9150 |   0.9820 |   0.9840 |  0.6470 |  0.7329
HotpotQA  | HippoRAG |  0.9305 |  0.9830 |  0.9925 |  0.9945 |  0.8640 |  0.9680 |   0.9850 |   0.9890 |  0.6050 |  0.7368
HotpotQA  | PropRAG  |  0.9510 |  0.9945 |  0.9995 |  0.9995 |  0.9070 |  0.9900 |   0.9990 |   0.9990 |  0.6180 |  0.7510
HotpotQA  | ETV4     |  0.9505 |  0.9885 |  0.9975 |  0.9980 |  0.9050 |  0.9780 |   0.9960 |   0.9970 |  0.6100 |  0.7413
MuSiQue   | HippoRAG |  0.6888 |  0.8318 |  0.9091 |  0.9365 |  0.3860 |  0.6180 |   0.7660 |   0.8370 |  0.3300 |  0.4330
MuSiQue   | PropRAG  |  0.7424 |  0.9037 |  0.9737 |  0.9841 |  0.4760 |  0.7600 |   0.9280 |   0.9530 |  0.3650 |  0.4760
MuSiQue   | ETV4     |  0.7442 |  0.8666 |  0.9432 |  0.9602 |  0.4720 |  0.6830 |   0.8520 |   0.8950 |  0.3670 |  0.4801
```

CSV form:

```csv
Dataset,Method,R@5,R@20,R@100,R@200,All@5,All@20,All@100,All@200,EM,F1
2Wiki,HippoRAG,0.7238,0.7990,0.8770,0.9120,0.4290,0.5580,0.7060,0.7690,0.5170,0.5619
2Wiki,PropRAG,0.9090,0.9633,0.9910,0.9960,0.7800,0.9050,0.9740,0.9890,0.6110,0.6909
2Wiki,ETV4,0.9350,0.9688,0.9925,0.9932,0.8260,0.9150,0.9820,0.9840,0.6470,0.7329
HotpotQA,HippoRAG,0.9305,0.9830,0.9925,0.9945,0.8640,0.9680,0.9850,0.9890,0.6050,0.7368
HotpotQA,PropRAG,0.9510,0.9945,0.9995,0.9995,0.9070,0.9900,0.9990,0.9990,0.6180,0.7510
HotpotQA,ETV4,0.9505,0.9885,0.9975,0.9980,0.9050,0.9780,0.9960,0.9970,0.6100,0.7413
MuSiQue,HippoRAG,0.6888,0.8318,0.9091,0.9365,0.3860,0.6180,0.7660,0.8370,0.3300,0.4330
MuSiQue,PropRAG,0.7424,0.9037,0.9737,0.9841,0.4760,0.7600,0.9280,0.9530,0.3650,0.4760
MuSiQue,ETV4,0.7442,0.8666,0.9432,0.9602,0.4720,0.6830,0.8520,0.8950,0.3670,0.4801
```

## Interpretation

2Wiki:

- ETV4 is the strongest line on retrieval and QA.
- ETV4 improves over PropRAG by `+0.0260` R@5, `+0.0460` All@5,
  `+0.0360` EM, and `+0.0420` F1.
- This is the cleanest evidence that query-local graph readout provides value
  beyond the baseline graph systems.

HotpotQA:

- PropRAG is slightly stronger overall.
- ETV4 is essentially tied on R@5 (`0.9505` vs `0.9510`) and All@5
  (`0.9050` vs `0.9070`), but trails on QA (`0.6100 / 0.7413` versus
  `0.6180 / 0.7510`).
- HotpotQA should be framed as near-tie retrieval with a small reader-utility
  gap, not as a decisive win.

MuSiQue:

- Under the unified title-level protocol, ETV4 slightly beats PropRAG on R@5
  (`0.7442` vs `0.7424`) and QA (`0.3670 / 0.4801` vs `0.3650 / 0.4760`).
- PropRAG has much stronger broad-pool coverage: ETV4 trails by `-0.0371` at
  R@20, `-0.0305` at R@100, `-0.0239` at R@200, and `-0.0770` at All@20.
- The correct diagnosis is not that ETV4 top5 fails on MuSiQue.  The real
  limitation is that the query-local candidate universe is narrower than the
  PropRAG broad pool at deeper ranks.

## MuSiQue Metric Hygiene

The earlier apparent MuSiQue drop was partly a metric mismatch:

```text
PropRAG/HippoRAG pool exports: title-level recall.
ETV4 native report: exact document-index recall.
```

When the metrics are aligned:

| MuSiQue protocol | PropRAG R@5 | ETV4 R@5 | Delta | PropRAG All@5 | ETV4 All@5 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Title-level | 0.7424 | 0.7442 | +0.0018 | 0.4760 | 0.4720 | -0.0040 |
| Exact doc-index | 0.7151 | 0.7183 | +0.0032 | 0.4310 | 0.4220 | -0.0090 |

This explains why PropRAG can look stronger in title-level retrieval while not
winning QA: duplicate-title hits in MuSiQue do not always correspond to the
exact passage the reader needs.

## Source Artifacts

ETV4 graph/retrieval run:

```text
run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/
```

ETV4 GPT-4o-mini reader outputs:

```text
run_logs/etv4_clean_mainline_qwen32b_reader_gpt4omini_20260512/reports/2wikimultihopqa_etv4_qwen32b_gpt4omini_reader_qa.json
run_logs/etv4_clean_mainline_qwen32b_reader_gpt4omini_20260512/reports/hotpotqa_etv4_qwen32b_gpt4omini_reader_qa.json
run_logs/etv4_clean_mainline_qwen32b_reader_gpt4omini_20260512/reports/musique_etv4_qwen32b_gpt4omini_reader_qa.json
```

Baseline pools and reader outputs:

```text
run_logs/baseline_qwen32b_graph_top200_gpt4omini_full_20260512/pools/hipporag/
run_logs/baseline_qwen32b_graph_top200_gpt4omini_full_20260512/pools/proprag/
run_logs/baseline_qwen32b_graph_top200_gpt4omini_full_20260512/reader_qa/hipporag/
run_logs/baseline_qwen32b_graph_top200_gpt4omini_full_20260512/reader_qa/proprag/
```

Launchers:

```text
run_logs/launch_etv4_qwen32b_graph_full1000_20260512.sh
run_logs/launch_ready_graph_reader_full_20260512.sh
run_logs/watch_etv4_musique_reader_full_20260512.sh
run_logs/launch_etv4_hotpot_reader_qwen32b_20260513.sh
```

## Paper-Plan Implications

- Main positive claim: query-local document graphs can deliver competitive or
  stronger GraphRAG E2E performance without a global graph.
- Strongest dataset evidence: 2Wiki, where ETV4 wins retrieval and reader QA.
- Honest limitation: ETV4's query-local candidate universe is narrower than
  PropRAG on MuSiQue at deeper ranks, even though top5 and QA remain
  competitive.
- Avoid claiming universal dominance.  The correct framing is strong 2Wiki,
  near-tie HotpotQA retrieval with small QA gap, and MuSiQue top5/QA competitive
  but broad-pool coverage weaker.

## HippoRAG Baseline Audit

The low 2Wiki HippoRAG score was audited on 2026-05-13 and traced to dense
fallback, not to HippoRAG graph behavior.  The same audit applies to HotpotQA
and MuSiQue.

The current Qwen32B HippoRAG export logs contain:

```text
V2 legacy fact-graph base retrieval requested, but no OpenIE results were found. Falling back to dense ranking.
```

This appears in all three current HippoRAG export logs:

```text
run_logs/baseline_qwen32b_graph_top200_gpt4omini_full_20260512/logs/export_hipporag_2wikimultihopqa.log
run_logs/baseline_qwen32b_graph_top200_gpt4omini_full_20260512/logs/export_hipporag_hotpotqa.log
run_logs/baseline_qwen32b_graph_top200_gpt4omini_full_20260512/logs/export_hipporag_musique.log
```

The corresponding output directories contain only chunk embeddings and no
OpenIE, entity embeddings, fact embeddings, or graph pickle:

```text
outputs_hipporag_qwen32b_nothink_top200_20260512_2wikimultihopqa/
outputs_hipporag_qwen32b_nothink_top200_20260512_hotpotqa/
outputs_hipporag_qwen32b_nothink_top200_20260512_musique/
```

Observed files:

```text
qwen3-32b-judge_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet
```

No graph artifacts were produced.

The current HippoRAG metrics match the historical dense pool metrics exactly at
R@5/20/100:

| Dataset | Dense R@5 | Current Hippo export R@5 | Dense R@20 | Current Hippo export R@20 | Dense R@100 | Current Hippo export R@100 | Top100 identical queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 0.7238 | 0.7238 | 0.7990 | 0.7990 | 0.8770 | 0.8770 | 996 / 1000 |
| HotpotQA | 0.9305 | 0.9305 | 0.9830 | 0.9830 | 0.9925 | 0.9925 | 1000 / 1000 |
| MuSiQue | 0.6888 | 0.6888 | 0.8318 | 0.8318 | 0.9091 | 0.9091 | 998 / 1000 |

For HotpotQA, current HippoRAG top5 and top100 are exactly identical to the
dense pool for all 1000 queries.  For MuSiQue, top5 is exactly identical for all
1000 queries and top100 is identical for 998 queries.

Historical valid HippoRAG exports were higher than the current dense fallback:

| Dataset | Historical HippoRAG R@5 | Historical HippoRAG R@20 | Historical HippoRAG R@100 | Current dense-fallback R@5 | Current dense-fallback R@20 | Current dense-fallback R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 0.8313 | 0.9048 | 0.9545 | 0.7238 | 0.7990 | 0.8770 |
| HotpotQA | 0.9230 | 0.9885 | 0.9965 | 0.9305 | 0.9830 | 0.9925 |
| MuSiQue | 0.7011 | 0.8662 | 0.9382 | 0.6888 | 0.8318 | 0.9091 |

Root cause:

- `scripts/export_hipporag_pool.py` builds a `BaseConfig` with
  `causal_engine_version="v2"`, `causal_enabled=False`, and
  `causal_v2_base_retrieval_mode="legacy_fact_graph"`.
- In `src/hipporag/HippoRAG.py`, the V2 indexing path inserts chunk embeddings
  and skips graph indexing when causal features are disabled and the base
  retrieval mode is not `general_relation_graph`.
- As a result, `load_existing_openie([])` returns empty during V2 legacy
  fact-graph preparation, and retrieval falls back to dense ranking.

Decision:

- Do not use the current Qwen32B `HippoRAG` rows as HippoRAG graph baselines.
- Either relabel them as dense fallback or rerun HippoRAG with a corrected
  legacy graph configuration.
- A corrected HippoRAG rerun should either use the legacy indexing path that
  actually performs OpenIE and graph construction, or explicitly reuse a valid
  OpenIE/graph asset directory.
