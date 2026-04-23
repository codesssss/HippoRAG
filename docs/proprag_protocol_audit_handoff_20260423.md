# PropRAG / DtC Protocol Audit Handoff

Date: 2026-04-23

This note is for auditing whether the new PropRAG artifacts, the HippoRAG baseline, and our DtC method are being compared under the same protocol. The short answer before audit is:

- Same broad evaluation environment: same datasets, Qwen3-8B no-think reader, NV-Embed-v2 endpoint, `retrieval_top_k=100`, `qa_top_k=5`, and `max_new_tokens=2048`.
- Not yet the same fixed-pool method protocol: PropRAG builds its own proposition graph and retrieves its own top-100 pool; DtC composes from HippoRAG's top-100 pool. Treat PropRAG as a cross-system baseline unless it is adapted to consume the exact same candidate pool.

## Repositories

- HippoRAG working repo:
  `/mnt/nvme/code/HippoRAG`
- PropRAG working repo:
  `/mnt/nvme/code/PropRAG`

HippoRAG relevant commits:

- `63909cd Add repairable filter to DtC greedy selection`
- `7596b90 Add repairable residual SC-SER`
- `7acf5e9 Add SC-SER pilot100 launch script`
- `71e3508 Add sufficiency-calibrated DtC repair`
- `187cfc8 Add demand-gated DtC selection`

## Core Question For Audit

Audit these three comparisons separately:

1. HippoRAG baseline vs DtC inside the same JSON.
   This is the clean same-pool comparison. Baseline is HippoRAG top-5; DtC selects from the same HippoRAG top-100.

2. PropRAG clean no-think 100 vs HippoRAG/DtC 100.
   This is same reader / same embedding / same top-k budget, but not same fixed candidate pool.

3. PropRAG clean no-think full1000 vs HippoRAG/DtC full1000.
   Currently running. Same caveat as above: this is cross-system unless PropRAG is forced to use HippoRAG's pool.

## HippoRAG Baseline / DtC Code

Main evaluation harness:

- `/mnt/nvme/code/HippoRAG/scripts/eval_causal_qwen3.py`

DtC selector implementation:

- `/mnt/nvme/code/HippoRAG/scripts/dtc_embed_utils.py`

Tests for DtC selector behavior:

- `/mnt/nvme/code/HippoRAG/tests/test_dtc_embed_utils.py`

Current key CLI flags to inspect in `eval_causal_qwen3.py`:

- `--setwise_selector dtc_embed`
- `--setwise_pool_k 100`
- `--qa_top_k 5`
- `--embedding_name VLLM/nvidia/NV-Embed-v2`
- `--embedding_base_url http://localhost:8019/v1/embeddings`
- `--llm_request_name qwen3-8b-train`
- `--dtc_rank_weight 0.2`
- `--dtc_repairable_filter_enabled`

Interpretation of HippoRAG JSON files:

- `overall_recomputed`: baseline top-5 QA/retrieval metrics before DtC.
- `setwise_selector_qa`: DtC method metrics and deltas versus baseline, evaluated in the same run.
- `setwise_selector_query_traces`: query-level DtC traces, selected positions, decomposition, coverage scores, repairable flags.

## Current DtC Main Result Artifacts

Repairable-filter pilot100, current strongest pilot:

- 2Wiki:
  `/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_2wikimultihopqa/eval_reports/dtc_embed_nvembed_rankw0p2_repairable_filter_pilot100_anchor2_8041.json`
- HotpotQA:
  `/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_hotpotqa/eval_reports/dtc_embed_nvembed_rankw0p2_repairable_filter_pilot100_anchor2_8042.json`
- MuSiQue:
  `/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_musique/eval_reports/dtc_embed_nvembed_rankw0p2_repairable_filter_pilot100_anchor2_8043.json`

Launch script:

- `/mnt/nvme/code/HippoRAG/run_logs/run_dtc_repairable_filter_pilot100_20260423.sh`

Run logs:

- `/mnt/nvme/code/HippoRAG/run_logs/dtc_repairable_filter_pilot100_20260423/2wikimultihopqa.log`
- `/mnt/nvme/code/HippoRAG/run_logs/dtc_repairable_filter_pilot100_20260423/hotpotqa.log`
- `/mnt/nvme/code/HippoRAG/run_logs/dtc_repairable_filter_pilot100_20260423/musique.log`

Pilot100 results:

| Dataset | Baseline EM | Baseline F1 | DtC Delta EM | DtC Delta F1 | DtC EM | DtC F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 0.4000 | 0.4531 | +0.0800 | +0.0946 | 0.4800 | 0.5477 |
| HotpotQA | 0.5700 | 0.6744 | +0.0200 | +0.0310 | 0.5900 | 0.7054 |
| MuSiQue | 0.3000 | 0.3852 | +0.0700 | +0.0632 | 0.3700 | 0.4484 |

Rank-regularized DtC full1000, previous main full run:

- 2Wiki:
  `/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_2wikimultihopqa/eval_reports/dtc_embed_nvembed_rankw0p2_limit1000_anchor2_8043.json`
- HotpotQA:
  `/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_hotpotqa/eval_reports/dtc_embed_nvembed_rankw0p2_limit1000_anchor2_8043.json`
- MuSiQue:
  `/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_musique/eval_reports/dtc_embed_nvembed_rankw0p2_limit1000_anchor2_8043.json`

Soft DtC non-instruction full1000, earlier full run:

- 2Wiki:
  `/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_2wikimultihopqa/eval_reports/dtc_embed_nvembed_noninstr_soft_limit1000_anchor2_8043.json`
- HotpotQA:
  `/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_hotpotqa/eval_reports/dtc_embed_nvembed_noninstr_soft_limit1000_anchor2_8042.json`
- MuSiQue:
  `/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_musique/eval_reports/dtc_embed_nvembed_noninstr_soft_limit1000_anchor2_8041.json`

## Other DtC Ablation Artifacts

Demand gate pilot100:

- Script:
  `/mnt/nvme/code/HippoRAG/run_logs/run_dtc_demand_gate_pilot100_20260423.sh`
- Log:
  `/mnt/nvme/code/HippoRAG/run_logs/dtc_demand_gate_pilot100_20260423.log`
- Outputs are in:
  `/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_{dataset}/eval_reports/dtc_embed_nvembed_rankw0p2_demandgate_a*_pilot100_anchor2_8043.json`

SC-SER pilot100:

- Script:
  `/mnt/nvme/code/HippoRAG/run_logs/run_sc_ser_pilot100_20260423.sh`
- Logs:
  `/mnt/nvme/code/HippoRAG/run_logs/sc_ser_pilot100_20260423/`
- Outputs:
  `/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_{dataset}/eval_reports/dtc_embed_nvembed_rankw0p2_ser_lam1p0_pilot100_anchor2_804*.json`

Repairable SC-SER pilot100:

- Script:
  `/mnt/nvme/code/HippoRAG/run_logs/run_sc_ser_repairable_pilot100_20260423.sh`
- Logs:
  `/mnt/nvme/code/HippoRAG/run_logs/sc_ser_repairable_pilot100_20260423/`
- Outputs:
  `/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_{dataset}/eval_reports/dtc_embed_nvembed_rankw0p2_ser_repairable_bind_lam1p0_pilot100_anchor2_804*.json`

Rank-weight sweep:

- Script:
  `/mnt/nvme/code/HippoRAG/run_logs/run_dtc_rank_sweep_pilot100_20260422.sh`
- Summary:
  `/mnt/nvme/code/HippoRAG/run_logs/dtc_rank_sweep_best_20260422.tsv`
- Full1000 summary:
  `/mnt/nvme/code/HippoRAG/run_logs/dtc_rank_best1000_summary_20260422.tsv`

## PropRAG Code To Audit

PropRAG entry point:

- `/mnt/nvme/code/PropRAG/main.py`

Main PropRAG pipeline:

- `/mnt/nvme/code/PropRAG/src/proprag/PropRAG.py`

OpenIE / proposition extraction:

- `/mnt/nvme/code/PropRAG/src/proprag/information_extraction/proposition_extraction.py`
- `/mnt/nvme/code/PropRAG/src/proprag/information_extraction/enhanced_openie.py`

LLM wrapper:

- `/mnt/nvme/code/PropRAG/src/proprag/llm/openai_gpt.py`

Important audit point: verify no-think is applied consistently to OpenIE/proposition extraction and QA, not only QA. Earlier dirty runs may have used think-mode assets; clean rebuild below should be the credible version.

Executable audit script and generated reports:

- Script:
  `/mnt/nvme/code/HippoRAG/scripts/audit_proprag_dtc_protocol.py`
- Protocol audit report:
  `/mnt/nvme/code/HippoRAG/docs/proprag_protocol_audit_report_20260423.md`
- Protocol audit JSON:
  `/mnt/nvme/code/HippoRAG/docs/proprag_protocol_audit_report_20260423.json`
- Pilot100 comparison report:
  `/mnt/nvme/code/HippoRAG/docs/proprag_dtc_pilot100_comparison_20260423.md`
- Pilot100 comparison JSON:
  `/mnt/nvme/code/HippoRAG/docs/proprag_dtc_pilot100_comparison_20260423.json`

## PropRAG Clean No-Think 100 Artifacts

Clean rebuild/run directory:

- `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_rebuild3_20260423/`

Launch script:

- `/mnt/nvme/code/PropRAG/run_logs/run_proprag_clean_nothink_rebuild3_20260423.sh`

Result JSONs:

- 2Wiki:
  `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_rebuild3_20260423/2wikimultihopqa.json`
- HotpotQA:
  `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_rebuild3_20260423/hotpotqa.json`
- MuSiQue:
  `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_rebuild3_20260423/musique.json`

Logs:

- `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_rebuild3_20260423/2wikimultihopqa.log`
- `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_rebuild3_20260423/hotpotqa.log`
- `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_rebuild3_20260423/musique.log`

Clean PropRAG-100 results:

| Dataset | EM | F1 | R@5 | R@20 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 0.6000 | 0.6468 | 0.9350 | 0.9625 | 0.9875 |
| HotpotQA | 0.5700 | 0.6912 | 0.9300 | 0.9950 | 1.0000 |
| MuSiQue | 0.4100 | 0.4630 | 0.6775 | 0.8900 | 0.9725 |

Clean PropRAG output assets are under:

- `/mnt/nvme/code/PropRAG/outputs_aligned_clean_nothink_top100_nvembed_rebuild_20260423/2wikimultihopqa/`
- `/mnt/nvme/code/PropRAG/outputs_aligned_clean_nothink_top100_nvembed_rebuild_20260423/hotpotqa/`
- `/mnt/nvme/code/PropRAG/outputs_aligned_clean_nothink_top100_nvembed_rebuild_20260423/musique/`

The concrete graph / embedding subdirs are named:

- `qwen3-8b-train_VLLM_nvidia_NV-Embed-v2/`

Inside each dataset directory, inspect:

- `openie_results_ner_qwen3-8b-train.json`
- `qwen3-8b-train_VLLM_nvidia_NV-Embed-v2/graph_0.8.graphml`
- `qwen3-8b-train_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet`
- `qwen3-8b-train_VLLM_nvidia_NV-Embed-v2/entity_embeddings/vdb_entity.parquet`
- `qwen3-8b-train_VLLM_nvidia_NV-Embed-v2/proposition_embeddings/vdb_proposition.parquet`

## PropRAG Clean No-Think Full1000 Artifacts

Current full1000 run directory:

- `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_full1000_20260423/`

Launch script:

- `/mnt/nvme/code/PropRAG/run_logs/run_proprag_clean_nothink_full1000_20260423.sh`

Logs:

- `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_full1000_20260423/2wikimultihopqa.log`
- `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_full1000_20260423/hotpotqa.log`
- `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_full1000_20260423/musique.log`

Status files:

- `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_full1000_20260423/2wikimultihopqa.status`
- `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_full1000_20260423/hotpotqa.status`
- `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_full1000_20260423/musique.status`

Expected result JSONs after completion:

- `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_full1000_20260423/2wikimultihopqa.json`
- `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_full1000_20260423/hotpotqa.json`
- `/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_full1000_20260423/musique.json`

As of 2026-04-23 14:34 CST, all three were still in retrieval:

- 2Wiki: about `202/1000`
- HotpotQA: about `180/1000`
- MuSiQue: about `99/1000`

No final full1000 JSON existed at that checkpoint.

## PropRAG vs DtC Pilot100 Comparison Snapshot

This table is useful for sanity checking, but should not be presented as same-pool comparison without caveat.

| Dataset | DtC Repairable Filter EM | DtC Repairable Filter F1 | PropRAG Clean EM | PropRAG Clean F1 | F1 Difference, DtC - PropRAG |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 0.4800 | 0.5477 | 0.6000 | 0.6468 | -0.0991 |
| HotpotQA | 0.5900 | 0.7054 | 0.5700 | 0.6912 | +0.0142 |
| MuSiQue | 0.3700 | 0.4484 | 0.4100 | 0.4630 | -0.0146 |

## Protocol Alignment Checklist

Shared or intended shared conditions:

- Dataset names: `2wikimultihopqa`, `hotpotqa`, `musique`
- Limit for pilot: `100`
- Limit for full run: `1000`
- Reader: `qwen3-8b-train`
- Reader endpoints: `http://localhost:8041/v1`, `8042`, `8043`
- Embedding model: `VLLM/nvidia/NV-Embed-v2`
- Embedding endpoint: `http://localhost:8019/v1/embeddings`
- Final QA budget: `qa_top_k=5`
- Candidate retrieval budget: `retrieval_top_k=100` / `setwise_pool_k=100`
- Generation budget: `max_new_tokens=2048`
- No-think mode intended for Qwen3 generation.

Non-aligned or audit-sensitive conditions:

- Candidate pool is not identical:
  PropRAG retrieves via its proposition graph; DtC selects from HippoRAG's retrieved top-100 pool.
- Retrieval system is not identical:
  PropRAG is a separate retriever / graph construction method, not a setwise selector over HippoRAG's pool.
- PropRAG uses proposition extraction assets created by OpenIE; DtC uses decomposition at eval time and HippoRAG assets.
- Prompt templates and context formatting may differ.
- PropRAG logs say `Retrieving with gold docs for evaluation`; audit should confirm `gold_docs` is only used for recall scoring and is not injected into retrieval or QA prompts.
- Query subset/order must be checked: confirm PropRAG's first 100 examples exactly match HippoRAG's first 100 for each dataset.
- Corpus/chunking may differ across repos; this affects Recall@K comparability.

## Specific Audit Tasks For Claude

1. Verify no-think enforcement in PropRAG:
   inspect `PropRAG.py`, `information_extraction/proposition_extraction.py`, `information_extraction/enhanced_openie.py`, `openai_gpt.py`, and clean rebuild logs.

2. Verify PropRAG does not leak gold supporting docs:
   inspect how `gold_docs` is passed through retrieval and QA in `main.py` and `src/proprag/PropRAG.py`.

3. Verify pilot100 query alignment:
   compare query strings and IDs in PropRAG JSON `examples` against HippoRAG DtC JSON `examples`.

4. Verify metric definitions:
   compare PropRAG `retrieval` / `qa` metrics with HippoRAG `overall_recomputed` / `setwise_selector_qa`.

5. Verify fixed-pool status:
   determine whether PropRAG ever consumes HippoRAG's top-100 pool. Current expectation: no.

6. Verify reader prompt/context budget:
   compare PropRAG QA prompt with HippoRAG reader prompt and confirm both use five passages and no-think.

7. Decide the valid comparison label:
   likely label is `cross-system same-reader/same-embedding baseline`, not `same-pool composition baseline`.

## Quick Commands

Inspect clean PropRAG-100 metrics:

```bash
python - <<'PY'
import json
for p in [
  '/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_rebuild3_20260423/2wikimultihopqa.json',
  '/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_rebuild3_20260423/hotpotqa.json',
  '/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_rebuild3_20260423/musique.json',
]:
    d=json.load(open(p))
    print(p, d['qa'], {k:d['retrieval'][k] for k in ['Recall@5','Recall@20','Recall@100']})
PY
```

Inspect DtC repairable-filter pilot100 metrics:

```bash
python - <<'PY'
import json, glob
for p in sorted(glob.glob('/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_*/eval_reports/dtc_embed_nvembed_rankw0p2_repairable_filter_pilot100_anchor2_804*.json')):
    d=json.load(open(p))
    print(p)
    print('baseline', d['overall_recomputed']['ExactMatch'], d['overall_recomputed']['F1'])
    print('selector', d['setwise_selector_qa'])
PY
```

Monitor PropRAG full1000:

```bash
for f in /mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_full1000_20260423/*.status; do
  echo "==== $f"
  cat "$f"
done
tail -n 50 /mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_full1000_20260423/2wikimultihopqa.log
tail -n 50 /mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_full1000_20260423/hotpotqa.log
tail -n 50 /mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_full1000_20260423/musique.log
```
