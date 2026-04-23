# DtC Repairable-Filter Full1000 Freeze

This document freezes the current HippoRAG-based DtC reference line before the
`satisfiable_by` parser-contract cleanup.

## Scope

- Method: `dtc_embed` with repairable filter + dependency binding + rank regularization
- Reader: `qwen3-8b-train`
- Embedding: `VLLM/nvidia/NV-Embed-v2`
- Setting: fixed `top-100` pool, final `top-5` evidence set
- Launcher: [run_dtc_repairable_filter_full1000_20260423.sh](/mnt/nvme/code/HippoRAG/run_logs/run_dtc_repairable_filter_full1000_20260423.sh)

## Frozen Full1000 Results

| Dataset | Baseline EM | Baseline F1 | DtC EM | DtC F1 | Delta EM | Delta F1 | Base R@5 | DtC R@5 | Base R@20 | DtC R@20 | Base R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 0.4750 | 0.5407 | 0.5380 | 0.6105 | +0.0630 | +0.0698 | 0.8315 | 0.8850 | 0.9048 | 0.9237 | 0.9545 |
| HotpotQA | 0.5730 | 0.7043 | 0.5970 | 0.7316 | +0.0240 | +0.0273 | 0.9230 | 0.9475 | 0.9885 | 0.9915 | 0.9965 |
| MuSiQue | 0.3070 | 0.4058 | 0.3340 | 0.4305 | +0.0270 | +0.0247 | 0.6778 | 0.7133 | 0.8558 | 0.8652 | 0.9347 |

## Frozen Output Files

- 2Wiki:
  [dtc_embed_nvembed_rankw0p2_repairable_filter_limit1000_anchor2_8041.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_2wikimultihopqa/eval_reports/dtc_embed_nvembed_rankw0p2_repairable_filter_limit1000_anchor2_8041.json)
- HotpotQA:
  [dtc_embed_nvembed_rankw0p2_repairable_filter_limit1000_anchor2_8042.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_hotpotqa/eval_reports/dtc_embed_nvembed_rankw0p2_repairable_filter_limit1000_anchor2_8042.json)
- MuSiQue:
  [dtc_embed_nvembed_rankw0p2_repairable_filter_limit1000_anchor2_8043.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_nvembed_musique/eval_reports/dtc_embed_nvembed_rankw0p2_repairable_filter_limit1000_anchor2_8043.json)

## Frozen Method Configuration

The current reference run uses:

```bash
--setwise_selector dtc_embed
--setwise_pool_k 100
--qa_top_k 5
--setwise_anchor_count 2
--setwise_reserve_top_m 0
--setwise_non_anchor_title_dedup true
--dtc_decomposition_mode llm
--dtc_enforce_dependencies true
--dtc_require_new_crossing false
--dtc_repairable_filter_enabled true
--dtc_enable_dependency_binding true
--dtc_binding_max_candidates 4
--dtc_binding_entity_hit_required true
--dtc_max_steps 4
--dtc_match_threshold 0.35
--dtc_redundancy_weight 0.10
--dtc_base_weight 0.05
--dtc_rank_weight 0.2
--dtc_anchor_bonus_weight 0.10
--dtc_dependency_bonus_weight 0.10
--dtc_max_completion_tokens 512
--dtc_ser_enabled false
```

## Notes

- This freeze point still uses the legacy regex-based inference-only fallback in
  `scripts/dtc_embed_utils.py`.
- The next cleanup step is to move the main repairability typing into the
  decomposition output via `satisfiable_by=document|inference`, while keeping
  regex only as a backward-compatible fallback.
- This document freezes the current reference numbers so future parser-contract
  cleanups can be evaluated against a stable baseline.
