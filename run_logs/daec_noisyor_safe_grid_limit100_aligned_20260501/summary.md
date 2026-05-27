# DAEC-L1-Safe Aligned Limit100 Grid Summary

Date: 2026-05-01

Protocol:
- Candidate pool: `run_logs/proprag_pool_exports_full1000_20260424/{dataset}_pool100.json`
- Reader: Qwen3-8B endpoint per dataset, no-think forced via `HIPPORAG_RERANK_FORCE_NO_THINK=1`
- Selector: `daec_noisyor_safe`
- Pool: `setwise_pool_k=100`
- Reader budget: `qa_top_k=5`, `qa_doc_max_chars=2048`
- Demand decomposition: `dtc_decomposition_mode=llm`, `dtc_binding_max_candidates=5`

Baseline aligned limit100 results are from `reports/dpathrag/prop_neocorr_daec_limit100_20260429.json`.

## Main Comparison

| Dataset | Method / Config | R@5 | R@20 | EM | F1 |
|---|---|---:|---:|---:|---:|
| 2wikimultihopqa | Prop | 0.9350 | 0.9625 | 0.5800 | 0.6318 |
| 2wikimultihopqa | Prop+DAEC | 0.9525 | 0.9675 | 0.5800 | 0.6435 |
| 2wikimultihopqa | Prop+DAEC-L1 | 0.9375 | 0.9650 | 0.5700 | 0.6164 |
| 2wikimultihopqa | Safe `s1_p1_g002` | 0.9175 | 0.9650 | 0.5500 | 0.5942 |
| 2wikimultihopqa | Safe `s1_p2_g005` | 0.9350 | 0.9625 | 0.5600 | 0.6118 |
| 2wikimultihopqa | Safe `s1_p2_g010` | 0.9375 | 0.9625 | 0.5600 | 0.6118 |
| hotpotqa | Prop | 0.9300 | 0.9950 | 0.5700 | 0.6912 |
| hotpotqa | Prop+DAEC | 0.9500 | 0.9950 | 0.5700 | 0.6912 |
| hotpotqa | Prop+DAEC-L1 | 0.9500 | 0.9950 | 0.6000 | 0.7079 |
| hotpotqa | Safe `s1_p1_g002` | 0.9150 | 0.9950 | 0.5600 | 0.6694 |
| hotpotqa | Safe `s1_p2_g005` | 0.9400 | 0.9950 | 0.5800 | 0.6952 |
| hotpotqa | Safe `s1_p2_g010` | 0.9350 | 0.9950 | 0.5700 | 0.6912 |
| musique | Prop | 0.6775 | 0.8900 | 0.3800 | 0.4374 |
| musique | Prop+DAEC | 0.7175 | 0.8983 | 0.3900 | 0.4660 |
| musique | Prop+DAEC-L1 | 0.6600 | 0.9058 | 0.3700 | 0.4202 |
| musique | Safe `s1_p1_g002` | 0.6450 | 0.8958 | 0.3700 | 0.4476 |
| musique | Safe `s1_p2_g005` | 0.6742 | 0.8958 | 0.3300 | 0.4063 |
| musique | Safe `s1_p2_g010` | 0.6750 | 0.8925 | 0.3200 | 0.3989 |

Config names:
- `s1_p1_g002`: `max_swaps=1`, `preserve_top_m=1`, `min_objective_gain=0.02`
- `s1_p2_g005`: `max_swaps=1`, `preserve_top_m=2`, `min_objective_gain=0.05`
- `s1_p2_g010`: `max_swaps=1`, `preserve_top_m=2`, `min_objective_gain=0.10`
- All three use `min_swap_gain=0.01`, `retriever_margin_threshold=1.01`.

## Safe Diagnostics

| Dataset | Config | Changed | Decisions | EM Delta vs Prop | F1 Delta vs Prop | Query gains/reg/same |
|---|---|---:|---|---:|---:|---:|
| 2wikimultihopqa | `s1_p1_g002` | 36 | fallback_low_rebuild_gain:64, minimal_edit_applied:36 | -0.0300 | -0.0376 | 1/4/95 |
| 2wikimultihopqa | `s1_p2_g005` | 29 | fallback_low_rebuild_gain:71, minimal_edit_applied:29 | -0.0200 | -0.0200 | 0/2/98 |
| 2wikimultihopqa | `s1_p2_g010` | 23 | fallback_low_rebuild_gain:77, minimal_edit_applied:23 | -0.0200 | -0.0200 | 0/2/98 |
| hotpotqa | `s1_p1_g002` | 19 | fallback_low_rebuild_gain:81, minimal_edit_applied:19 | -0.0100 | -0.0218 | 1/2/97 |
| hotpotqa | `s1_p2_g005` | 18 | fallback_low_rebuild_gain:82, minimal_edit_applied:18 | +0.0100 | +0.0040 | 1/0/99 |
| hotpotqa | `s1_p2_g010` | 17 | fallback_low_rebuild_gain:83, minimal_edit_applied:17 | +0.0000 | +0.0000 | 0/0/100 |
| musique | `s1_p1_g002` | 63 | fallback_low_rebuild_gain:37, minimal_edit_applied:63 | -0.0100 | +0.0102 | 5/6/89 |
| musique | `s1_p2_g005` | 60 | fallback_low_rebuild_gain:40, minimal_edit_applied:60 | -0.0500 | -0.0311 | 3/8/89 |
| musique | `s1_p2_g010` | 50 | fallback_low_rebuild_gain:50, minimal_edit_applied:50 | -0.0600 | -0.0385 | 2/8/90 |

## Judgment

Do not scale current DAEC-L1-Safe variants to full1000.

The safe projection reduces some L1 damage, but it does not beat the already available DAEC baseline:
- On 2Wiki, all Safe configs are worse than Prop and worse than DAEC.
- On HotpotQA, the best Safe config only gives a small gain over Prop and is still below DAEC-L1.
- On MuSiQue, the least bad Safe config improves F1 over Prop but loses EM and remains below original DAEC F1.

The failure mode is not excessive two-hop swapping. Even with `max_swaps=1`, accepted swaps still create more regressions than gains on 2Wiki, and conservative thresholds erase MuSiQue gains before recovering cross-dataset stability.

Next useful DAEC enhancement should change the admission signal, not only the safe projection thresholds. The likely next branch is sentence-level demand scoring or relation-aware admission; threshold-only tuning is not a promising route.
