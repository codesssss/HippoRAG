# Result Registry

Last updated: 2026-03-30

This file stores concrete results only. Every result must have a file path.

## Non-Oracle Pilot Summary

Summary note:
- `research_memory/emnlp_expand_then_compose/05_non_oracle_bridge_greedy.md`
- `research_memory/emnlp_expand_then_compose/06_bridge_beam_search.md`

Key status:
- The minimal non-oracle method is now implemented.
- There is a positive small-sample signal on `2Wiki` under the earlier `general_relation_graph` path.
- On the canonical `legacy_fact_graph` backbone, `bridge_beam` now gives a stronger positive `2Wiki` signal than `bridge_greedy`.
- `MuSiQue bridge_beam@100` is now a negative overall result despite a positive `4-doc` bucket signal.
- The main remaining question is whether `beam` still beats matched `greedy` on `MuSiQue`, or whether the score itself is the limiting factor.
- The failed `seed union + title dedup` tweak has been recorded and reverted.

## 2WikiMultihopQA-1000

Reports:
- `outputs_step0_general_2wikimultihopqa/eval_reports/oracle_select_sweep_1000_reorder20.json`
- `outputs_step0_general_2wikimultihopqa/eval_reports/oracle_select_sweep_1000_reorder20.oracle_analysis.json`
- `outputs_step0_general_2wikimultihopqa/eval_reports/oracle_select_sweep_1000_reorder20.oracle_analysis.md`

Baseline:
- `EM = 0.436`
- `F1 = 0.4916`

Oracle reorder@20:
- `EM = 0.498`
- `F1 = 0.5643`
- `Delta EM = +0.062`
- `Full support in top-20 = 0.659`

Oracle-select sweep:

| K | EM | Delta EM | F1 | Delta F1 | FS in pool |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.498 | +0.062 | 0.5643 | +0.0727 | 0.659 |
| 30 | 0.512 | +0.076 | 0.5812 | +0.0896 | 0.682 |
| 50 | 0.531 | +0.095 | 0.6020 | +0.1104 | 0.708 |
| 100 | 0.560 | +0.124 | 0.6318 | +0.1402 | 0.752 |

Support depth:
- `2-doc`: `count=765`, `fully_supported=726`, `median=3`, `mean=12.8`, `p90=27`
- `4-doc`: `count=235`, `fully_supported=77`, `median=64`, `mean=76.9`, `p90=166`

Bucket breakdown at `K=100`:
- `2-doc`: `FS=0.9190`, `EM=0.6039`, `F1=0.6870`
- `4-doc`: `FS=0.2085`, `EM=0.4170`, `F1=0.4521`

Bridge vs anchor diagnosis:
- anchor `Recall@20 = 0.9995`
- bridge `Recall@20 = 0.5997`
- anchor median depth `= 1.0`
- bridge median depth `= 6.0`
- bridge mean depth `= 29.3`
- bridge P90 depth `= 100.6`
- partial top-5 failures that are bridgeable `= 0.7115`
- missing top-5 gold docs that are bridgeable `= 0.7265`

Interpretation:
- This is the strongest evidence for the structural bottleneck story.
- Easy queries saturate early; hard queries require much deeper support pools.
- The bridge bottleneck is visible directly in recall and depth asymmetry.

Non-oracle pilot:
- report path: `outputs_step0_general_2wikimultihopqa/eval_reports/setwise_bridge_greedy_pilot_20_v2.json`
- setup: `limit=20`, `pool_k=100`, selector `bridge_greedy`
- baseline `EM/F1 = 0.5000 / 0.5000`
- selector `EM/F1 = 0.6000 / 0.6375`
- delta `EM/F1 = +0.1000 / +0.1375`
- baseline `Recall@5 = 0.7125`, selector `Recall@5 = 0.8125`
- baseline `Recall@20 = 0.8125`, selector `Recall@20 = 0.9500`
- `2-doc EM delta = +0.0667`
- `4-doc EM delta = +0.2000`

Canonical-backbone non-oracle pilot:
- report path: `outputs_step0_general_2wikimultihopqa/eval_reports/setwise_bridge_greedy_pilot_20_legacy.json`
- setup: `limit=20`, `pool_k=100`, selector `bridge_greedy`, base mode `legacy_fact_graph`
- baseline `EM/F1 = 0.5000 / 0.5523`
- selector `EM/F1 = 0.5500 / 0.5774`
- delta `EM/F1 = +0.0500 / +0.0251`
- baseline `Recall@5 = 0.8375`, selector `Recall@5 = 0.8250`
- baseline `Recall@20 = 0.8875`, selector `Recall@20 = 0.8875`
- `2-doc EM delta = +0.1333`
- `4-doc EM delta = -0.2000`

Canonical-backbone search ablation:
- greedy report path: `outputs_step0_general_2wikimultihopqa/eval_reports/setwise_bridge_greedy_pilot_100_legacy.json`
- beam report path: `outputs_step0_general_2wikimultihopqa/eval_reports/setwise_bridge_beam_pilot_100_legacy.json`
- setup: `limit=100`, `pool_k=100`, anchor `2`, base mode `legacy_fact_graph`
- baseline `EM/F1 = 0.3800 / 0.4332`
- greedy `EM/F1 = 0.4300 / 0.4726`
- beam `EM/F1 = 0.4600 / 0.4924`
- greedy delta `EM/F1 = +0.0500 / +0.0394`
- beam delta `EM/F1 = +0.0800 / +0.0592`
- greedy `Recall@5 / Recall@20 = 0.8150 / 0.8825`
- beam `Recall@5 / Recall@20 = 0.8075 / 0.8825`
- greedy `2-doc EM delta = +0.0909`
- beam `2-doc EM delta = +0.0909`
- greedy `4-doc EM delta = -0.0870`
- beam `4-doc EM delta = +0.0435`

Failed micro-tweak:
- report path: `outputs_step0_general_2wikimultihopqa/eval_reports/setwise_bridge_greedy_pilot_20_legacy_seedunion_dedup.json`
- tweak: union `fact seeds` with `lexical seeds` and add title dedup
- delta `EM/F1 = -0.0500 / -0.0647`
- `4-doc EM delta = -0.4000`
- status: reverted from code

## HotpotQA-1000

Report:
- `outputs_step0_general_hotpotqa/eval_reports/oracle_select_sweep_1000_reorder20.json`

Baseline:
- `EM = 0.563`
- `F1 = 0.6760`

Oracle reorder@20:
- `EM = 0.631`
- `F1 = 0.7548`
- `Delta EM = +0.068`
- `Full support in top-20 = 0.950`

Oracle-select sweep:

| K | EM | Delta EM | F1 | Delta F1 | FS in pool |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.631 | +0.068 | 0.7548 | +0.0788 | 0.950 |
| 30 | 0.637 | +0.074 | 0.7606 | +0.0846 | 0.963 |
| 50 | 0.642 | +0.079 | 0.7658 | +0.0898 | 0.973 |
| 100 | 0.645 | +0.082 | 0.7707 | +0.0947 | 0.986 |

Support depth:
- `2-doc`: `count=1000`, `fully_supported=992`, `median=3`, `mean=6.2`, `p90=9`, `max=187`

Interpretation:
- Cross-dataset oracle ceiling generalizes.
- However, this dataset is all `2-doc`, so it acts as a shallow contrast case rather than a hard structural analysis dataset.

## MuSiQue-1000

Report:
- `outputs_step0_general_musique/eval_reports/oracle_select_sweep_1000.json`

Baseline:
- `EM = 0.257`
- `F1 = 0.3427`

Oracle reorder@20:
- `EM = 0.363`
- `F1 = 0.4612`
- `Delta EM = +0.106`
- `Full support in top-20 = 0.531`

Oracle-select sweep:

| K | EM | Delta EM | F1 | Delta F1 | FS in pool |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.363 | +0.106 | 0.4612 | +0.1185 | 0.531 |
| 30 | 0.379 | +0.122 | 0.4807 | +0.1380 | 0.591 |
| 50 | 0.403 | +0.146 | 0.5071 | +0.1644 | 0.661 |
| 100 | 0.436 | +0.179 | 0.5401 | +0.1974 | 0.757 |

Support depth:
- `2-doc`: `count=518`, `fully_supported=475`, `median=5`, `mean=18.0`, `p90=48.6`
- `3-doc`: `count=316`, `fully_supported=266`, `median=18`, `mean=39.4`, `p90=116.0`
- `4-doc`: `count=166`, `fully_supported=91`, `median=55`, `mean=69.2`, `p90=152.0`

Bucket breakdown at `K=100`:
- `2-doc`: `FS=0.8784`, `EM=0.4846`, `F1=0.5942`
- `3-doc`: `FS=0.7405`, `EM=0.4304`, `F1=0.5344`
- `4-doc`: `FS=0.4096`, `EM=0.2952`, `F1=0.3823`

Interpretation:
- This is the main cross-dataset generalization result for the hard-query story.
- Support depth increases monotonically with query composition difficulty, and the oracle headroom exceeds both `2Wiki` and `HotpotQA`.

Canonical-backbone non-oracle pilot:
- report path: `outputs_step0_general_musique/eval_reports/setwise_bridge_greedy_pilot_20_legacy.json`
- setup: `limit=20`, `pool_k=100`, selector `bridge_greedy`, base mode `legacy_fact_graph`
- baseline `EM/F1 = 0.3000 / 0.3250`
- selector `EM/F1 = 0.1500 / 0.1559`
- delta `EM/F1 = -0.1500 / -0.1691`
- baseline `Recall@5 = 0.5417`, selector `Recall@5 = 0.4333`
- baseline `Recall@20 = 0.6917`, selector `Recall@20 = 0.6917`
- `2-doc EM delta = -0.2727`
- `3-doc EM delta = 0.0000`
- `4-doc EM delta = 0.0000`

Canonical-backbone beam pilot:
- report path: `outputs_step0_general_musique/eval_reports/setwise_bridge_beam_pilot_100_legacy.json`
- setup: `limit=100`, `pool_k=100`, selector `bridge_beam`, anchor `2`, base mode `legacy_fact_graph`
- baseline `EM/F1 = 0.2600 / 0.3466`
- selector `EM/F1 = 0.1800 / 0.2498`
- delta `EM/F1 = -0.0800 / -0.0968`
- baseline `Recall@5 / Recall@20 = 0.6150 / 0.7858`
- beam `Recall@5 / Recall@20 = 0.4950 / 0.7933`
- `2-doc EM delta = -0.1458`
- `3-doc EM delta = -0.0667`
- `4-doc EM delta = +0.0455`

Interpretation:
- The first matched `MuSiQue` beam run is not paper-ready as a practical method result.
- It helps the hardest `4-doc` bucket but hurts `2-doc` and `3-doc` enough to become destructive overall.
- This currently supports a narrow hard-case mechanism story, not a broad robustness claim.

## Cross-Dataset Summary

Summary files:
- `research_memory/emnlp_expand_then_compose/oracle_select_cross_dataset_summary_20260329_resume.md`
- `research_memory/emnlp_expand_then_compose/oracle_select_cross_dataset_summary_20260329_resume.json`

Takeaways:
- `2Wiki` is the main mechanism dataset: it has the bridge diagnosis and the sharp `2-doc` vs `4-doc` split.
- `MuSiQue` is the main hard-generalization dataset: support depth grows from `2-doc` to `4-doc`, and oracle gains keep growing with larger pools.
- `HotpotQA` is the shallow contrast dataset: nearly all support is already near the top, so larger pools still help but much less.
