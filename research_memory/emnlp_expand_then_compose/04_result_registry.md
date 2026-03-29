# Result Registry

Last updated: 2026-03-29

This file stores concrete results only. Every result must have a file path.

## Non-Oracle Pilot Summary

Summary note:
- `research_memory/emnlp_expand_then_compose/05_non_oracle_bridge_greedy.md`

Key status:
- The minimal non-oracle method is now implemented.
- `2Wiki` pilot already shows a positive signal.
- `MuSiQue` pilot is running but not complete yet.

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

## Cross-Dataset Summary

Summary files:
- `research_memory/emnlp_expand_then_compose/oracle_select_cross_dataset_summary_20260329_resume.md`
- `research_memory/emnlp_expand_then_compose/oracle_select_cross_dataset_summary_20260329_resume.json`

Takeaways:
- `2Wiki` is the main mechanism dataset: it has the bridge diagnosis and the sharp `2-doc` vs `4-doc` split.
- `MuSiQue` is the main hard-generalization dataset: support depth grows from `2-doc` to `4-doc`, and oracle gains keep growing with larger pools.
- `HotpotQA` is the shallow contrast dataset: nearly all support is already near the top, so larger pools still help but much less.
