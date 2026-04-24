# Result Registry

Last updated: 2026-04-24

This file stores concrete results only. Every result must have a file path.

## Non-Oracle Pilot Summary

Summary note:
- `research_memory/emnlp_expand_then_compose/05_non_oracle_bridge_greedy.md`
- `research_memory/emnlp_expand_then_compose/06_bridge_beam_search.md`

Key status:
- The minimal non-oracle method is now implemented.
- There is a positive small-sample signal on `2Wiki` under the earlier `general_relation_graph` path.
- On the canonical `legacy_fact_graph` backbone, `bridge_beam` now gives a stronger positive `2Wiki` signal than `bridge_greedy`.
- The strongest `2Wiki` result still comes from the more aggressive beam setting, but that version does not transfer safely across datasets.
- The new shared closure-only selector keeps a positive `2Wiki` gain while removing the earlier `HotpotQA` failure and neutralizing the overall `MuSiQue` loss.
- The tradeoff is that the closure-only selector is more conservative on the hardest cases, especially `2Wiki 4-doc`.
- The failed `seed union + title dedup` tweak has been recorded and reverted.
- A separate strongest-sidecar backup line now exists:
  - positive `2Wiki` smoke signal
  - stabilizing but still sub-baseline `HotpotQA` end-to-end behavior
  - see the report paths below
- `DtC-Embed` is now the active fixed-pool composition method candidate:
  - aligned NV-Embed non-instruction full1000 is positive on all three datasets
  - hard-crossing ablation is negative as a main fix
  - Layer-1 retriever-agnostic runs are positive on both PropRAG top-100 pools and dense NV-Embed top-100 pools
  - 2Wiki PropRAG-pool ablation supports dependency binding, but does not support rank prior or repair typing as necessary components

## Fixed-Pool Composition Baselines and DtC

Diversity baseline memo:
- `research_memory/emnlp_expand_then_compose/11_diversity_selector_baselines_20260421.md`

DtC direction memo:
- `research_memory/emnlp_expand_then_compose/14_dtc_nv_full1000_rank_prior_20260422.md`

Demand gate ablation memo:
- `research_memory/emnlp_expand_then_compose/15_demand_gate_ablation_20260423.md`

Satisfiable-by prompt sweep memo:
- `research_memory/emnlp_expand_then_compose/satisfiable_by_prompt_sweep_20260424.md`

Layer-1 retriever-agnostic composition memo:
- `research_memory/emnlp_expand_then_compose/16_layer1_retriever_agnostic_composition_20260424.md`

Layer-1 follow-up memo:
- `research_memory/emnlp_expand_then_compose/17_layer1_followup_taxonomy_significance_20260424.md`

Structure-blind diversity baselines:

| Dataset | Baseline F1 | MMR F1 | DPP F1 | Oracle@100 F1 |
|---|---:|---:|---:|---:|
| 2Wiki | 0.4868 | 0.4805 | 0.4863 | 0.6318 |
| HotpotQA | 0.6839 | 0.6830 | 0.6829 | 0.7707 |
| MuSiQue | 0.3439 | 0.3545 | 0.3569 | 0.5401 |

Interpretation:
- MMR/DPP do not recover the fixed-pool oracle gap.
- Generic embedding diversity is not enough for the `Assemble` part of `Expand-then-Compose`.

Aligned NV-Embed non-instruction DtC full1000:

| Dataset | Report | Baseline EM/F1 | DtC EM/F1 | Delta EM/F1 |
|---|---|---:|---:|---:|
| 2Wiki | `outputs_step0_general_nvembed_2wikimultihopqa/eval_reports/dtc_embed_nvembed_noninstr_soft_limit1000_anchor2_8043.json` | 0.475 / 0.5413 | 0.481 / 0.5495 | +0.006 / +0.0082 |
| HotpotQA | `outputs_step0_general_nvembed_hotpotqa/eval_reports/dtc_embed_nvembed_noninstr_soft_limit1000_anchor2_8042.json` | 0.574 / 0.7050 | 0.590 / 0.7225 | +0.016 / +0.0175 |
| MuSiQue | `outputs_step0_general_nvembed_musique/eval_reports/dtc_embed_nvembed_noninstr_soft_limit1000_anchor2_8041.json` | 0.307 / 0.4051 | 0.332 / 0.4296 | +0.025 / +0.0245 |

Hard-crossing pilot100 ablation:

| Dataset | Soft Delta F1 | Hard-Cross Delta F1 | Result |
|---|---:|---:|---|
| 2Wiki | +0.0256 | +0.0157 | weaker |
| HotpotQA | +0.0310 | +0.0010 | much weaker |
| MuSiQue | +0.0497 | +0.0023 | much weaker |

Interpretation:
- Hard-crossing removes wins but does not remove losses.
- Losses are better explained by false-positive requirement coverage, especially from deeper pool positions.
- The next method should regularize coverage by base retriever rank instead of adding a hard acceptance gate.

Demand-gate pilot100 ablation:

| Dataset | No-Gate Delta F1 | Best Gate Delta F1 | Best Gate Wins/Losses | Interpretation |
|---|---:|---:|---:|---|
| 2Wiki | +0.0456 | +0.0382 | 6 / 1 | reduces losses but loses wins |
| HotpotQA | +0.0310 | +0.0050 | 1 / 0 | over-abstains |
| MuSiQue | +0.0397 | +0.0034 | 4 / 4 | over-abstains badly |

Interpretation:
- A hard compose-or-preserve gate is too conservative as the main method.
- Gate remains useful as a diagnostic and negative ablation.
- The next method should treat composition as conservative repair: continuous residual-demand reward plus sufficiency-calibrated baseline preservation, rather than binary abstention.

Satisfiable-by prompt/policy sweep:

| Variant | Avg Selector F1 | Interpretation |
|---|---:|---|
| `old_repairable` | 0.5672 | best validated prompt |
| `sat_by_strict` | 0.5587 | too strict; loses dependent evidence |
| `binding_override` | 0.5601 | safer than strict but still below old prompt |
| `grounded_override` | 0.5616 | best schema-based variant, still below old prompt |
| `regex_only` | 0.5590 | shows the prompt/schema perturbation itself causes most of the drop |

Interpretation:
- Keep `satisfiable_by` parsing and downstream policy support in code.
- Default `--dtc_include_satisfiable_by=false` for main experiments to preserve the validated decomposition prompt.
- Use `--dtc_include_satisfiable_by=true` only for diagnostic/appendix taxonomy runs unless later evidence reverses this result.

Layer-1 PropRAG-pool + DAEC/DtC full1000:

| Dataset | Report | Base EM/F1 | DAEC EM/F1 | Delta EM/F1 | Oracle F1 | R@5 / R@20 / R@100 |
|---|---|---:|---:|---:|---:|---:|
| 2Wiki | `run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json` | 0.5750 / 0.6457 | 0.6070 / 0.6810 | +0.0320 / +0.0353 | 0.7311 | 0.9028 / 0.9607 / 0.9872 |
| HotpotQA | `run_logs/layer1_proprag_pool_eval_fixed_20260424/hotpotqa_proprag_pool_daec_oracle.json` | 0.5950 / 0.7227 | 0.6060 / 0.7348 | +0.0110 / +0.0121 | 0.7697 | 0.9500 / 0.9925 / 0.9990 |
| MuSiQue | `run_logs/layer1_proprag_pool_eval_fixed_20260424/musique_proprag_pool_daec_oracle.json` | 0.3300 / 0.4266 | 0.3390 / 0.4404 | +0.0090 / +0.0138 | 0.6019 | 0.7131 / 0.8942 / 0.9677 |

Layer-1 dense-pool + DAEC/DtC full1000:

| Dataset | Report | Base EM/F1 | DAEC EM/F1 | Delta EM/F1 | Oracle F1 | R@5 / R@20 / R@100 |
|---|---|---:|---:|---:|---:|---:|
| 2Wiki | `run_logs/layer1_dense_pool_eval_20260424/2wikimultihopqa_dense_pool_daec_oracle.json` | 0.4550 / 0.4984 | 0.5060 / 0.5628 | +0.0510 / +0.0644 | 0.6396 | 0.7238 / 0.7990 / 0.8770 |
| HotpotQA | `run_logs/layer1_dense_pool_eval_20260424/hotpotqa_dense_pool_daec_oracle.json` | 0.5950 / 0.7106 | 0.6140 / 0.7326 | +0.0190 / +0.0220 | 0.7669 | 0.9305 / 0.9830 / 0.9925 |
| MuSiQue | `run_logs/layer1_dense_pool_eval_20260424/musique_dense_pool_daec_oracle.json` | 0.2980 / 0.3896 | 0.3160 / 0.4138 | +0.0180 / +0.0242 | 0.5521 | 0.6628 / 0.8197 / 0.9055 |

Layer-1 2Wiki PropRAG-pool internal ablation:

| Variant | Report | Selector EM/F1 | Delta EM/F1 | Interpretation |
|---|---|---:|---:|---|
| Full | `run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json` | 0.6070 / 0.6810 | +0.0320 / +0.0353 | current conservative full config |
| `nobinding` | `run_logs/layer1_proprag_pool_ablation_2wiki_20260424/2wikimultihopqa_nobinding.json` | 0.5900 / 0.6599 | +0.0150 / +0.0142 | dependency binding is load-bearing |
| `norepairtyping` | `run_logs/layer1_proprag_pool_ablation_2wiki_20260424/2wikimultihopqa_norepairtyping.json` | 0.6110 / 0.6841 | +0.0360 / +0.0384 | repair typing not supported as necessary on 2Wiki |
| `norank` | `run_logs/layer1_proprag_pool_ablation_2wiki_20260424/2wikimultihopqa_norank.json` | 0.6110 / 0.6846 | +0.0360 / +0.0389 | rank prior not supported as necessary on 2Wiki |

Interpretation:
- DAEC/DtC improves all three datasets on both PropRAG top-100 and dense NV-Embed top-100 pools.
- This supports a retriever-agnostic fixed-pool composition claim, not only a HippoRAG-specific selector claim.
- Dependency binding is the strongest supported component in the current ablation evidence.
- Rank prior and repair typing should not be overclaimed until cross-dataset ablations support them.
- MuSiQue has substantial oracle headroom but weak recovery, so residual failure analysis is now higher priority than adding another gate.

Layer-1 paired significance:

Source:
- `run_logs/layer1_significance_20260424/summary.md`

| Dataset | Pool | Delta F1 | 95% CI | F1 p-value | Delta EM | 95% CI | EM p-value |
|---|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | PropRAG | +0.0353 | [+0.0230, +0.0480] | 0.0001 | +0.0320 | [+0.0190, +0.0450] | 0.0001 |
| HotpotQA | PropRAG | +0.0121 | [+0.0047, +0.0200] | 0.0021 | +0.0110 | [+0.0040, +0.0190] | 0.0074 |
| MuSiQue | PropRAG | +0.0138 | [+0.0012, +0.0266] | 0.0352 | +0.0090 | [-0.0040, +0.0220] | 0.2138 |
| 2Wiki | Dense | +0.0644 | [+0.0489, +0.0802] | 0.0001 | +0.0510 | [+0.0360, +0.0660] | 0.0001 |
| HotpotQA | Dense | +0.0221 | [+0.0123, +0.0326] | 0.0001 | +0.0190 | [+0.0090, +0.0300] | 0.0005 |
| MuSiQue | Dense | +0.0242 | [+0.0111, +0.0376] | 0.0001 | +0.0180 | [+0.0050, +0.0310] | 0.0112 |

Interpretation:
- All six F1 gains have bootstrap 95% confidence intervals above zero.
- EM is significant for five of six settings; `MuSiQue x PropRAG` remains F1-positive but EM-uncertain.

Layer-1 failure taxonomy:

Source:
- `run_logs/failure_taxonomy_layer1_20260424/summary.md`

| Pool | Dataset | Changed | Wins | Losses | Mean Delta F1 | Loss: gold pushed | Loss: reader noise | Loss: binding candidate | Loss: other |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PropRAG | 2Wiki | 191 | 47 | 8 | +0.0353 | 1 | 3 | 1 | 3 |
| PropRAG | HotpotQA | 130 | 17 | 3 | +0.0121 | 0 | 3 | 0 | 0 |
| PropRAG | MuSiQue | 334 | 39 | 28 | +0.0138 | 1 | 9 | 9 | 9 |
| Dense | 2Wiki | 226 | 79 | 6 | +0.0644 | 1 | 3 | 1 | 1 |
| Dense | HotpotQA | 117 | 35 | 6 | +0.0221 | 2 | 3 | 0 | 1 |
| Dense | MuSiQue | 307 | 52 | 22 | +0.0242 | 3 | 6 | 7 | 6 |

Interpretation:
- `MuSiQue` remains the residual failure dataset: it has the largest number of losses and a mixed loss profile.
- `HotpotQA` losses are mostly reader interference, consistent with its shallow support structure.
- The automatic taxonomy is conservative; the emitted `*.loss_cases.jsonl` files should be manually audited before writing final qualitative claims.

Layer-1 targeted `nobinding` ablation:

| Dataset | Pool | Full DAEC Delta F1 | NoBinding Delta F1 | Binding Contribution | NoBinding Report |
|---|---|---:|---:|---:|---|
| 2Wiki | PropRAG | +0.0353 | +0.0142 | +0.0211 | `run_logs/layer1_proprag_pool_ablation_2wiki_20260424/2wikimultihopqa_nobinding.json` |
| 2Wiki | Dense | +0.0644 | +0.0100 | +0.0544 | `run_logs/layer1_targeted_nobinding_20260424/2wikimultihopqa_dense_nvembed_top100_nobinding.json` |
| HotpotQA | PropRAG | +0.0121 | +0.0009 | +0.0112 | `run_logs/layer1_targeted_nobinding_20260424/hotpotqa_proprag_clean_nothink_top100_nobinding.json` |
| MuSiQue | PropRAG | +0.0138 | +0.0055 | +0.0083 | `run_logs/layer1_targeted_nobinding_20260424/musique_proprag_clean_nothink_top100_nobinding.json` |
| MuSiQue | Dense | +0.0242 | -0.0003 | +0.0245 | `run_logs/layer1_targeted_nobinding_20260424/musique_dense_nvembed_top100_nobinding.json` |

Interpretation:
- Dependency binding is not a `2Wiki + PropRAG` artifact.
- All five tested `(dataset, pool)` pairs lose F1 when binding is disabled.
- Binding should remain a main-method component; rank prior and repair typing should remain optional/appendix until broader ablations support them.

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

Canonical-backbone closure-only shared config:
- report path: `outputs_step0_general_2wikimultihopqa/eval_reports/setwise_bridge_beam_pilot_100_legacy_reserve4_bridge1_gate015.json`
- setup: `limit=100`, `pool_k=100`, selector `bridge_beam`, anchor `2`, `reserve_top_m=4`, `max_bridge_slots=1`, `non_anchor_title_dedup=true`, `gate_mode=suffix_bridge`, `gate_min_structure_score=0.15`, base mode `legacy_fact_graph`
- baseline `EM/F1 = 0.3800 / 0.4332`
- selector `EM/F1 = 0.4300 / 0.4719`
- delta `EM/F1 = +0.0500 / +0.0387`
- baseline `Recall@5 / Recall@20 = 0.7800 / 0.8675`
- selector `Recall@5 / Recall@20 = 0.8075 / 0.8775`
- `2-doc EM delta = +0.0649`
- `4-doc EM delta = +0.0000`
- gate apply / skip `= 21 / 79`

Interpretation:
- This is the current safest shared selector setting.
- It preserves a clear positive `2Wiki` gain, but is less aggressive than the earlier beam run.
- The benefit now concentrates on easier `2-doc` cases; the old positive `4-doc` lift is mostly gone under the tighter gate.

Failed micro-tweak:
- report path: `outputs_step0_general_2wikimultihopqa/eval_reports/setwise_bridge_greedy_pilot_20_legacy_seedunion_dedup.json`
- tweak: union `fact seeds` with `lexical seeds` and add title dedup
- delta `EM/F1 = -0.0500 / -0.0647`
- `4-doc EM delta = -0.4000`
- status: reverted from code

Strongest backup smoke:
- summary note: `outputs_step0_general_2wikimultihopqa/eval_reports/strongest_bridge_append_none_applypool_smoke40_20260416.md`
- setup: `bridge_append`, `assemble_mode=none`, `strongest_shadow_apply_to_pool=true`
- control `EM/F1 = 0.3500 / 0.4170`
- strongest `EM/F1 = 0.4000 / 0.4824`
- delta `EM/F1 = +0.0500 / +0.0654`
- control `Recall@5 = 0.7937`
- strongest `Recall@5 = 0.8375`

Interpretation:
- This is enough to keep strongest as a live backup candidate on a harder dataset.
- It is not enough to promote it to the paper mainline by itself.

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

Canonical-backbone shared-config boundary check:
- report path: `outputs_step0_general_hotpotqa/eval_reports/setwise_bridge_beam_pilot_100_legacy_reserve3_dedup.json`
- setup: `limit=100`, `pool_k=100`, selector `bridge_beam`, anchor `2`, `reserve_top_m=3`, `non_anchor_title_dedup=true`, base mode `legacy_fact_graph`
- baseline `EM/F1 = 0.5900 / 0.7114`
- selector `EM/F1 = 0.5500 / 0.6610`
- delta `EM/F1 = -0.0400 / -0.0504`
- baseline `Recall@5 / Recall@20 = 0.9150 / 0.9650`
- selector `Recall@5 / Recall@20 = 0.8600 / 0.9700`
- `2-doc EM delta = -0.0400`

Interpretation:
- The same selector settings that help hard datasets do not transfer cleanly to shallow `HotpotQA`.
- `Recall@20` ticks up slightly, but `Recall@5` and QA both drop, so the issue is still evidence composition quality near the top of the set.
- This should be treated as a boundary condition for the current method, not as a cross-dataset success.

Canonical-backbone closure-only boundary repair:
- report path: `outputs_step0_general_hotpotqa/eval_reports/setwise_bridge_beam_pilot_100_legacy_reserve4_bridge1_gate015.json`
- setup: `limit=100`, `pool_k=100`, selector `bridge_beam`, anchor `2`, `reserve_top_m=4`, `max_bridge_slots=1`, `non_anchor_title_dedup=true`, `gate_mode=suffix_bridge`, `gate_min_structure_score=0.15`, base mode `legacy_fact_graph`
- baseline `EM/F1 = 0.5900 / 0.7114`
- selector `EM/F1 = 0.5900 / 0.7064`
- delta `EM/F1 = +0.0000 / -0.0050`
- baseline `Recall@5 / Recall@20 = 0.9150 / 0.9650`
- selector `Recall@5 / Recall@20 = 0.9150 / 0.9700`
- `2-doc EM delta = +0.0000`
- gate apply / skip `= 15 / 85`

Interpretation:
- The closure-only gate removes the earlier destructive `HotpotQA` drop.
- This makes the shared setting effectively non-destructive on the shallow dataset, with only a very small `F1` loss.
- The result supports a narrow claim: shallow datasets should mostly skip bridge exploration unless there is strong off-prefix structure.

Strongest / GBC end-to-end audit:
- audit note: `outputs_step0_general_hotpotqa/eval_reports/strongest_gbc_e2e_audit_hotpotqa20_20260416.md`
- setup: `bridge_append`, `assemble_mode=none`, strongest sidecar applied to pool
- baseline `EM/F1 = 0.5500 / 0.6568`
- standard strongest `EM/F1 = 0.4500 / 0.5568`
- clean GBC `EM/F1 = 0.5000 / 0.6068`

Interpretation:
- Raw strongest is too aggressive on the shallow dataset.
- `clean GBC` recovers part of the loss and keeps the line interesting as a backup.
- The line is still below baseline end-to-end, so it should remain a reserve option only.

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

Canonical-backbone matched greedy control:
- report path: `outputs_step0_general_musique/eval_reports/setwise_bridge_greedy_pilot_100_legacy.json`
- setup: `limit=100`, `pool_k=100`, selector `bridge_greedy`, anchor `2`, base mode `legacy_fact_graph`
- baseline `EM/F1 = 0.2600 / 0.3466`
- selector `EM/F1 = 0.1800 / 0.2426`
- delta `EM/F1 = -0.0800 / -0.1040`
- baseline `Recall@5 / Recall@20 = 0.6150 / 0.7858`
- greedy `Recall@5 / Recall@20 = 0.4925 / 0.7933`
- `2-doc EM delta = -0.1250`
- `3-doc EM delta = -0.0667`
- `4-doc EM delta = +0.0000`

Matched beam vs greedy reading:
- `beam` and `greedy` have the same `EM = 0.1800`
- `beam` is slightly better on `F1` (`0.2498` vs `0.2426`)
- `beam` is better on the `4-doc` bucket (`+0.0455` vs `+0.0000`)
- therefore the main MuSiQue failure is not search alone; the score family itself is miscalibrated for shallow cases

Canonical-backbone reserved-prefix beam rescue:
- report path: `outputs_step0_general_musique/eval_reports/setwise_bridge_beam_pilot_100_legacy_reserve3_dedup.json`
- setup: `limit=100`, `pool_k=100`, selector `bridge_beam`, anchor `2`, `reserve_top_m=3`, `non_anchor_title_dedup=true`, base mode `legacy_fact_graph`
- baseline `EM/F1 = 0.2600 / 0.3466`
- selector `EM/F1 = 0.2700 / 0.3382`
- delta `EM/F1 = +0.0100 / -0.0084`
- baseline `Recall@5 / Recall@20 = 0.6150 / 0.7858`
- selector `Recall@5 / Recall@20 = 0.5508 / 0.7958`
- `2-doc EM delta = -0.0208`
- `3-doc EM delta = +0.0333`
- `4-doc EM delta = +0.0455`

Interpretation:
- The stronger reserved prefix removes the catastrophic MuSiQue failure.
- Overall `EM` is now positive, though still modest.
- The method now looks plausibly usable across hard datasets, but `F1` remains slightly below baseline and should be treated as still stabilizing.

Canonical-backbone closure-only shared config:
- report path: `outputs_step0_general_musique/eval_reports/setwise_bridge_beam_pilot_100_legacy_reserve4_bridge1_gate015.json`
- setup: `limit=100`, `pool_k=100`, selector `bridge_beam`, anchor `2`, `reserve_top_m=4`, `max_bridge_slots=1`, `non_anchor_title_dedup=true`, `gate_mode=suffix_bridge`, `gate_min_structure_score=0.15`, base mode `legacy_fact_graph`
- baseline `EM/F1 = 0.2600 / 0.3466`
- selector `EM/F1 = 0.2600 / 0.3452`
- delta `EM/F1 = +0.0000 / -0.0014`
- baseline `Recall@5 / Recall@20 = 0.6150 / 0.7858`
- selector `Recall@5 / Recall@20 = 0.5842 / 0.7908`
- `2-doc EM delta = -0.0208`
- `3-doc EM delta = +0.0000`
- `4-doc EM delta = +0.0455`
- gate apply / skip `= 55 / 45`

Interpretation:
- This shared closure-only setting removes the overall MuSiQue `EM` loss without needing dataset-specific parameters.
- The hard-case signal remains real: `4-doc` stays positive while the aggregate result recovers to neutral.
- The remaining issue is calibration, not pure search: the gate still fires on too many MuSiQue queries, so shallow-case damage cancels the hard-case gains.

## Cross-Dataset Summary

Summary files:
- `research_memory/emnlp_expand_then_compose/oracle_select_cross_dataset_summary_20260329_resume.md`
- `research_memory/emnlp_expand_then_compose/oracle_select_cross_dataset_summary_20260329_resume.json`

Takeaways:
- `2Wiki` is the main mechanism dataset: it has the bridge diagnosis and the sharp `2-doc` vs `4-doc` split.
- `MuSiQue` is the main hard-generalization dataset: support depth grows from `2-doc` to `4-doc`, and oracle gains keep growing with larger pools.
- `HotpotQA` is the shallow contrast dataset: nearly all support is already near the top, so larger pools still help but much less.
- The aggressive `bridge_beam` config is strongest on `2Wiki`, but not safe as a shared cross-dataset setting.
- The closure-only shared config is the current best compromise: `2Wiki` stays positive, `HotpotQA` becomes non-destructive, and `MuSiQue` recovers to neutral overall while keeping a positive `4-doc` signal.
- So the method story is now safer, but still not a clean universal improvement claim.
