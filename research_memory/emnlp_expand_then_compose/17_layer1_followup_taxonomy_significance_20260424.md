# Layer-1 Follow-up: Failure Taxonomy, Significance, and Binding Ablation

Date: 2026-04-24

## Purpose

This note records the follow-up actions after the Layer-1 retriever-agnostic composition result.

The goal is to move from "the method improves" to paper-ready evidence:

- quantify whether the observed gains are statistically stable;
- identify residual failure modes, especially on `MuSiQue`;
- test whether dependency binding generalizes beyond the existing `2Wiki + PropRAG` ablation.

## Completed: Failure Taxonomy Script Fix

Script:

- `scripts/analyze_failure_taxonomy.py`

Bug fixed:

- `classify_case()` was reading missing `baseline_metrics` / `selector_metrics` fields from the row.
- The row stores expanded fields `baseline_f1` / `selector_f1`.
- Before the fix, all cases were incorrectly labeled as `tie` despite non-zero mean delta F1.

Output root:

- `run_logs/failure_taxonomy_layer1_20260424/`

Command:

```bash
.venv-hipporag/bin/python scripts/analyze_failure_taxonomy.py \
  run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json \
  run_logs/layer1_proprag_pool_eval_fixed_20260424/hotpotqa_proprag_pool_daec_oracle.json \
  run_logs/layer1_proprag_pool_eval_fixed_20260424/musique_proprag_pool_daec_oracle.json \
  run_logs/layer1_dense_pool_eval_20260424/2wikimultihopqa_dense_pool_daec_oracle.json \
  run_logs/layer1_dense_pool_eval_20260424/hotpotqa_dense_pool_daec_oracle.json \
  run_logs/layer1_dense_pool_eval_20260424/musique_dense_pool_daec_oracle.json \
  --output_dir run_logs/failure_taxonomy_layer1_20260424
```

Initial automatic taxonomy:

| Pool | Dataset | N | Changed | Wins | Losses | Ties | Mean dF1 | Loss: gold pushed | Loss: reader noise | Loss: binding cand | Loss: other |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PropRAG | 2Wiki | 1000 | 191 (19.1%) | 47 | 8 | 945 | +0.0353 | 1 | 3 | 1 | 3 |
| PropRAG | HotpotQA | 1000 | 130 (13.0%) | 17 | 3 | 980 | +0.0121 | 0 | 3 | 0 | 0 |
| PropRAG | MuSiQue | 1000 | 334 (33.4%) | 39 | 28 | 933 | +0.0138 | 1 | 9 | 9 | 9 |
| Dense | 2Wiki | 1000 | 226 (22.6%) | 79 | 6 | 915 | +0.0644 | 1 | 3 | 1 | 1 |
| Dense | HotpotQA | 1000 | 117 (11.7%) | 35 | 6 | 959 | +0.0221 | 2 | 3 | 0 | 1 |
| Dense | MuSiQue | 1000 | 307 (30.7%) | 52 | 22 | 926 | +0.0242 | 3 | 6 | 7 | 6 |

Interpretation:

- `MuSiQue` is confirmed as the residual failure dataset: many more losses than the other datasets and a mixed loss profile.
- `PropRAG + MuSiQue` losses split across reader interference, binding/wrong-entity candidates, and incomplete residual failures.
- `HotpotQA` losses are mostly reader interference, consistent with a strong baseline and shallow support structure.
- This automatic taxonomy is conservative; wrong-entity/binding labels should be manually audited from the emitted `*.loss_cases.jsonl` files.

## Completed: Paired Significance Script

Script:

- `scripts/analyze_layer1_significance.py`

Output root:

- `run_logs/layer1_significance_20260424/`
- `run_logs/layer1_significance_20260424/layer1_significance.md`
- `run_logs/layer1_significance_20260424/layer1_significance.json`

Method:

- paired per-query delta between selector and baseline;
- bootstrap 95% confidence interval over mean delta;
- paired sign-flip two-sided p-value;
- metrics: `F1`, `ExactMatch`.

Command:

```bash
.venv-hipporag/bin/python scripts/analyze_layer1_significance.py \
  run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json \
  run_logs/layer1_proprag_pool_eval_fixed_20260424/hotpotqa_proprag_pool_daec_oracle.json \
  run_logs/layer1_proprag_pool_eval_fixed_20260424/musique_proprag_pool_daec_oracle.json \
  run_logs/layer1_dense_pool_eval_20260424/2wikimultihopqa_dense_pool_daec_oracle.json \
  run_logs/layer1_dense_pool_eval_20260424/hotpotqa_dense_pool_daec_oracle.json \
  run_logs/layer1_dense_pool_eval_20260424/musique_dense_pool_daec_oracle.json \
  --output_dir run_logs/layer1_significance_20260424
```

Result:

| Dataset | Pool | N | dF1 mean | dF1 95% CI | F1 p | F1 +/-/0 | dEM mean | dEM 95% CI | EM p | EM +/-/0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | PropRAG | 1000 | +0.035258 | [+0.023043, +0.048033] | 0.000100 | 47/8/945 | +0.032000 | [+0.019000, +0.045000] | 0.000100 | 38/6/956 |
| HotpotQA | PropRAG | 1000 | +0.012067 | [+0.004733, +0.020000] | 0.002100 | 17/3/980 | +0.011000 | [+0.004000, +0.019000] | 0.007399 | 13/2/985 |
| MuSiQue | PropRAG | 1000 | +0.013770 | [+0.001199, +0.026607] | 0.035196 | 39/28/933 | +0.009000 | [-0.004000, +0.022000] | 0.213779 | 25/16/959 |
| 2Wiki | Dense | 1000 | +0.064369 | [+0.048909, +0.080193] | 0.000100 | 79/6/915 | +0.051000 | [+0.036000, +0.066000] | 0.000100 | 56/5/939 |
| HotpotQA | Dense | 1000 | +0.022068 | [+0.012252, +0.032570] | 0.000100 | 35/6/959 | +0.019000 | [+0.009000, +0.030000] | 0.000500 | 24/5/971 |
| MuSiQue | Dense | 1000 | +0.024235 | [+0.011145, +0.037636] | 0.000100 | 52/22/926 | +0.018000 | [+0.005000, +0.031000] | 0.011199 | 32/14/954 |

Interpretation:

- All six `F1` gains have bootstrap 95% CI above zero.
- All six `F1` gains have paired sign-flip p-values below `0.05`.
- `PropRAG + MuSiQue` is the weakest statistically stable F1 signal: positive CI, but small margin and EM is not significant.
- The strongest and most paper-safe effects are:
  - `Dense + 2Wiki`
  - `PropRAG + 2Wiki`
  - `Dense + HotpotQA`
  - `Dense + MuSiQue`

## Completed: Targeted Nobinding Ablation

Launcher:

- `run_logs/run_layer1_targeted_nobinding_20260424.sh`

Output root:

- `run_logs/layer1_targeted_nobinding_20260424/`

Completed:

- `2026-04-24T20:14:39+08:00`

Targeted runs:

| Dataset | Pool | Variant | Output |
|---|---|---|---|
| HotpotQA | PropRAG | `nobinding` | `run_logs/layer1_targeted_nobinding_20260424/hotpotqa_proprag_clean_nothink_top100_nobinding.json` |
| MuSiQue | PropRAG | `nobinding` | `run_logs/layer1_targeted_nobinding_20260424/musique_proprag_clean_nothink_top100_nobinding.json` |
| 2Wiki | Dense | `nobinding` | `run_logs/layer1_targeted_nobinding_20260424/2wikimultihopqa_dense_nvembed_top100_nobinding.json` |
| MuSiQue | Dense | `nobinding` | `run_logs/layer1_targeted_nobinding_20260424/musique_dense_nvembed_top100_nobinding.json` |

Rationale:

- This targeted set tests whether dependency binding is only useful on `2Wiki + PropRAG`, or transfers across dataset/pool settings.
- It is cheaper than the full `3 datasets x 2 pools x full/nobinding` matrix.
- If at least two additional runs show a meaningful F1 drop without binding, dependency binding can remain a main component.
- If not, dependency binding should be framed as a 2Wiki/entity-binding mechanism rather than a universal core component.

NoBinding raw results:

| Dataset | Pool | Base EM | Base F1 | NoBinding EM | NoBinding F1 | Delta EM | Delta F1 | R@5 | R@20 | R@100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | Dense | 0.4550 | 0.4984 | 0.4650 | 0.5084 | +0.0100 | +0.0100 | 0.7238 | 0.7990 | 0.8770 |
| HotpotQA | PropRAG | 0.5950 | 0.7227 | 0.5960 | 0.7236 | +0.0010 | +0.0009 | 0.9500 | 0.9925 | 0.9990 |
| MuSiQue | PropRAG | 0.3300 | 0.4266 | 0.3310 | 0.4321 | +0.0010 | +0.0055 | 0.7131 | 0.8942 | 0.9677 |
| MuSiQue | Dense | 0.2980 | 0.3896 | 0.2970 | 0.3893 | -0.0010 | -0.0003 | 0.6628 | 0.8197 | 0.9055 |

Binding contribution table:

| Dataset | Pool | Full DAEC Delta F1 | NoBinding Delta F1 | Binding Contribution |
|---|---|---:|---:|---:|
| 2Wiki | PropRAG | +0.0353 | +0.0142 | +0.0211 |
| 2Wiki | Dense | +0.0644 | +0.0100 | +0.0544 |
| HotpotQA | PropRAG | +0.0121 | +0.0009 | +0.0112 |
| MuSiQue | PropRAG | +0.0138 | +0.0055 | +0.0083 |
| MuSiQue | Dense | +0.0242 | -0.0003 | +0.0245 |

Interpretation:

- Dependency binding is not a `2Wiki + PropRAG` artifact.
- All five tested `(dataset, pool)` pairs lose F1 when binding is disabled.
- The largest effects are on dense-pool settings:
  - `2Wiki x Dense`: `+0.0544 F1`
  - `MuSiQue x Dense`: `+0.0245 F1`
- The PropRAG settings still show consistent positive binding contribution:
  - `2Wiki`: `+0.0211 F1`
  - `HotpotQA`: `+0.0112 F1`
  - `MuSiQue`: `+0.0083 F1`
- This is enough to keep dependency binding as a main-method component.
- Rank prior and repair typing remain unsupported as necessary components based on the current 2Wiki ablation.

## Next Actions

1. Manually audit a sample of `MuSiQue` loss cases from:
   - `run_logs/failure_taxonomy_layer1_20260424/musique_proprag_pool_daec_oracle.loss_cases.jsonl`
   - `run_logs/failure_taxonomy_layer1_20260424/musique_dense_pool_daec_oracle.loss_cases.jsonl`
2. Keep dependency binding in the main method.
3. Treat rank prior and repair typing as optional/appendix components unless additional cross-dataset ablations support them.
4. Start paper skeleton around the retriever-agnostic fixed-pool composition claim.
