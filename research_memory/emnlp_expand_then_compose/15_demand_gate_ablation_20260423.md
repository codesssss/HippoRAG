# Demand Gate Ablation

Date: 2026-04-23

## Purpose

This note records the `DtC-Embed + demand_gate` pilot-100 sweep.

The tested hypothesis was:

> If the baseline top-5 already covers enough decomposed evidence demands, preserving the baseline context should reduce harmful DtC interventions without losing many wins.

The result is negative as a main-method direction. Demand gate is useful as a diagnostic and negative ablation, but it over-abstains and loses too much of unconditional DtC's positive signal.

## Protocol

Common protocol:

- Reader: `qwen3-8b-train`
- Reader endpoint: `http://localhost:8043/v1`
- Embedding: `VLLM/nvidia/NV-Embed-v2`
- Embedding endpoint: `http://localhost:8019/v1/embeddings`
- NV-Embed instruction prefix: off
- Selector: `dtc_embed`
- Pool: `top-100`
- Final evidence budget: `top-5`
- `dtc_rank_weight = 0.2`
- `dtc_match_threshold = 0.35`
- `dtc_demand_gate_alpha in {0.50, 0.67, 0.80, 0.90, 1.00}`

Reports:

- No gate:
  - `outputs_step0_general_nvembed_2wikimultihopqa/eval_reports/dtc_embed_nvembed_rankw0p2_pilot100_anchor2_8043.json`
  - `outputs_step0_general_nvembed_hotpotqa/eval_reports/dtc_embed_nvembed_rankw0p2_pilot100_anchor2_8043.json`
  - `outputs_step0_general_nvembed_musique/eval_reports/dtc_embed_nvembed_rankw0p2_pilot100_anchor2_8043.json`
- Demand gate sweep:
  - `outputs_step0_general_nvembed_2wikimultihopqa/eval_reports/dtc_embed_nvembed_rankw0p2_demandgate_a*_pilot100_anchor2_8043.json`
  - `outputs_step0_general_nvembed_hotpotqa/eval_reports/dtc_embed_nvembed_rankw0p2_demandgate_a*_pilot100_anchor2_8043.json`
  - `outputs_step0_general_nvembed_musique/eval_reports/dtc_embed_nvembed_rankw0p2_demandgate_a*_pilot100_anchor2_8043.json`

## Raw Results

| Dataset | Variant | Baseline EM | Selector EM | Delta EM | Baseline F1 | Selector F1 | Delta F1 | Wins | Losses | Ties | Changed | Preserve | Avg DS | Selector R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | no gate | 0.4000 | 0.4400 | +0.0400 | 0.4531 | 0.4987 | +0.0456 | 8 | 2 | 90 | 46 | - | - | 0.8525 |
| 2Wiki | gate 0.50 | 0.4000 | 0.4000 | +0.0000 | 0.4503 | 0.4503 | +0.0000 | 0 | 0 | 100 | 0 | 99 | 0.8608 | 0.8150 |
| 2Wiki | gate 0.67 | 0.4000 | 0.4300 | +0.0300 | 0.4503 | 0.4884 | +0.0381 | 6 | 1 | 93 | 22 | 70 | 0.8608 | 0.8500 |
| 2Wiki | gate 0.80 | 0.4000 | 0.4300 | +0.0300 | 0.4531 | 0.4913 | +0.0382 | 6 | 1 | 93 | 23 | 69 | 0.8608 | 0.8500 |
| 2Wiki | gate 0.90 | 0.4000 | 0.4300 | +0.0300 | 0.4531 | 0.4893 | +0.0362 | 6 | 1 | 93 | 23 | 69 | 0.8608 | 0.8500 |
| 2Wiki | gate 1.00 | 0.4000 | 0.4300 | +0.0300 | 0.4531 | 0.4902 | +0.0371 | 6 | 1 | 93 | 23 | 69 | 0.8608 | 0.8500 |
| HotpotQA | no gate | 0.5700 | 0.5900 | +0.0200 | 0.6744 | 0.7054 | +0.0310 | 4 | 1 | 95 | 32 | - | - | 0.9400 |
| HotpotQA | gate 0.50 | 0.5700 | 0.5700 | +0.0000 | 0.6744 | 0.6744 | +0.0000 | 0 | 0 | 100 | 1 | 99 | 0.9742 | 0.9250 |
| HotpotQA | gate 0.67 | 0.5700 | 0.5700 | +0.0000 | 0.6744 | 0.6794 | +0.0050 | 1 | 0 | 99 | 4 | 95 | 0.9742 | 0.9250 |
| HotpotQA | gate 0.80 | 0.5700 | 0.5700 | +0.0000 | 0.6744 | 0.6794 | +0.0050 | 1 | 0 | 99 | 4 | 95 | 0.9742 | 0.9250 |
| HotpotQA | gate 0.90 | 0.5700 | 0.5700 | +0.0000 | 0.6744 | 0.6794 | +0.0050 | 1 | 0 | 99 | 4 | 95 | 0.9742 | 0.9250 |
| HotpotQA | gate 1.00 | 0.5700 | 0.5700 | +0.0000 | 0.6744 | 0.6794 | +0.0050 | 1 | 0 | 99 | 4 | 95 | 0.9742 | 0.9250 |
| MuSiQue | no gate | 0.3000 | 0.3400 | +0.0400 | 0.3819 | 0.4216 | +0.0397 | 11 | 5 | 84 | 66 | - | - | 0.6850 |
| MuSiQue | gate 0.50 | 0.3000 | 0.3000 | +0.0000 | 0.3852 | 0.3852 | +0.0000 | 1 | 1 | 98 | 10 | 89 | 0.8267 | 0.6533 |
| MuSiQue | gate 0.67 | 0.3000 | 0.3000 | +0.0000 | 0.3864 | 0.3898 | +0.0034 | 4 | 4 | 92 | 23 | 72 | 0.8267 | 0.6633 |
| MuSiQue | gate 0.80 | 0.3000 | 0.3000 | +0.0000 | 0.3864 | 0.3898 | +0.0034 | 4 | 4 | 92 | 25 | 70 | 0.8267 | 0.6633 |
| MuSiQue | gate 0.90 | 0.3000 | 0.3000 | +0.0000 | 0.3819 | 0.3853 | +0.0034 | 4 | 4 | 92 | 25 | 70 | 0.8267 | 0.6633 |
| MuSiQue | gate 1.00 | 0.3000 | 0.3000 | +0.0000 | 0.3859 | 0.3893 | +0.0034 | 4 | 4 | 92 | 25 | 70 | 0.8267 | 0.6633 |

Notes:

- Small baseline F1 differences across gate runs come from independent reader calls/caching variance. The conclusion uses deltas within each report and the no-gate reference run.
- `Avg DS` is the average baseline demand-satisfaction rate reported by the selector traces.

## Findings

1. Demand gate removes too many interventions.
   - On HotpotQA, gate `alpha >= 0.67` preserves 95/100 queries and leaves only 4 changed queries.
   - On MuSiQue, gate `alpha >= 0.67` preserves 70-72/100 queries and leaves only 23-25 changed queries.

2. Loss reduction is not enough to compensate for lost wins.
   - 2Wiki improves losses from 2 to 1, but wins fall from 8 to 6.
   - HotpotQA improves losses from 1 to 0, but wins fall from 4 to 1.
   - MuSiQue improves losses from 5 to 4, but wins fall from 11 to 4.

3. The best gate setting underperforms unconditional DtC.
   - 2Wiki best gate: `Delta F1 +0.0382`, no gate: `+0.0456`.
   - HotpotQA best gate: `Delta F1 +0.0050`, no gate: `+0.0310`.
   - MuSiQue best gate: `Delta F1 +0.0034`, no gate: `+0.0397`.

4. The result rejects gate-only as a main method.
   - The sufficiency signal is directionally meaningful, because high-DS queries are often safer to preserve.
   - But a hard compose-or-preserve decision discards too much useful repair capacity.

## Decision

Do not promote demand gate as the paper-facing method.

Use it as:

- a negative ablation;
- a diagnostic showing why unconditional composition is risky;
- motivation for a softer conservative-repair objective.

The next method should not be an if-else gate. It should internalize baseline sufficiency as a continuous preservation prior while still allowing targeted positive swaps.

## Next Line

Move to **SC-SER: Sufficiency-Calibrated Selective Evidence Repair**.

Core change:

- Treat fixed-pool evidence composition as conservative repair from the baseline top-5, not from-scratch recomposition.
- Use residual demand weighting so requirements already covered by baseline have lower repair reward.
- Use a sufficiency-calibrated preservation term so high-sufficiency baselines become harder to edit without making editing impossible.
- Optimize by warm-start 1-swap from the baseline top-5 and stop when no positive objective-improving swap exists.

