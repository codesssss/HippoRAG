# DtC NV-Embed Full-Scale Results and Rank-Prior Direction

Date: 2026-04-22

## Purpose

This note records the current state of `DtC-Embed` as the active fixed-pool evidence composition method.

The key question is no longer whether generic fixed-pool diversity helps. The diversity study already showed that MMR/DPP recover little of the oracle gap. The current question is:

> Can demand-aware composition select reader-useful evidence from a fixed top-100 pool without over-trusting false positives from deep pool positions?

## Canonical Protocol

All completed numbers in this note use the aligned non-instruction NV-Embed protocol:

- Pool: `setwise_pool_k=100`
- Reader budget: `qa_top_k=5`
- Reader: frozen `qwen3-8b-train`
- Workdir model: `qwen3-8b`
- Embedding model: `VLLM/nvidia/NV-Embed-v2`
- Embedding endpoint: `http://localhost:8019/v1/embeddings`
- NV instruction prefix: disabled
- Decomposition: one-shot LLM parser, `dtc_decomposition_mode=llm`
- Dependency constraints: enabled
- Dependency binding: disabled for the canonical v1 run
- Hard crossing gate: disabled for the canonical v1 run

Canonical full1000 result files:

- `outputs_step0_general_nvembed_2wikimultihopqa/eval_reports/dtc_embed_nvembed_noninstr_soft_limit1000_anchor2_8043.json`
- `outputs_step0_general_nvembed_hotpotqa/eval_reports/dtc_embed_nvembed_noninstr_soft_limit1000_anchor2_8042.json`
- `outputs_step0_general_nvembed_musique/eval_reports/dtc_embed_nvembed_noninstr_soft_limit1000_anchor2_8041.json`

## Full1000 Main Results

| Dataset | Baseline EM | DtC EM | Delta EM | Baseline F1 | DtC F1 | Delta F1 | Changed | Wins / Losses / Ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.475 | 0.481 | +0.006 | 0.5413 | 0.5495 | +0.0082 | 487 / 1000 | 46 / 40 / 914 |
| HotpotQA | 0.574 | 0.590 | +0.016 | 0.7050 | 0.7225 | +0.0175 | 362 / 1000 | 41 / 23 / 936 |
| MuSiQue | 0.307 | 0.332 | +0.025 | 0.4051 | 0.4296 | +0.0245 | 633 / 1000 | 95 / 72 / 833 |

Interpretation:

- DtC is positive on all three datasets under the aligned NV non-instruction protocol.
- HotpotQA and MuSiQue are the stable signals; 2Wiki is positive but weaker because wins and losses are close.
- The full1000 effect is smaller than pilot100 because the rest900 slice has a stronger baseline and less oracle repair room.

## Pilot100 vs Full1000

Soft DtC pilot100:

| Dataset | Baseline F1 | DtC F1 | Delta F1 | Changed | Wins / Losses / Ties |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 0.4531 | 0.4787 | +0.0256 | 50 / 100 | 7 / 3 / 90 |
| HotpotQA | 0.6744 | 0.7054 | +0.0310 | 31 / 100 | 4 / 1 / 95 |
| MuSiQue | 0.3790 | 0.4287 | +0.0497 | 68 / 100 | 11 / 4 / 85 |

Full1000 shrinks the delta because many later queries are already easier for baseline top-5. The selector still changes many queries, which exposes a precision problem: when there is little remaining composition headroom, false-positive coverage can replace reader-friendly baseline context.

## Hard-Crossing Ablation

Hard-crossing means a DtC candidate can be inserted only if it crosses at least one previously uncovered requirement threshold. Soft coverage gain is only allowed as a tie-break among crossing candidates.

Pilot100 result files:

- `outputs_step0_general_nvembed_2wikimultihopqa/eval_reports/dtc_embed_nvembed_hardcross_pilot100_anchor2_fresh8043.json`
- `outputs_step0_general_nvembed_hotpotqa/eval_reports/dtc_embed_nvembed_hardcross_pilot100_anchor2_fresh8043.json`
- `outputs_step0_general_nvembed_musique/eval_reports/dtc_embed_nvembed_hardcross_pilot100_anchor2_fresh8043.json`

| Dataset | Soft Delta F1 | Hard-Cross Delta F1 | Soft Changed | Hard Changed | Soft W/L/T | Hard W/L/T |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | +0.0256 | +0.0157 | 50 / 100 | 30 / 100 | 7 / 3 / 90 | 4 / 3 / 93 |
| HotpotQA | +0.0310 | +0.0010 | 31 / 100 | 8 / 100 | 4 / 1 / 95 | 1 / 1 / 98 |
| MuSiQue | +0.0497 | +0.0023 | 68 / 100 | 38 / 100 | 11 / 4 / 85 | 4 / 5 / 91 |

Conclusion:

- Hard-crossing is not the right main fix.
- It removes many wins while leaving losses largely intact.
- This falsifies the simple hypothesis that losses mostly come from soft-gain-without-crossing insertions.
- The remaining failure mode is scoring false positives: some distractor documents cross a requirement threshold but do not carry the gold evidence the reader needs.

## Depth and Rank Diagnosis

Losses tend to pull from deeper pool positions than wins:

| Dataset | Avg deepest selected position in wins | Avg deepest selected position in losses |
|---|---:|---:|
| 2Wiki | 21.8 | 31.5 |
| HotpotQA | 11.3 | 22.4 |
| MuSiQue | 20.0 | 33.6 |

Post-hoc depth abstention on existing full1000 traces gives a small but consistent improvement when deep interventions are suppressed:

| Dataset | Soft DtC Delta F1 | Best observed depth-abstain Delta F1 |
|---|---:|---:|
| 2Wiki | +0.0082 | about +0.0093 |
| HotpotQA | +0.0175 | about +0.0180 |
| MuSiQue | +0.0245 | about +0.0318 |

This is useful analysis but should not become the paper method as a fixed `D=50` rule. A hard depth cutoff looks like a dataset-tuned trick.

## Current Design Decision

Do not promote hard-crossing or fixed-depth abstention as the main method.

The next method variant should be:

> Rank-regularized demand coverage.

Current DtC objective:

```text
total_gain =
  coverage_gain
  + 0.25 * new_requirement_count
  + base_weight * normalized_base_score
  - redundancy_weight * redundancy
```

Problem:

- `base_weight=0.05` is too weak.
- `normalized_base_score` is min-max normalized per query, which is less stable than rank.
- A deep false-positive crossing can easily dominate the base retriever prior.

Proposed objective:

```text
normalized_rank = pool_position / max(pool_k - 1, 1)

total_gain =
  coverage_gain
  + 0.25 * new_requirement_count
  - rank_weight * normalized_rank
  - redundancy_weight * redundancy
```

Recommended pilot sweep:

- `rank_weight in {0.1, 0.2, 0.3, 0.6, 1.0}`
- Keep the canonical v1 run as `rank_weight=0.0`.
- Evaluate on pilot100 first, then promote to full1000 only if wins are mostly preserved and losses drop.

## Paper Implication

The clean story is:

1. Fixed-pool oracle analysis shows composition headroom.
2. MMR/DPP show generic structure-blind diversity is not enough.
3. Soft DtC shows demand-aware composition works, but can over-trust deep false positives.
4. Hard-crossing shows per-candidate acceptance gates are too conservative.
5. Rank-regularized DtC is the principled correction: it internalizes the base retriever rank prior into the demand coverage objective.

This keeps the method as an objective-level contribution rather than a hand-coded depth gate.
