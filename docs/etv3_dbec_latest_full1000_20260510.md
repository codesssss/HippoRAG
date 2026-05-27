# ETv3 Pool + Stable DBEC Full1000 Results

Date: 2026-05-10

## Scope

This note records the completed full1000 evaluation of the latest
LLM-binding DBEC/DAEC selector on top of the frozen ETv3 variable-flow pool.

This is not ETv4 and not a retrieval-side change.  ETv3 is treated as the
fixed candidate generator/readout baseline; DBEC is applied as a local
baseline-stable composition selector over the ETv3 pool.

## Configuration

| Item | Setting |
| --- | --- |
| Run root | `run_logs/etv3_dbec_latest_full1000_20260510` |
| Wrapper | `evidence_transition_graphragv3_dbec_latest/run_eval.py` |
| Selector | `daec_noisyor_safe_llm` |
| Rows | full1000 per dataset |
| Datasets | `2wikimultihopqa`, `hotpotqa`, `musique` |
| LLM | Qwen3-8B no-think substrate |
| Embedding | `nvidia/NV-Embed-v2` |
| Pool | ETv3 pool, `setwise_pool_k=100` |
| Reader budget | `qa_top_k=5` |
| Objective | `frozen_binding_noisy_or` |
| Binding mode | LLM binding, `wiki_title` title match |
| Local edit policy | preserve top1, at most 2 swaps, min objective gain `0.0`, min swap gain `1e-06` |

Launcher:

```text
run_logs/launch_etv3_dbec_latest_full1000_20260510.sh
```

Outputs:

| Dataset | Output |
| --- | --- |
| 2Wiki | `run_logs/etv3_dbec_latest_full1000_20260510/evals/2wikimultihopqa_etv3_pool100_dbec_stable_limit1000.json` |
| HotpotQA | `run_logs/etv3_dbec_latest_full1000_20260510/evals/hotpotqa_etv3_pool100_dbec_stable_limit1000.json` |
| MuSiQue | `run_logs/etv3_dbec_latest_full1000_20260510/evals/musique_etv3_pool100_dbec_stable_limit1000.json` |

## Same-Run Results

The baseline below is the same-run ETv3 pool100 top5 baseline exported inside
each DBEC evaluation JSON.  This is the fair delta for this experiment.

| Dataset | Baseline EM | DBEC EM | Delta EM | Baseline F1 | DBEC F1 | Delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 0.5790 | 0.6160 | +0.0370 | 0.6516 | 0.6867 | +0.0351 |
| HotpotQA | 0.6160 | 0.6240 | +0.0080 | 0.7324 | 0.7368 | +0.0044 |
| MuSiQue | 0.3330 | 0.3040 | -0.0290 | 0.4332 | 0.3989 | -0.0343 |

Selector retrieval recall:

| Dataset | R@5 | R@10 | R@20 |
| --- | ---: | ---: | ---: |
| 2Wiki | 0.9300 | 0.9667 | 0.9715 |
| HotpotQA | 0.9385 | 0.9865 | 0.9905 |
| MuSiQue | 0.6912 | 0.8105 | 0.8587 |

## MuSiQue Depth Breakdown

This was the pre-registered decision slice.

| Gold docs | Count | Baseline EM | DBEC EM | Delta EM | Baseline F1 | DBEC F1 | Delta F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 518 | 0.4170 | 0.3842 | -0.0328 | 0.5173 | 0.4788 | -0.0384 |
| 3 | 316 | 0.2975 | 0.2595 | -0.0380 | 0.3978 | 0.3594 | -0.0384 |
| 4 | 166 | 0.1386 | 0.1386 | +0.0000 | 0.2382 | 0.2245 | -0.0137 |

## Pre-Registered Gate Outcome

Primary MuSiQue 4-doc gate:

- Required for promotion: 4-doc F1 `>= 0.3000`.
- Residual-audit gray zone: `[0.2500, 0.3000)`.
- Observed: `0.2245`.
- Outcome: failed.  DBEC-on-ETv3 should not be promoted as the mainline fix.

Regression gate:

- Required: MuSiQue 2-doc F1 at least `0.5068`.
- Observed: `0.4788`.
- Outcome: failed.  The local DBEC selector hurts shallow MuSiQue cases beyond
  the allowed tolerance.

## Cost / Calls

The run used real LLM binding calls rather than an empty or no-op binding path.

| Dataset | Binding calls | Cache hits | Failures | Total tokens | Avg calls/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 7,527 | 128 | 0 | 1,214,112 | 7.53 |
| HotpotQA | 6,323 | 307 | 0 | 1,228,565 | 6.32 |
| MuSiQue | 6,746 | 649 | 0 | 1,363,334 | 6.75 |

## Interpretation

The result is asymmetric:

- On 2Wiki, stable DBEC over ETv3 pool is clearly positive, especially on the
  4-doc slice (`F1 +0.1165`).
- On HotpotQA, it is almost neutral-positive (`F1 +0.0044`), consistent with a
  shallow 2-doc dataset where ETv3 already has high set completeness.
- On MuSiQue, it fails the core target: all depth buckets are hurt, and the
  4-doc slice remains far below the promotion threshold.

This means the native ETv3 audit conclusion still stands: MuSiQue has large
candidate-pool headroom, but direct DBEC local editing does not convert that
headroom into a better top5 evidence set.  The failure is not explained by a
missing LLM-binding execution path, because the binding cache shows thousands
of successful calls and zero failures.

The immediate next step is residual audit, not ETv4-state-binding by assumption.
That audit should separate at least three possibilities:

- DBEC selects plausible but wrong branch/entity documents.
- DBEC's noisy-OR demand coverage objective saturates on shallow textual
  matches and therefore swaps out reader-useful evidence.
- MuSiQue candidate quality/positioning requires composition over deeper ranks
  than this local top5-edit policy can safely exploit.

Before residual evidence supports a state-specific mechanism, the safe label
remains `ETv4-composition` or unnamed; this result does not justify the
stronger `ETv4-state-binding` claim.

## Residual Audit

Offline residual audit was run after this result:

```text
reports/etv3_dbec_latest_full1000_residual_audit_20260510/etv3_dbec_full1000_residual_audit.md
```

Key findings:

| Dataset | Changed | Base title-all@5 | DBEC title-all@5 | Swaps | Swap-in gold | Swap-out gold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 487 | 0.7060 | 0.8200 | 539 | 186 | 94 |
| HotpotQA | 209 | 0.9050 | 0.8880 | 226 | 33 | 57 |
| MuSiQue | 553 | 0.4600 | 0.4520 | 740 | 143 | 228 |

MuSiQue harm summary:

- Worsened queries: `92`.
- Worsened and changed queries: `92`.
- Worsened with a gold title swapped out: `75`.
- Worsened with a gold title swapped in: `14`.
- Worsened without a gold title swapped out: `17`.
  - `6` preserve title-level gold completeness and likely reflect ordering or
    extra-context sensitivity.
  - `9` keep the same title recall and likely reflect non-gold reshuffling or
    reader sensitivity.
  - `2` increase title recall but still hurt reader F1.
- Pool title-complete but DBEC top5 title-incomplete: `374`.

Interpretation: the MuSiQue failure is not that DBEC is too conservative.  It
edits frequently, but its local noisy-OR objective is miscalibrated on MuSiQue:
it often swaps out currently useful evidence while only weakly improving
title-level set completeness.  This points away from simple "more swapping" or
budget-only fixes.  ETv4 should first address composition objective calibration
and residual case structure, not assume state-binding as the mechanism.

2Wiki positive result is concentrated in the deep slice rather than uniform:

| 2Wiki slice | Count | Baseline F1 | DBEC F1 | Delta F1 | Title-all@5 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2-doc | 765 | 0.6235 | 0.6337 | +0.0101 | 0.8784 -> 0.8954 |
| 4-doc | 235 | 0.7431 | 0.8596 | +0.1165 | 0.1447 -> 0.5745 |

The 4-doc slice contributes `77.9%` of the overall 2Wiki F1 gain.  DBEC still
has a real narrow-domain value: deep, title-identifiable 2Wiki-style chains.
The failure is that this value does not transfer to MuSiQue, where preservation
errors dominate.

## Additive-Only Follow-Up

A minimal preservation-first diagnostic was run on MuSiQue full1000:

```text
docs/etv3_dbec_additive_only_musique_full1000_20260510.md
```

It keeps ETv3 top4 fixed and allows DBEC to replace only the fifth slot
(`--daec-safe-preserve-top-m 4 --daec-safe-max-swaps 1`).  It reuses the same
pool and the same binding cache, with `0` new binding calls.

| Method | Overall F1 | 2-doc F1 | 3-doc F1 | 4-doc F1 |
| --- | ---: | ---: | ---: | ---: |
| ETv3 baseline | 0.4332 | 0.5173 | 0.3978 | 0.2382 |
| Stable DBEC top1/max2 | 0.3989 | 0.4788 | 0.3594 | 0.2245 |
| Additive-only top4/max1 | 0.4468 | 0.5338 | 0.3946 | 0.2744 |

This flips the diagnosis from "DBEC is unusable" to a more precise statement:
unconstrained DBEC local editing is unusable on MuSiQue, but DBEC as a
preservation-constrained add-on signal is promising.  ETv4-composition should
therefore start from a hard evidence-retention constraint and treat DBEC
utility as a secondary admission score.

## Preservation Grid

The follow-up cross-dataset preservation grid is recorded here:

```text
docs/etv3_dbec_preservation_grid_full1000_20260510.md
```

The key update is that top4/max1 is not merely a MuSiQue-specific rescue:

| Dataset | Variant | F1 delta | Key slice |
| --- | --- | ---: | --- |
| 2Wiki | top4/max1 | +0.0489 | 4-doc `+0.1080` |
| HotpotQA | top4/max1 | +0.0156 | 2-doc `+0.0155` |
| MuSiQue | top4/max1 | +0.0136 | 4-doc `+0.0362` |
| MuSiQue | top3/max2 | +0.0047 | 4-doc `+0.0239` |

This strengthens the design conclusion: preservation-first composition is a
cross-dataset constraint, while simply allowing more replacement capacity
(`top3/max2`) is weaker than protecting top4 and admitting one residual
document.
