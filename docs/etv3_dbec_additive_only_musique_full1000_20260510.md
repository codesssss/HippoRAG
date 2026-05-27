# ETv3 + DBEC Additive-Only Diagnostic on MuSiQue

Date: 2026-05-10

## Scope

This note records a minimal preservation-first diagnostic after the failed
ETv3-pool + stable DBEC full1000 run.

It is not a new method implementation.  It reuses the existing stable DBEC
local-edit selector and changes only the edit policy:

```text
--daec-safe-preserve-top-m 4
--daec-safe-max-swaps 1
--daec-safe-min-objective-gain 0.0
--daec-safe-min-swap-gain 1e-06
```

Under a fixed top5 reader budget, this approximates an "additive-only" policy:
keep the first four ETv3 documents fixed and allow DBEC to replace only the
fifth slot.  This tests whether evidence preservation is the key constraint,
not whether unrestricted DBEC utility maximization can work.

## Configuration

| Item | Setting |
| --- | --- |
| Dataset | `musique` full1000 |
| Run root | `run_logs/etv3_dbec_additive_only_full1000_20260510` |
| Output JSON | `run_logs/etv3_dbec_additive_only_full1000_20260510/evals/musique_etv3_pool100_dbec_additive_only_limit1000.json` |
| Pool JSON | `run_logs/etv3_dbec_latest_full1000_20260510/pools/musique_etv3_pool100_limit1000.json` |
| Binding cache | `run_logs/etv3_dbec_latest_full1000_20260510/evals/musique_etv3_pool100_dbec_latest.binding_cache.json` |
| Selector | `daec_noisyor_safe_llm` |
| LLM | Qwen3-8B no-think substrate |
| Embedding | `VLLM/nvidia/NV-Embed-v2` |
| New binding calls | `0` |
| Binding cache hits | `7,395` |

## Result

| Method | EM | F1 | Delta EM | Delta F1 |
| --- | ---: | ---: | ---: | ---: |
| ETv3 pool100 top5 baseline | 0.3330 | 0.4332 | - | - |
| Stable DBEC, preserve top1/max2 swaps | 0.3040 | 0.3989 | -0.0290 | -0.0343 |
| Additive-only diagnostic, preserve top4/max1 swap | 0.3430 | 0.4468 | +0.0100 | +0.0136 |

Depth breakdown:

| Gold docs | Count | Baseline F1 | Stable DBEC F1 | Additive-only F1 | Additive delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 518 | 0.5173 | 0.4788 | 0.5338 | +0.0166 |
| 3 | 316 | 0.3978 | 0.3594 | 0.3946 | -0.0031 |
| 4 | 166 | 0.2382 | 0.2245 | 0.2744 | +0.0362 |

Retrieval recall after additive-only selection:

| R@1 | R@2 | R@5 | R@10 | R@20 |
| ---: | ---: | ---: | ---: | ---: |
| 0.3235 | 0.4779 | 0.7291 | 0.8101 | 0.8577 |

## Preservation Audit

| Variant | Changed queries | Swaps | Title-all@5 | Rescues | Regressions | Worsened queries | Worsened with gold swapped out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stable DBEC top1/max2 | 553 | 740 | 0.4520 | 55 | 63 | 92 | 75 |
| Additive-only top4/max1 | 533 | 539 | 0.4850 | 47 | 22 | 51 | 25 |

Depth-level preservation audit:

| Gold docs | Changed | Swaps | Title-all@5 | Rescues | Regressions | Worsened | Gold swapped out among worsened |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 212 | 215 | 0.7046 | 23 | 7 | 18 | 9 |
| 3 | 192 | 195 | 0.3513 | 20 | 14 | 27 | 13 |
| 4 | 129 | 129 | 0.0542 | 4 | 1 | 6 | 3 |

## Interpretation

This diagnostic strongly supports the reviewer's preservation-first hypothesis.

The unrestricted stable DBEC local editor failed on MuSiQue because it was
allowed to remove too much existing useful evidence.  Restricting the edit to
the final slot reverses the overall result (`-0.0343` to `+0.0136` F1), removes
the 2-hop regression (`-0.0384` to `+0.0166`), and improves the 4-doc slice
(`-0.0137` to `+0.0362`).

The important design lesson is:

```text
Evidence preservation should be a hard constraint; DBEC utility should be a
soft add-on objective.
```

This does not yet solve MuSiQue 4-doc fully.  The 4-doc F1 reaches `0.2744`,
which is inside the previous residual-audit gray zone but still below the
`0.3000` promotion threshold.  It is enough to justify ETv4-composition as a
preservation-first design direction, not enough to claim a finished method.
