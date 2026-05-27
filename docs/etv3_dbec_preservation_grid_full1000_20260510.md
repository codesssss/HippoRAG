# ETv3 + DBEC Preservation Grid Full1000

Date: 2026-05-10

## Scope

This note records the follow-up preservation grid after the MuSiQue
additive-only diagnostic.

The goal is to test two questions without implementing a new method:

1. Does preservation-first DBEC remain useful beyond MuSiQue?
2. Is MuSiQue limited by having only one admission slot, or by relaxing
   preservation too much?

All runs use the existing `daec_noisyor_safe_llm` selector, the same ETv3
pool100 files, and the existing LLM-binding caches.  No new binding calls were
made.

## Runs

| Dataset | Variant | Preserve top-m | Max swaps | Output |
| --- | --- | ---: | ---: | --- |
| 2Wiki | top4/max1 | 4 | 1 | `run_logs/etv3_dbec_preservation_grid_full1000_20260510/evals/2wikimultihopqa_etv3_pool100_dbec_top4_max1_limit1000.json` |
| HotpotQA | top4/max1 | 4 | 1 | `run_logs/etv3_dbec_preservation_grid_full1000_20260510/evals/hotpotqa_etv3_pool100_dbec_top4_max1_limit1000.json` |
| MuSiQue | top3/max2 | 3 | 2 | `run_logs/etv3_dbec_preservation_grid_full1000_20260510/evals/musique_etv3_pool100_dbec_top3_max2_limit1000.json` |

The prior MuSiQue top4/max1 diagnostic is included as a reference:

```text
run_logs/etv3_dbec_additive_only_full1000_20260510/evals/musique_etv3_pool100_dbec_additive_only_limit1000.json
```

## Main Results

| Dataset | Variant | Baseline EM | Selector EM | Delta EM | Baseline F1 | Selector F1 | Delta F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | top4/max1 | 0.5790 | 0.6240 | +0.0450 | 0.6516 | 0.7005 | +0.0489 |
| HotpotQA | top4/max1 | 0.6160 | 0.6290 | +0.0130 | 0.7324 | 0.7480 | +0.0156 |
| MuSiQue | top4/max1 | 0.3330 | 0.3430 | +0.0100 | 0.4332 | 0.4468 | +0.0136 |
| MuSiQue | top3/max2 | 0.3330 | 0.3340 | +0.0010 | 0.4332 | 0.4379 | +0.0047 |

## Depth Results

| Dataset | Variant | Gold docs | Count | Baseline F1 | Selector F1 | Delta F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | top4/max1 | 2 | 765 | 0.6235 | 0.6542 | +0.0307 |
| 2Wiki | top4/max1 | 4 | 235 | 0.7431 | 0.8511 | +0.1080 |
| HotpotQA | top4/max1 | 2 | 1000 | 0.7324 | 0.7480 | +0.0155 |
| MuSiQue | top4/max1 | 2 | 518 | 0.5173 | 0.5338 | +0.0166 |
| MuSiQue | top4/max1 | 3 | 316 | 0.3978 | 0.3946 | -0.0031 |
| MuSiQue | top4/max1 | 4 | 166 | 0.2382 | 0.2744 | +0.0362 |
| MuSiQue | top3/max2 | 2 | 518 | 0.5173 | 0.5242 | +0.0069 |
| MuSiQue | top3/max2 | 3 | 316 | 0.3978 | 0.3888 | -0.0089 |
| MuSiQue | top3/max2 | 4 | 166 | 0.2382 | 0.2621 | +0.0239 |

## Preservation Audit

| Dataset | Variant | Changed | Swaps | Base title-all@5 | Selector title-all@5 | Rescues | Regressions | Worsened | Worsened w/ gold-out |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | top4/max1 | 483 | 483 | 0.7060 | 0.8560 | 155 | 5 | 19 | 2 |
| HotpotQA | top4/max1 | 203 | 203 | 0.9050 | 0.9320 | 28 | 1 | 8 | 1 |
| MuSiQue | top4/max1 | 533 | 539 | 0.4600 | 0.4850 | 47 | 22 | 51 | 25 |
| MuSiQue | top3/max2 | 550 | 717 | 0.4600 | 0.4810 | 55 | 34 | 57 | 31 |

Depth-level preservation summary for the two most informative cases:

| Dataset | Variant | Gold docs | Title-all@5 | Rescues | Regressions | Worsened | Worsened w/ gold-out |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | top4/max1 | 2 | 0.9307 | 42 | 2 | 11 | 1 |
| 2Wiki | top4/max1 | 4 | 0.6128 | 113 | 3 | 8 | 1 |
| MuSiQue | top4/max1 | 4 | 0.0542 | 4 | 1 | 6 | 2 |
| MuSiQue | top3/max2 | 4 | 0.0602 | 6 | 2 | 7 | 3 |

## Interpretation

The preservation-first result is not MuSiQue-specific.  The same top4/max1
policy improves all three datasets:

- 2Wiki: `+0.0489` F1 overall and `+0.1080` on the 4-doc slice.
- HotpotQA: `+0.0156` F1 on the shallow 2-doc dataset.
- MuSiQue: `+0.0136` F1 overall and `+0.0362` on the 4-doc slice.

This is stronger than the earlier unrestricted stable DBEC result on 2Wiki and
HotpotQA, and it avoids the MuSiQue regression.  It keeps most of DBEC's 2Wiki
deep-chain value (`+0.1080` vs unrestricted stable DBEC's `+0.1165`) while
removing the MuSiQue damage.

The MuSiQue top3/max2 result answers the admission-slot question: simply
allowing two replacements by relaxing preservation from top4 to top3 is not the
right next step.  It is positive over baseline, but weaker than top4/max1:

- overall F1: `0.4379` vs `0.4468`;
- 4-doc F1: `0.2621` vs `0.2744`;
- worsened queries: `57` vs `51`;
- gold-out worsened queries: `31` vs `25`.

So the current bottleneck is not just "one more admission slot".  Relaxing
evidence preservation introduces enough damage to offset the extra admission
capacity.  ETv4-composition should therefore keep preservation as a hard
constraint and design a more principled admission/retention rule, rather than
expanding the number of free replacement slots.

## Decision

Use `top4/max1` as the current strongest parameter-level floor for
preservation-first ETv3-pool composition.

Do not promote `top3/max2` as the next line.  It is a useful negative diagnostic
showing that extra admission capacity without stronger retention is not enough.

The next method design should be:

```text
ETv4-composition = preservation-constrained residual evidence admission
```

with no-op as the default, DBEC utility as a soft admission score, and document
removal allowed only under a stronger retention test than raw ETv3 rank cutoff.
