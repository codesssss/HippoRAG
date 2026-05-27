# ETv4-Composition A0 Selector-Only Dominance Probe

Date: 2026-05-10

## Scope

This note records A0 for the proposed ETv4-composition method:
fixed-budget residual admission under heterogeneous retention dominance and
positive DBEC set-marginal utility.

The experiment is selector-only.  It does not run reader QA.  It keeps ETv3
retrieval frozen and reuses the existing DBEC demand decomposition and LLM
binding caches from `run_logs/etv3_dbec_latest_full1000_20260510/`.

The question is whether cross-signal agreement retention can replace the
rank-cutoff preservation used by `top4/max1`.

## Implementation

Runner:

```text
scripts/run_etv4_composition_a0_selector.py
```

DBEC objective change:

```text
scripts/dtc_embed_utils.py
```

The new path is opt-in through `safe_projection_mode=agreement_*`; the old
rank-cutoff safe projection remains the default `rank_cutoff` mode.

Swap rule:

```text
swap d -> c iff R(c) >= R(d) and ΔU(c, d | S) > 0
```

or strict:

```text
swap d -> c iff R(c) > R(d) and ΔU(c, d | S) > 0
```

where:

- `R2 = ETv3 top5 vote + dense/source-prior top5 vote`
- `R3 = ETv3 top5 vote + dense/source-prior top5 vote + DBEC greedy top5 vote`
- `ΔU(c, d | S)` is the DBEC noisy-OR set objective gain for replacing `d`
  with `c`
- retained document order is preserved because the admitted document fills the
  displaced slot

Variants:

| Variant | Retention signal | Rule |
| --- | --- | --- |
| `r2_ge` | ETv3 + dense | `R(c) >= R(d)` and `ΔU > 0` |
| `r2_gt` | ETv3 + dense | `R(c) > R(d)` and `ΔU > 0` |
| `r3_ge` | ETv3 + dense + DBEC | `R(c) >= R(d)` and `ΔU > 0` |
| `r3_gt` | ETv3 + dense + DBEC | `R(c) > R(d)` and `ΔU > 0` |

Run root:

```text
run_logs/etv4_composition_a0_selector_full1000_20260510/
```

Cache status:

| Dataset | Frozen requirements | Binding cache entries | Binding cache misses |
| --- | ---: | ---: | ---: |
| 2Wiki | 1000 | 7527 | 0 |
| HotpotQA | 1000 | 6323 | 0 |
| MuSiQue | 999 | 6746 | 0 |

MuSiQue has one query without a frozen requirement trace; it falls back to the
existing no-requirement behavior.

## Overall Results

| Dataset | Variant | Title-all@5 | Δ all@5 | Title recall@5 | Changed | Swaps | Gold-out queries |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | ETv3 baseline | 0.7060 | 0.0000 | 0.9015 | 0 | 0 | 0 |
| 2Wiki | rank top4/max1 | 0.8560 | +0.1500 | 0.9503 | 483 | 483 | 13 |
| 2Wiki | rank top3/max2 | **0.8790** | **+0.1730** | **0.9563** | 487 | 539 | 19 |
| 2Wiki | `r2_ge` | 0.7070 | +0.0010 | 0.8998 | 70 | 70 | 13 |
| 2Wiki | `r2_gt` | 0.7060 | +0.0000 | 0.9015 | 0 | 0 | 0 |
| 2Wiki | `r3_ge` | 0.7200 | +0.0140 | 0.9035 | 178 | 178 | 55 |
| 2Wiki | `r3_gt` | 0.7080 | +0.0020 | 0.9012 | 16 | 16 | 7 |
| HotpotQA | ETv3 baseline | 0.9050 | 0.0000 | 0.9505 | 0 | 0 | 0 |
| HotpotQA | rank top4/max1 | **0.9320** | **+0.0270** | **0.9645** | 203 | 203 | 7 |
| HotpotQA | `r2_ge` | 0.9050 | +0.0000 | 0.9505 | 0 | 0 | 0 |
| HotpotQA | `r2_gt` | 0.9050 | +0.0000 | 0.9505 | 0 | 0 | 0 |
| HotpotQA | `r3_ge` | 0.9050 | +0.0000 | 0.9505 | 0 | 0 | 0 |
| HotpotQA | `r3_gt` | 0.9050 | +0.0000 | 0.9505 | 0 | 0 | 0 |
| MuSiQue | ETv3 baseline | 0.4600 | 0.0000 | 0.7391 | 0 | 0 | 0 |
| MuSiQue | rank top4/max1 | **0.4850** | **+0.0250** | **0.7551** | 522 | 539 | 137 |
| MuSiQue | `r2_ge` | 0.4600 | +0.0000 | 0.7385 | 3 | 3 | 2 |
| MuSiQue | `r2_gt` | 0.4600 | +0.0000 | 0.7391 | 0 | 0 | 0 |
| MuSiQue | `r3_ge` | 0.4580 | -0.0020 | 0.7369 | 7 | 13 | 6 |
| MuSiQue | `r3_gt` | 0.4600 | +0.0000 | 0.7391 | 0 | 0 | 0 |

## Depth Slices

### 2Wiki

| Variant | Slice | Title-all@5 | Δ all@5 | Title recall@5 | Swaps | Gold-out queries |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `r2_ge` | 2-doc | 0.8797 | +0.0013 | 0.9399 | 59 | 2 |
| `r2_ge` | 4-doc | 0.1447 | +0.0000 | 0.7691 | 11 | 11 |
| `r3_ge` | 2-doc | 0.9033 | +0.0248 | 0.9516 | 120 | 8 |
| `r3_ge` | 4-doc | 0.1234 | -0.0213 | 0.7468 | 58 | 47 |
| `r3_gt` | 2-doc | 0.8810 | +0.0026 | 0.9405 | 10 | 1 |
| `r3_gt` | 4-doc | 0.1447 | +0.0000 | 0.7734 | 6 | 6 |

### MuSiQue

| Variant | Slice | Title-all@5 | Δ all@5 | Title recall@5 | Swaps | Gold-out queries |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `r2_ge` | 2-doc | 0.6737 | +0.0000 | 0.8292 | 0 | 0 |
| `r2_ge` | 3-doc | 0.3323 | +0.0000 | 0.7247 | 2 | 1 |
| `r2_ge` | 4-doc | 0.0361 | +0.0000 | 0.4819 | 1 | 1 |
| `r3_ge` | 2-doc | 0.6699 | -0.0039 | 0.8272 | 3 | 2 |
| `r3_ge` | 3-doc | 0.3323 | +0.0000 | 0.7236 | 7 | 3 |
| `r3_ge` | 4-doc | 0.0361 | +0.0000 | 0.4804 | 3 | 1 |
| `r3_gt` | 2-doc | 0.6737 | +0.0000 | 0.8292 | 0 | 0 |
| `r3_gt` | 3-doc | 0.3323 | +0.0000 | 0.7257 | 0 | 0 |
| `r3_gt` | 4-doc | 0.0361 | +0.0000 | 0.4834 | 0 | 0 |

## Interpretation

A0 is a clean negative result for the current cross-signal agreement dominance
instantiation.

`R2` is too conservative.  Because ETv3 and dense/source-prior top5 often
corroborate the same baseline documents, few residual candidates can pass the
retention gate.  This makes `r2_gt` inert and `r2_ge` nearly inert.

`R3_ge` admits more candidates, but it does not recover the rank-cutoff
frontier.  On 2Wiki it improves title-all@5 by only `+0.0140`, far below
top4/max1 (`+0.1500`) and top3/max2 (`+0.1730`), while causing much more
gold-out than top4/max1 (`55` vs `13`).  It also hurts the important 2Wiki 4-doc
slice (`0.1447 -> 0.1234`).

On MuSiQue, agreement dominance does not move the set-level metric in the right
direction.  `r3_ge` slightly reduces overall title-all@5 (`0.4600 -> 0.4580`);
strict variants are inert.

HotpotQA is fully inert under all four variants, while rank top4/max1 had
previously improved title-all@5 (`0.9050 -> 0.9320`).

## Decision

Do not promote this agreement-dominance instantiation to reader QA.  It fails
the A0 selector-only gate:

- no variant is Pareto-competitive with rank top4/max1;
- `R2` is too inert to be useful;
- `R3_ge` is a weak positive on 2Wiki overall but damages the 4-doc slice and
  raises gold-out risk;
- MuSiQue and HotpotQA do not benefit.

The result does not refute preservation-constrained residual admission.  It
does show that naive cross-signal top5 agreement is not a better retention
object than ETv3 rank-cutoff preservation on this substrate.

Paper-facing implication:

> Rank position in ETv3 is a stronger retention proxy than simple heterogeneous
> top5 agreement.  The robust `top4/max1` result should be treated as a strong
> minimal instantiation rather than an obvious trick to be replaced by naive
> agreement counting.

Next method work should not continue by adding weights to agreement signals.
That would reintroduce the same trick smell A0 was designed to avoid.  A
cleaner follow-up would need a different evidence-level retention object, or the
paper can keep rank-cutoff preservation as the empirically strongest minimal
composition instantiation and report this A0 as a negative diagnostic.
