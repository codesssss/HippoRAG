# PCEC m=4 Mechanism Audit

Date: 2026-05-11

Scope: offline 8B full1000 title-level analysis over existing ETv3+DBEC/PCEC
frontier artifacts.  No reader rerun is included in this audit.

Outputs:

```text
reports/pcec_m4_mechanism_8b_20260511/summary.json
reports/pcec_m4_mechanism_8b_20260511/summary.md
```

Analysis script:

```text
scripts/analyze_pcec_m4_mechanism.py
```

## Question

The paper-facing PCEC default uses reader budget `K=5` and preservation prefix
`m=4`, i.e. one residual admission slot.  Prior evidence showed that this point
is robust, but not why the boundary should be at rank 4.

This audit tests whether `m=4` has a mechanism-level explanation.

## Main Finding

`m=4` should be framed as a robust `K=5` operating point, not as a universal
optimum.  The mechanism is not a score cliff.  The stored `pool_doc_scores` are
rank-coded (`200..196`, then `95..`), so a rank-4/5 score-gap argument would be
circular.

The useful mechanism is instead slot vulnerability:

```text
rank 4 is consistently more gold-critical than rank 5;
allowing replacement of rank 4 and earlier sharply increases gold-out risk.
```

## Baseline Slot Vulnerability

| Dataset | Top4 all-gold | Top5 all-gold | Rank4 critical | Rank5 critical |
| --- | ---: | ---: | ---: | ---: |
| 2Wiki | 68.6% | 70.6% | 6.5% | 2.6% |
| HotpotQA | 88.0% | 90.5% | 10.2% | 3.0% |
| MuSiQue | 40.8% | 46.0% | 22.7% | 14.5% |

`Rank5 critical` means rank 5 is gold and the top-4 prefix is not already
all-gold.  This is the direct gold-out risk of replacing slot 5.  `Rank4
critical` is the analogous risk introduced by allowing replacement of slot 4.

The gap is strongest on 2Wiki and HotpotQA.  MuSiQue is harder: even rank 5 is
frequently gold-critical, which explains why PCEC remains conservative and why
looser policies are risky.

## Swap Hazard

| Dataset | Variant | Swaps | Gold-out queries | Complete rescues | Complete regressions |
| --- | --- | ---: | ---: | ---: | ---: |
| 2Wiki | top4/max1 | 483 | 13 | 155 | 5 |
| 2Wiki | top3/max2 | 539 | 19 | 179 | 6 |
| 2Wiki | top2/max3 | 540 | 39 | 178 | 18 |
| HotpotQA | top4/max1 | 203 | 7 | 28 | 1 |
| HotpotQA | top3/max2 | 224 | 15 | 29 | 8 |
| HotpotQA | top2/max3 | 228 | 37 | 28 | 27 |
| MuSiQue | top4/max1 | 539 | 137 | 47 | 22 |
| MuSiQue | top3/max2 | 717 | 174 | 55 | 34 |
| MuSiQue | top2/max3 | 784 | 241 | 52 | 50 |

The pattern is consistent: relaxing below `m=4` adds some rescues, but the
gold-out/regression cost grows faster.  This is clearest on HotpotQA and
MuSiQue.  2Wiki is the exception where `top3/max2` is reader-positive, which is
why the paper must not claim `m=4` is universally optimal.

## Oracle Variant View

For each query, we chose the largest preservation prefix among the already-run
frontier variants that achieves the maximum title coverage for that query.

| Dataset | Non-baseline oracle need | m=4 share | Below-m4 share |
| --- | ---: | ---: | ---: |
| 2Wiki | 184 (18.4%) | 158 (85.9%) | 26 (14.1%) |
| HotpotQA | 30 (3.0%) | 29 (96.7%) | 1 (3.3%) |
| MuSiQue | 117 (11.7%) | 79 (67.5%) | 38 (32.5%) |

Interpretation:

- Most queries do not require composition beyond ETv3 top-5.
- Among queries where composition matters, `m=4` captures most of the
  conservative oracle need on 2Wiki and HotpotQA.
- MuSiQue has a larger below-m4 oracle tail, but the reader and gold-out results
  show that looser fixed cutoffs damage context too often.

## Paper Claim

Use this wording:

> For `K=5`, `m=4` is a robust preservation-first operating point because it
> treats the fifth rank as the residual admission slot while preserving the
> more gold-critical top-4 prefix.  Offline oracle analysis shows that this
> one-slot boundary captures most non-baseline composition opportunities on
> 2Wiki and HotpotQA, while looser fixed cutoffs increase gold displacement and
> reader-context risk, especially on MuSiQue.

Do not write:

```text
m=4 is universally optimal.
```

The correct limitation is:

```text
The default m=4 is scoped to K=5 and to the frozen ETv3 confidence ordering.
Adaptive per-query retention remains future work.
```
