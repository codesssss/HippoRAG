# DAEC/DBEC Depth x Gate-Resolvability Conditional Analysis

Date: 2026-05-08

Purpose: test the paper narrative that explicit dependency binding has conditional value rather than universal dominance. This report slices existing PropRAG full1000 outputs by support depth and the pre-selection DBEC-IG title-uniqueness gate decision.

Protocol: existing outputs only; no new LLM calls. `support_depth` is measured by gold support-document count, not manually annotated reasoning depth. `gate=bind` is a pre-selection binding-resolvability proxy from DBEC-IG's title-uniqueness gate; it is not an oracle identifiability label. CIs use query-paired percentile bootstrap with 10,000 resamples.

## Slice Profile

| Dataset | Slice | N | Mean Support Depth | Bind Rate | Abstain Rate | Mean Title-Unique Rate |
|---|---|---:|---:|---:|---:|---:|
| 2Wiki | `all` | 1000 | 2.4700 | 0.6980 | 0.3020 | 0.6980 |
| 2Wiki | `gate=bind` | 698 | 2.6676 | 1.0000 | 0.0000 | 1.0000 |
| 2Wiki | `gate=abstain` | 302 | 2.0132 | 0.0000 | 1.0000 | 0.0000 |
| 2Wiki | `support_depth>=3|gate=bind` | 233 | 4.0000 | 1.0000 | 0.0000 | 1.0000 |
| HotpotQA | `all` | 1000 | 2.0000 | 0.6570 | 0.3430 | 0.6570 |
| HotpotQA | `gate=bind` | 657 | 2.0000 | 1.0000 | 0.0000 | 1.0000 |
| HotpotQA | `gate=abstain` | 343 | 2.0000 | 0.0000 | 1.0000 | 0.0000 |
| MuSiQue | `all` | 1000 | 2.6480 | 0.3710 | 0.6290 | 0.4777 |
| MuSiQue | `gate=bind` | 371 | 2.5687 | 1.0000 | 0.0000 | 0.9997 |
| MuSiQue | `gate=abstain` | 629 | 2.6948 | 0.0000 | 1.0000 | 0.1698 |
| MuSiQue | `support_depth>=3|gate=bind` | 159 | 3.3270 | 1.0000 | 0.0000 | 0.9994 |
| MuSiQue | `support_depth>=4|gate=bind` | 52 | 4.0000 | 1.0000 | 0.0000 | 1.0000 |
| MuSiQue | `support_depth>=4|gate=abstain` | 114 | 4.0000 | 0.0000 | 1.0000 | 0.3782 |

## Main Conditional F1 Table

| Dataset | Slice | N | DBEC-IG | Nobinding | SetR-faithful | RankGPT-style | DBEC-IG - Nobind | 95% CI | DBEC-IG - SetR | 95% CI | DBEC-IG - RankGPT | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | `all` | 1000 | 0.7118 | 0.6120 | 0.6746 | 0.6659 | +0.0998 | [+0.0787, +0.1211] | +0.0372 | [+0.0155, +0.0596] | +0.0460 | [+0.0245, +0.0680] |
| 2Wiki | `gate=bind` | 698 | 0.6853 | 0.5422 | 0.6240 | 0.6194 | +0.1430 | [+0.1138, +0.1729] | +0.0613 | [+0.0342, +0.0888] | +0.0658 | [+0.0390, +0.0926] |
| 2Wiki | `gate=abstain` | 302 | 0.7732 | 0.7732 | 0.7916 | 0.7733 | +0.0000 | [+0.0000, +0.0000] | -0.0184 | [-0.0552, +0.0180] | -0.0000 | [-0.0353, +0.0362] |
| 2Wiki | `support_depth>=3|gate=bind` | 233 | 0.9142 | 0.7811 | 0.7778 | 0.8598 | +0.1331 | [+0.0815, +0.1846] | +0.1364 | [+0.0825, +0.1931] | +0.0544 | [+0.0114, +0.0987] |
| HotpotQA | `all` | 1000 | 0.7473 | 0.7387 | 0.7435 | 0.6845 | +0.0086 | [-0.0045, +0.0213] | +0.0038 | [-0.0148, +0.0229] | +0.0628 | [+0.0424, +0.0846] |
| HotpotQA | `gate=bind` | 657 | 0.7630 | 0.7500 | 0.7590 | 0.6815 | +0.0130 | [-0.0065, +0.0320] | +0.0041 | [-0.0181, +0.0262] | +0.0815 | [+0.0544, +0.1100] |
| HotpotQA | `gate=abstain` | 343 | 0.7172 | 0.7172 | 0.7139 | 0.6901 | +0.0000 | [+0.0000, +0.0000] | +0.0033 | [-0.0296, +0.0364] | +0.0271 | [-0.0020, +0.0564] |
| MuSiQue | `all` | 1000 | 0.4548 | 0.4458 | 0.4467 | 0.4093 | +0.0090 | [-0.0053, +0.0238] | +0.0082 | [-0.0185, +0.0343] | +0.0455 | [+0.0187, +0.0728] |
| MuSiQue | `gate=bind` | 371 | 0.4878 | 0.4636 | 0.5030 | 0.4649 | +0.0242 | [-0.0146, +0.0647] | -0.0152 | [-0.0581, +0.0281] | +0.0230 | [-0.0220, +0.0674] |
| MuSiQue | `gate=abstain` | 629 | 0.4353 | 0.4353 | 0.4134 | 0.3765 | +0.0000 | [+0.0000, +0.0000] | +0.0219 | [-0.0112, +0.0551] | +0.0588 | [+0.0246, +0.0946] |
| MuSiQue | `support_depth>=3|gate=bind` | 159 | 0.3432 | 0.3332 | 0.3406 | 0.3419 | +0.0100 | [-0.0506, +0.0725] | +0.0026 | [-0.0661, +0.0707] | +0.0013 | [-0.0626, +0.0675] |
| MuSiQue | `support_depth>=4|gate=bind` | 52 | 0.1912 | 0.2588 | 0.2788 | 0.2275 | -0.0676 | [-0.1699, +0.0283] | -0.0876 | [-0.1955, +0.0160] | -0.0362 | [-0.1285, +0.0555] |
| MuSiQue | `support_depth>=4|gate=abstain` | 114 | 0.3140 | 0.3140 | 0.2465 | 0.2310 | +0.0000 | [+0.0000, +0.0000] | +0.0675 | [-0.0147, +0.1511] | +0.0830 | [+0.0060, +0.1621] |

## Gate Fallback Check

| Dataset | Slice | DBEC-IG - Nobinding F1 | 95% CI | Exact fallback? |
|---|---|---:|---:|---|
| 2Wiki | `gate=abstain` | +0.0000 | [+0.0000, +0.0000] | yes |
| HotpotQA | `gate=abstain` | +0.0000 | [+0.0000, +0.0000] | yes |
| MuSiQue | `gate=abstain` | +0.0000 | [+0.0000, +0.0000] | yes |

## Key Findings

- On the 2Wiki high-support-depth, gate-bind slice, explicit binding is load-bearing: DBEC-IG beats DBEC-nobinding by `+0.1331` F1 with 95% CI `[+0.0815, +0.1846]`.
- The same 2Wiki slice also preserves the stronger-baseline story: DBEC-IG beats SetR-faithful by `+0.1364` F1 and RankGPT-style sliding by `+0.0544` F1.
- HotpotQA is not where the mechanism shows large marginal value: on gate-bind cases, DBEC-IG - DBEC-nobinding is only `+0.0130` F1.
- MuSiQue is the boundary case, not a clean positive replication: on support-depth>=4 and gate-bind, DBEC-IG trails DBEC-nobinding by `-0.0676` F1 and SetR-faithful by `-0.0876` F1.
- Therefore the paper-safe claim is conditional: explicit binding provides strong value on 2Wiki deep, resolvable dependencies, while MuSiQue exposes the ambiguity/budget boundary that motivates the identifiability gate and limitations section.

## Paper-Facing Interpretation

Allowed:

```text
DBEC-IG's explicit binding is strongly load-bearing on the 2Wiki high-support-depth gate-bind slice, but this is a conditional mechanism result rather than a universal advantage over set selection.
```

Allowed:

```text
MuSiQue exposes the boundary of the current binding instantiation: deep support count alone is insufficient when intermediate referents are ambiguous or reader budget is too tight.
```

Not allowed:

```text
Explicit binding is universally better on all deep multi-hop queries.
```

Not allowed:

```text
The gate decision is a ground-truth dependency identifiability label.
```

## Files

- Slice profile CSV: `reports/daec_depth_resolvability_20260508/slice_profile.csv`
- Method summary CSV: `reports/daec_depth_resolvability_20260508/method_summary.csv`
- Paired CI CSV: `reports/daec_depth_resolvability_20260508/paired_ci.csv`
- JSON: `reports/daec_depth_resolvability_20260508/summary.json`
