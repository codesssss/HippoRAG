# V13B Limit100 vs PropRAG Reflection

Date: 2026-05-03

This note checks whether the current V13B / minimal-evidence-interface line is
on the right research track.

Short answer:

```text
As a pure performance line: not yet.
As a diagnosis-driven research line: still worth continuing, but only if we
move beyond current V13B selector and avoid prompt/schema specialization.
```

## Protocol

All rows below are `limit=100`.

PropRAG numbers come from:

```text
reports/dpathrag/prop_neocorr_daec_limit100_20260429.json
```

Current V13B structural numbers come from:

```text
run_logs/v13b_structural_pool100_limit100_deg30_20260502/
```

V13B deg30 configuration:

| Item | Value |
|---|---|
| Candidate source | PropRAG pool100 |
| OpenIE | Original Qwen3-8B OpenIE |
| Graph expansion | Low-degree endpoint closure, max endpoint doc degree 30 |
| Selector | Obligation-closed STO local PPR |
| Lexical fallback | off |
| Partial obligation grounding | off |
| V13B-specific prompt | not used |

Important caveat:

```text
The current structural deg30 run has retrieval metrics, not QA metrics.
QA-comparable V13B numbers exist for the earlier dense-adapter run, but that is
not the current structural mainline.
```

## Retrieval Comparison

| Dataset | PropRAG R@5 | V13B deg30 R@5 | Delta | PropRAG + DAEC R@5 | V13B vs DAEC | V13B all-gold@5 | V13B feasible | V13B changed | V13B gains / losses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.9350 | 0.9550 | +0.0200 | 0.9525 | +0.0025 | 0.9200 | 53 | 38 | 7 / 0 |
| HotpotQA | 0.9300 | 0.9300 | +0.0000 | 0.9500 | -0.0200 | 0.8700 | 28 | 18 | 0 / 0 |
| MuSiQue | 0.6775 | 0.6825 | +0.0050 | 0.7175 | -0.0350 | 0.4000 | 23 | 14 | 1 / 0 |

Interpretation:

```text
Current V13B deg30 does not establish a strong performance claim over PropRAG.
It has a clean 2Wiki retrieval gain, ties HotpotQA, and barely improves MuSiQue.
Against PropRAG + DAEC, it is behind on HotpotQA and MuSiQue.
```

## Prompt Probe Check

Claude's V13B-specific OpenIE prompt was a useful probe, but it should not be
the mainline method.

| Dataset | V13B deg30 R@5 | V13B prompt R@5 | Delta | deg30 feasible | prompt feasible | deg30 gains/losses | prompt gains/losses |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.9550 | 0.9525 | -0.0025 | 53 | 58 | 7 / 0 | 6 / 0 |
| HotpotQA | 0.9300 | 0.9300 | +0.0000 | 28 | 27 | 0 / 0 | 0 / 0 |
| MuSiQue | 0.6825 | 0.6825 | +0.0000 | 23 | 18 | 1 / 0 | 1 / 0 |

Exact gold OpenIE frame rate:

| Dataset | Original Qwen | V13B prompt | Change |
|---|---:|---:|---:|
| 2Wiki | 42.73% | 49.09% | +6.36 |
| HotpotQA | 17.13% | 12.96% | -4.17 |
| MuSiQue | 9.49% | 10.28% | +0.79 |

Interpretation:

```text
The prompt confirms that corpus-side evidence representation matters, but it is
not robust. It improves 2Wiki frame rate, hurts HotpotQA, and does not improve
retrieval.
```

## QA-Comparable Dense-Adapter Numbers

These are included only for scale. They come from:

```text
run_logs/v13b_proprag_pool100_limit100_20260430/
```

This run used a dense PropRAG pool adapter, not the current structural deg30
source report.

| Dataset | PropRAG EM | Dense-adapter V13B EM | Delta | PropRAG F1 | Dense-adapter V13B F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.5800 | 0.6000 | +0.0200 | 0.6318 | 0.6479 | +0.0161 |
| HotpotQA | 0.5700 | 0.5700 | +0.0000 | 0.6912 | 0.6912 | +0.0000 |
| MuSiQue | 0.3800 | 0.3900 | +0.0100 | 0.4374 | 0.4474 | +0.0100 |

Interpretation:

```text
Reader-level deltas are small and belong to a non-mainline dense-adapter run.
They should not be used as proof that the current V13B graph method is solved.
```

## Source-Support Diagnosis

The strongest reason to continue is not current R@5. It is the Phase 1
source-support audit.

| Dataset | Retrieval-critical obligations | OpenIE exact available | OpenIE failed but source supports | Unresolved/descriptive query issue | Main signal |
|---|---:|---:|---:|---:|---|
| 2Wiki | 220 | 127 / 57.73% | 18 / 8.18% | 22 / 10.00% | OpenIE exact often available |
| HotpotQA | 216 | 38 / 17.59% | 58 / 26.85% | 39 / 18.06% | Source support exceeds OpenIE exact gap |
| MuSiQue | 253 | 26 / 10.28% | 37 / 14.62% | 85 / 33.60% | Query compiler or binding bottleneck |

Interpretation:

```text
HotpotQA is the strongest evidence that fixed OpenIE relation labels are the
wrong abstraction: 26.85% of retrieval-critical demands have source evidence
available even when exact OpenIE grounding fails.

MuSiQue should not be attacked by better matching first. Its visible bottleneck
is query-demand / variable-binding quality.

2Wiki is not a good optimization target now because exact OpenIE already works
relatively often; it can encourage overfitting prompt/schema design.
```

## Research-Line Judgment

Current V13B deg30 is not yet competitive enough as a final method.

| Weak point | Evidence |
|---|---|
| Narrow retrieval gain | +2.0pp R@5 on 2Wiki, tied on HotpotQA, +0.5pp on MuSiQue vs PropRAG. |
| Behind stronger prior rerank line | -2.0pp vs PropRAG + DAEC on HotpotQA, -3.5pp on MuSiQue. |
| Prompt route is unstable | HotpotQA exact frame rate regresses under V13B-specific prompt. |
| QA claim is not aligned yet | Structural deg30 has no QA run; dense-adapter QA is not mainline. |

But the minimal evidence interface line is still research-worthy if the next
step targets the source-support gap:

```text
OpenIE relation labels are often too brittle. Source-grounded evidence units may
support query demands without requiring a benchmark-specific relation schema.
```

## Decision

Continue, but do not claim current V13B beats PropRAG in a meaningful general
sense.

Correct next experiment:

```text
HotpotQA limit100:
Build a diagnostic schema-light source-grounded demand matcher probe.
Measure whether it converts "OpenIE failed but source supports" cases into valid
demand matches without relation synonym tables or V13B-specific prompts.
```

Do not do these yet:

```text
Do not tune PPR.
Do not tune alpha/path length.
Do not add fallback.
Do not continue V13B-specific OpenIE prompt engineering.
Do not use fixed relation schema as the method.
Do not run full1000 before the Hotpot source-grounded matcher shows mechanism gain.
```

Go/no-go criterion:

```text
If Hotpot source-grounded matching cannot convert the 26.85% source-support gap
into better evidence-cover feasibility, downgrade this line.

If it can, continue toward Minimal Connected Evidence Cover.
```

## Matcher Probe Update

The HotpotQA limit100 source-grounded matcher probe has now been implemented:

```text
build_minimal_evidence_units.py
match_query_demands_to_evidence_units.py
```

It follows the intended clean constraints:

```text
No fixed relation schema.
No relation synonym table.
No selector/PPR/fallback change.
No V13B-specific OpenIE prompt.
No selector pseudo-match seed by default.
```

Gold-doc diagnostic result:

| Dataset | Retrieval-critical obligations | OpenIE exact | New source-grounded matches | Exact or source grounded |
|---|---:|---:|---:|---:|
| 2Wiki | 220 | 127 / 57.73% | 19 / 8.64% | 146 / 66.36% |
| HotpotQA | 216 | 38 / 17.59% | 56 / 25.93% | 94 / 43.52% |
| MuSiQue | 253 | 26 / 10.28% | 39 / 15.42% | 65 / 25.69% |

HotpotQA candidate-doc diagnostic result:

| Dataset | Retrieval-critical obligations | OpenIE exact | New source-grounded matches | Exact or source grounded |
|---|---:|---:|---:|---:|
| HotpotQA candidate docs | 216 | 41 / 18.98% | 65 / 30.09% | 106 / 49.07% |

What this means:

```text
The line is not justified by current V13B R@5. It is justified by a mechanism
signal: HotpotQA contains many retrieval-critical demands where source-grounded
sentence evidence is available even though exact OpenIE fact grounding fails.
```

What this does not mean:

```text
This is not yet a retrieval gain.
This is not yet a QA gain.
This is not yet selector-ready because variable bindings can still be broad
explicit mention sets.
```

Updated next step:

```text
Implement Minimal Connected Evidence Cover over source-grounded units.
The selector may change top5 only when it covers retrieval-critical demands with
a binding-consistent connected evidence set.
```

## Minimal Connected Cover Update

Implemented diagnostic-only cover:

```text
select_minimal_connected_evidence_cover.py
```

Limit100 result, aligned against current V13B selector and PropRAG-family
baselines:

| Dataset | PropRAG R@5 | PropRAG+DAEC R@5 | V13B selector R@5 | Source-grounded cover R@5 | Cover vs selector | Cover vs DAEC |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.9350 | 0.9525 | 0.9550 | 0.9600 | +0.0050 | +0.0075 |
| HotpotQA | 0.9300 | 0.9500 | 0.9300 | 0.9400 | +0.0100 | -0.0100 |
| MuSiQue | 0.6775 | 0.7175 | 0.6825 | 0.6892 | +0.0067 | -0.0283 |

All-gold@5 and regression view:

| Dataset | V13B selector all-gold@5 | Cover all-gold@5 | Gains/losses vs selector |
|---|---:|---:|---:|
| 2Wiki | 0.9200 | 0.9100 | 2 / 2 |
| HotpotQA | 0.8700 | 0.8900 | 2 / 0 |
| MuSiQue | 0.4000 | 0.4100 | 3 / 1 |

What this means:

```text
The source-grounded evidence-cover direction is now stronger than a pure
diagnostic: it produces small but real retrieval-side gains on limit100.

It still does not beat PropRAG+DAEC on HotpotQA or MuSiQue, and it is not safe
as an unconditional selector replacement because 2Wiki all-gold@5 regresses.
```

Correct next integration principle:

```text
Do not fuse scores.
Do not add a new weight.
Do not replace selector globally.

Use the source-grounded cover only as a structural certifier:
override current top5 when the cover is binding-consistent, connected, and does
not reduce already-covered gold evidence; otherwise keep the current selector.
```

Since gold cannot be used by the method, a first non-oracle certificate was
tested:

```text
admit cover only when full_cover_feasible and connected; otherwise keep selector
top5.
```

Certified admission result:

| Dataset | Admitted covers | Certified R@5 | Selector R@5 | Certified all-gold@5 | Selector all-gold@5 | Gains/losses vs selector |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | 23/100 | 0.9575 | 0.9550 | 0.9200 | 0.9200 | 1 / 1 |
| HotpotQA | 25/100 | 0.9350 | 0.9300 | 0.8800 | 0.8700 | 1 / 0 |
| MuSiQue | 21/100 | 0.6825 | 0.6825 | 0.4000 | 0.4000 | 0 / 0 |

Updated judgment:

```text
The clean path is viable but still fragile.

Unconditional source-grounded cover gives larger gains but has regressions.
The full-cover-and-connected certificate is safer, but still has one 2Wiki
query-level loss.  The next useful work is admitted-loss analysis, not more
parameterization.
```

Admitted-loss analysis:

```text
The 2Wiki loss is not caused by bad source-grounded matching.  It is caused by
query-demand scope: a same-country comparison query had country constraints
materialized but marked non-retrieval, so the cover certified only director
evidence and missed one country evidence document.
```

Materialized-obligation diagnostic:

| Dataset | Certified R@5 | Selector R@5 | Certified all-gold@5 | Selector all-gold@5 | Gains/losses vs selector |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 0.9600 | 0.9550 | 0.9300 | 0.9200 | 1 / 0 |
| HotpotQA | 0.9350 | 0.9300 | 0.8800 | 0.8700 | 1 / 0 |
| MuSiQue | 0.6792 | 0.6825 | 0.3900 | 0.4000 | 0 / 1 |

This rules out a global materialized-obligation switch:

```text
It fixes the 2Wiki admitted loss but hurts MuSiQue.  The next clean algorithmic
problem is to decide which materialized constraints are retrieval-critical from
the query-demand graph itself.
```
