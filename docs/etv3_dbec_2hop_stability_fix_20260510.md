# ETv3 + DBEC 2-Hop Stability Fix

## Problem

The first ETv3+DBEC run used full DBEC reconstruction:

```text
--setwise_selector daec_noisyor_llm
```

It improved MuSiQue 3-hop and 4-hop slices, but hurt 2-hop:

| subset | ETv3 F1 | full-rebuild DBEC F1 | delta |
| --- | ---: | ---: | ---: |
| 2-doc | 0.5837 | 0.5489 | -0.0347 |
| 3-doc | 0.3800 | 0.4356 | +0.0556 |
| 4-doc | 0.1545 | 0.2000 | +0.0455 |
| all | 0.4282 | 0.4382 | +0.0100 |

Flip audit:

| subset | wins | losses | ties | changed top5 |
| --- | ---: | ---: | ---: | ---: |
| 2-doc | 3 | 4 | 41 | 40/48 |
| 3-doc | 6 | 3 | 21 | 30/30 |
| 4-doc | 2 | 0 | 20 | 19/22 |

The 2-hop loss was caused by full reconstruction being too invasive.  On
shorter queries, ETv3 top5 is often already sufficient.  DBEC then saturates
its noisy-OR objective after one or two demand-covering documents, but still
rebuilds/reorders the reader-facing top5.  That can displace useful ETv3 tail
evidence or change the reader order without a real dependency-composition need.

## Fix

Use DBEC as a local edit objective over ETv3 top5:

```text
--setwise_selector daec_noisyor_safe_llm
--daec_safe_min_objective_gain 0
--daec_safe_min_swap_gain 0.000001
--daec_safe_max_swaps 2
--daec_safe_preserve_top_m 1
```

This is not a hop-specific fallback.  It does not inspect whether the query is
2-hop.  It changes the composition semantics from:

```text
rebuild the whole top5 from DBEC noisy-OR
```

to:

```text
start from ETv3 top5; apply only strict positive DBEC-objective swaps
```

## Result

| subset | ETv3 F1 | full-rebuild DBEC F1 | stable DBEC F1 | stable delta |
| --- | ---: | ---: | ---: | ---: |
| 2-doc | 0.5837 | 0.5489 | 0.5987 | +0.0150 |
| 3-doc | 0.3800 | 0.4356 | 0.4111 | +0.0311 |
| 4-doc | 0.1545 | 0.2000 | 0.2000 | +0.0455 |
| all | 0.4282 | 0.4382 | 0.4547 | +0.0265 |

Retrieval R@5 under stable DBEC is `0.7008`, slightly below ETv3's `0.7092`,
but QA F1 is higher.  The improvement therefore comes from a better
reader-facing composition, not from a simple gold-document recall increase.

## Conservative Diagnostic

The existing thresholded minimal-edit default also worked:

```text
--daec_safe_min_objective_gain 0.02
--daec_safe_min_swap_gain 0.01
```

| subset | F1 |
| --- | ---: |
| all | 0.4575 |
| 2-doc | 0.5906 |
| 3-doc | 0.4000 |
| 4-doc | 0.2455 |

This is the best MuSiQue limit100 number so far, but it uses explicit gain
thresholds.  Keep it as a diagnostic until it is validated across datasets.

## Consequence For ETv4

The immediate 2-hop problem is solved by making DBEC baseline-stable.  ETv4
should not be another full top5 reconstruction or a hop router.  The next clean
ETv4 direction should focus on long-chain residual failure:

1. Keep the baseline-stable local-edit principle.
2. Diagnose 4-hop failures where candidate pool has all supports but stable
   DBEC still does not select/arrange them.
3. Decide whether the residual is binding failure, top5 budget, or reader
   ordering before changing the retrieval graph again.

## ETv4 Reader-Budget Follow-Up

A direct stack of stable DBEC top5 plus ETv3 candidate-prefix reader@10 was also
tested:

| method | all F1 | 2-doc F1 | 3-doc F1 | 4-doc F1 |
| --- | ---: | ---: | ---: | ---: |
| ETv3 top5 | 0.4282 | 0.5837 | 0.3800 | 0.1545 |
| ETv4 candidate-prefix reader@10 | 0.4247 | 0.5486 | 0.3711 | 0.2273 |
| stable DBEC top5 | 0.4547 | 0.5987 | 0.4111 | 0.2000 |
| stable DBEC top5 + candidate-prefix reader@10 | 0.4164 | 0.5694 | 0.3212 | 0.2121 |

So ETv4 should not simply append more documents.  The auxiliary context must be
selected by residual demand, not by raw candidate prefix.  A clean next ETv4
candidate is:

```text
primary context: baseline-stable DBEC top5
auxiliary context: append only documents with positive residual DBEC-demand gain
reader presentation: keep primary top5 first; auxiliary passages are extra context
```

This keeps the same DBEC objective and avoids a hop-specific router, while
targeting the actual failure mode: long-chain queries need a few missing
support passages, but short queries are harmed by noisy extra context.
