# ETv3 Native Full1000 Failure Audit

Date: 2026-05-10

This diagnostic audits the completed ETv3 variable-flow full1000 run as a
native method-owned retrieval/readout line.  It does not use PropRAG pools,
does not call an LLM, and does not modify ETv3/ETv4 method code.

## Core Question

For ETv3 failures, is the evidence missing from the method-owned candidate
universe, present in candidate200 but not composed into top5, or already in
top5 but not answered exactly by the reader?

## Overall

| Dataset | Count | EM | F1 | all_gold@5 | all_gold@20 | all_gold@100 | all_gold@200 | R@5 | R@20 | R@100 | R@200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 1000 | 0.5860 | 0.6601 | 0.7060 | 0.7940 | 0.9720 | 0.9750 | 0.9015 | 0.9367 | 0.9882 | 0.9895 |
| hotpotqa | 1000 | 0.6170 | 0.7330 | 0.9050 | 0.9780 | 0.9940 | 0.9940 | 0.9505 | 0.9885 | 0.9965 | 0.9965 |
| musique | 1000 | 0.3320 | 0.4319 | 0.4200 | 0.6410 | 0.8140 | 0.8610 | 0.7184 | 0.8481 | 0.9268 | 0.9468 |

## R@k vs Set Completeness

`R@k` measures average per-document support recall.  Multi-hop QA, however,
needs the evidence set to be complete.  `all_gold@k` is therefore the
stricter composition metric: all supporting documents must be present in
the reader context.  The large `R@5 - all_gold@5` gap shows why pointwise
recall can overstate long-chain retrieval quality.

| Dataset | R@5 | all_gold@5 | R@5 - all_gold@5 | R@200 | all_gold@200 | R@200 - all_gold@200 |
|---|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 0.9015 | 0.7060 | 0.1955 | 0.9895 | 0.9750 | 0.0145 |
| hotpotqa | 0.9505 | 0.9050 | 0.0455 | 0.9965 | 0.9940 | 0.0025 |
| musique | 0.7184 | 0.4200 | 0.2984 | 0.9468 | 0.8610 | 0.0858 |

## Depth Breakdown

| Dataset | Gold docs | Count | EM | F1 | all_gold@5 | all_gold@20 | all_gold@100 | all_gold@200 | R@5 | R@20 | R@100 | R@200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 2 | 765 | 0.5359 | 0.6320 | 0.8784 | 0.9464 | 0.9752 | 0.9778 | 0.9392 | 0.9732 | 0.9876 | 0.9889 |
| 2wikimultihopqa | 4 | 235 | 0.7489 | 0.7516 | 0.1447 | 0.2979 | 0.9617 | 0.9660 | 0.7787 | 0.8181 | 0.9904 | 0.9915 |
| hotpotqa | 2 | 1000 | 0.6170 | 0.7330 | 0.9050 | 0.9780 | 0.9940 | 0.9940 | 0.9505 | 0.9885 | 0.9965 | 0.9965 |
| musique | 2 | 518 | 0.4170 | 0.5168 | 0.6467 | 0.8224 | 0.9228 | 0.9344 | 0.8147 | 0.9102 | 0.9604 | 0.9672 |
| musique | 3 | 316 | 0.2943 | 0.3946 | 0.2595 | 0.5823 | 0.8006 | 0.8418 | 0.6951 | 0.8460 | 0.9314 | 0.9473 |
| musique | 4 | 166 | 0.1386 | 0.2382 | 0.0181 | 0.1867 | 0.5000 | 0.6687 | 0.4623 | 0.6581 | 0.8133 | 0.8825 |

## Selection Gap by Depth

The selection gap is not unique to 4-doc examples; it grows with evidence
depth.  This is the full1000 version of the Expand-then-Compose failure
pattern: the candidate universe often contains the set, but the top5
readout does not compose the full support set.

| Dataset | Gold docs | Count | all_gold@5 | all_gold@200 | all_gold@200 - all_gold@5 | R@5 - all_gold@5 |
|---|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 2 | 765 | 0.8784 | 0.9778 | 0.0993 | 0.0608 |
| 2wikimultihopqa | 4 | 235 | 0.1447 | 0.9660 | 0.8213 | 0.6340 |
| hotpotqa | 2 | 1000 | 0.9050 | 0.9940 | 0.0890 | 0.0455 |
| musique | 2 | 518 | 0.6467 | 0.9344 | 0.2876 | 0.1680 |
| musique | 3 | 316 | 0.2595 | 0.8418 | 0.5823 | 0.4357 |
| musique | 4 | 166 | 0.0181 | 0.6687 | 0.6506 | 0.4443 |

## Failure Buckets

Bucket definitions:

- `top5_complete_answer_exact`: all gold docs are in top5 and reader exact match is correct.
- `top5_complete_reader_not_exact`: all gold docs are in top5, but reader exact match is not correct.
- `candidate200_complete_top5_incomplete`: all gold docs are in ETv3 candidate200, but not all are selected into top5.
- `candidate200_incomplete_top5_partial`: at least one gold doc is in top5, but candidate200 still misses at least one gold doc.
- `candidate200_incomplete_not_top5`: candidate200 has at least one gold doc, but top5 has none and candidate200 is incomplete.
- `candidate200_no_gold`: candidate200 contains no gold support.

| Dataset | Gold docs | Bucket | Count |
|---|---:|---|---:|
| 2wikimultihopqa | 2 | `candidate200_complete_top5_incomplete` | 76 |
| 2wikimultihopqa | 2 | `candidate200_incomplete_top5_partial` | 17 |
| 2wikimultihopqa | 2 | `top5_complete_answer_exact` | 409 |
| 2wikimultihopqa | 2 | `top5_complete_reader_not_exact` | 263 |
| 2wikimultihopqa | 4 | `candidate200_complete_top5_incomplete` | 193 |
| 2wikimultihopqa | 4 | `candidate200_incomplete_top5_partial` | 8 |
| 2wikimultihopqa | 4 | `top5_complete_answer_exact` | 33 |
| 2wikimultihopqa | 4 | `top5_complete_reader_not_exact` | 1 |
| hotpotqa | 2 | `candidate200_complete_top5_incomplete` | 89 |
| hotpotqa | 2 | `candidate200_incomplete_top5_partial` | 5 |
| hotpotqa | 2 | `candidate200_no_gold` | 1 |
| hotpotqa | 2 | `top5_complete_answer_exact` | 599 |
| hotpotqa | 2 | `top5_complete_reader_not_exact` | 306 |
| musique | 2 | `candidate200_complete_top5_incomplete` | 149 |
| musique | 2 | `candidate200_incomplete_not_top5` | 2 |
| musique | 2 | `candidate200_incomplete_top5_partial` | 32 |
| musique | 2 | `top5_complete_answer_exact` | 192 |
| musique | 2 | `top5_complete_reader_not_exact` | 143 |
| musique | 3 | `candidate200_complete_top5_incomplete` | 184 |
| musique | 3 | `candidate200_incomplete_not_top5` | 1 |
| musique | 3 | `candidate200_incomplete_top5_partial` | 49 |
| musique | 3 | `top5_complete_answer_exact` | 42 |
| musique | 3 | `top5_complete_reader_not_exact` | 40 |
| musique | 4 | `candidate200_complete_top5_incomplete` | 108 |
| musique | 4 | `candidate200_incomplete_not_top5` | 1 |
| musique | 4 | `candidate200_incomplete_top5_partial` | 54 |
| musique | 4 | `top5_complete_answer_exact` | 1 |
| musique | 4 | `top5_complete_reader_not_exact` | 2 |

## Rank Distribution Within Candidate-Complete Top5 Failures

For `candidate200_complete_top5_incomplete` cases, the table buckets each
query by the worst-ranked missing gold document in ETv3 candidate200.  If
most rows fall in `rank6_10` or `rank11_20`, budget extension alone is a
plausible fix.  If many rows require `rank51_100` or `rank101_200`, the
problem is a deeper composition/selection objective, not just top-k budget.

| Dataset | Gold docs | Queries | max 6-10 | max 11-20 | max 21-50 | max 51-100 | max 101-200 | all missing <=20 | all missing <=50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 2 | 76 | 29 | 23 | 19 | 3 | 2 | 52 | 71 |
| 2wikimultihopqa | 4 | 193 | 18 | 18 | 139 | 17 | 1 | 36 | 175 |
| hotpotqa | 2 | 89 | 60 | 13 | 15 | 1 | 0 | 73 | 88 |
| musique | 2 | 149 | 49 | 42 | 49 | 3 | 6 | 91 | 140 |
| musique | 3 | 184 | 66 | 36 | 48 | 21 | 13 | 102 | 150 |
| musique | 4 | 108 | 14 | 14 | 31 | 21 | 28 | 28 | 59 |

## MuSiQue Long-Chain Reading

MuSiQue 3-doc:

- count: `316`
- top5 all-gold: `0.2595`
- candidate200 all-gold: `0.8418`
- R@5 / R@200: `0.6951` / `0.9473`
- EM/F1: `0.2943` / `0.3946`

MuSiQue 4-doc:

- count: `166`
- top5 all-gold: `0.0181`
- candidate200 all-gold: `0.6687`
- R@5 / R@200: `0.4623` / `0.8825`
- EM/F1: `0.1386` / `0.2382`

Interpretation:

- If `candidate200_complete_top5_incomplete` dominates, ETv3 has a composition/readout problem.
- If candidate200 all-gold is low, the candidate universe itself is a hard ceiling; a top5-only selector cannot fix those cases.
- If `top5_complete_reader_not_exact` is large, the next bottleneck is reader/context rather than retrieval.

This audit is not a DBEC residual audit.  A DBEC residual audit needs
ETv3-pool + stable DBEC full1000 outputs first.
