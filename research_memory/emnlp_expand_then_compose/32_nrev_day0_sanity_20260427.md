# NREV Day-0 Sanity - 2026-04-27

## Purpose

This note records the Day-0 test for `NREV-GraphRAG`, the null-dominated reversible evidence score:

```text
NREV(a, S) = l_plus(a, S) - max(l_minus(a, S), l0(a), l_alt(a, S))
```

The tested question was:

```text
Can reader likelihood with destructive evidence nulls, closed-book nulls, and type-compatible alternative-answer nulls separate gold answer-evidence pairs from plausible wrong pairs?
```

This was a Day-0 sanity test only. It did not implement the full fixed-pool candidate-set family and did not run active retrieval.

## Artifacts

Implementation:

- `scripts/run_nrev_day0_sanity.py`
- `tests/dpathrag/test_nrev_day0_sanity.py`

Primary outputs:

- `reports/nrev/day0_sanity.md`
- `reports/nrev/day0_sanity.json`
- `reports/nrev/day0_sanity.rows.jsonl`
- `reports/nrev/day0_sanity.components.jsonl`
- `reports/nrev/throughput_benchmark.md`
- `reports/nrev/day0_probe5/day0_sanity.md`

Validation:

```text
.venv-hipporag/bin/python -m py_compile scripts/run_nrev_day0_sanity.py tests/dpathrag/test_nrev_day0_sanity.py
.venv-hipporag/bin/python -m pytest tests/dpathrag/test_nrev_day0_sanity.py -q
5 passed, 2 warnings
```

## Protocol

Input data:

- dataset: `2WikiMultiHopQA`
- split slice: first `100` records from the existing local dev/eval fold
- pool: `run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json`
- plausible wrong source: `reports/caps/caps_day2_proof_separability.json`
- reader / scorer: local `qwen3-8b-train` at `http://localhost:8043/v1`

Pair construction:

- `(a_gold, S_gold)`: gold answer plus gold support docs, padded to `k=5` with top-ranked PropRAG pool docs.
- `(a_wrong, S_wrong)`: first non-gold CAPS Day-2 top-5 answer, with evidence approximated from wrong-answer-containing pool docs plus high-ranked distractors when source evidence was unavailable.

This wrong-evidence construction is a limitation: CAPS Day-2 preserved plausible wrong answers but not a full native wrong evidence set. Therefore this run tests a practical gold-vs-plausible-wrong proxy, not a perfectly replayed CAPS evidence trace.

Scoring:

- `l_plus`: mean length-normalized Qwen prompt log-likelihood over four fidelity-preserving order perturbations.
- `l_minus`: mean length-normalized likelihood over singleton functional-cell deletion and deterministic matched replacement, up to `5 x 2 = 10` destructive contexts.
- `l0`: length-normalized likelihood with empty evidence context.
- `l_alt`: max `l_plus` over type-compatible alternative answers only.

Implementation details fixed during the run:

- High-confidence question-type heuristics now override Qwen type classification for cases such as `place of birth`; this prevents `song/film` words inside a place question from incorrectly forcing `work`.
- Closed-book stratification uses a separate own-knowledge prompt and final-answer parser. The initial evidence-only empty-context prompt made every closed-book answer abstain, which was a prompt artifact.
- Reports now include diagnostics: paired outcomes, dominant null counts, component means, alt-count distribution, and question-type counts.

## Main Results

Decision:

```text
STOP_NREV_DAY0_FAIL
```

Overall metrics:

| Score | AUC | 95% CI | Paired Win Rate |
|---|---:|---:|---:|
| `l_plus` | 0.7085 | [0.6273, 0.7790] | 0.7000 |
| `rev = l_plus - l_minus` | 0.4266 | [0.3555, 0.4960] | 0.4000 |
| `nrev_no_alt = l_plus - max(l_minus, l0)` | 0.4221 | [0.3417, 0.4972] | 0.4100 |
| `nrev_no_l0 = l_plus - max(l_minus, l_alt)` | 0.6146 | [0.5241, 0.6969] | 0.5700 |
| `nrev_full` | 0.6042 | [0.5149, 0.6889] | 0.5600 |

Closed-book stratification:

| Subset | N | Full NREV AUC | 95% CI | Paired Win Rate |
|---|---:|---:|---:|---:|
| closed-book correct | 2 | 0.5000 | [0.0000, 1.0000] | 0.5000 |
| closed-book wrong | 98 | 0.6057 | [0.5068, 0.6968] | 0.5612 |

Gate evaluation:

- Strong proceed gate failed: full NREV AUC `0.6042 < 0.75`.
- CI gate failed: lower bound `0.5149 < 0.70`.
- Evidence-required scoped gate failed: closed-book-wrong full NREV AUC `0.6057 < 0.75`.
- Marginal redesign gate failed under the registered full-NREV criterion: full NREV AUC `< 0.65`.

Therefore the full fixed-pool NREV prototype should not be launched from this result.

## Diagnostics

Paired outcomes:

| Score | Gold Wins | Wrong Wins |
|---|---:|---:|
| `l_plus` | 70 | 30 |
| `rev` | 40 | 60 |
| `nrev_no_alt` | 41 | 59 |
| `nrev_no_l0` | 57 | 43 |
| `nrev_full` | 56 | 44 |

Dominant null counts:

| Side | `l_minus` | `l0` | `l_alt` |
|---|---:|---:|---:|
| gold | 51 | 31 | 18 |
| wrong | 46 | 6 | 48 |

Component means:

| Side | `l_plus` | `l_minus` | `l0` | `l_alt` |
|---|---:|---:|---:|---:|
| gold | -2.2209 | -2.3087 | -3.7157 | -5.2356 |
| wrong | -4.0172 | -4.1657 | -6.1476 | -2.4365 |

Question-type breakdown for full NREV:

| Type | N | Full NREV AUC | 95% CI | Paired Win Rate |
|---|---:|---:|---:|---:|
| `person` | 22 | 0.7975 | [0.6074, 0.9277] | 0.6818 |
| `place` | 30 | 0.6956 | [0.5233, 0.8511] | 0.7000 |
| `work` | 29 | 0.4851 | [0.3115, 0.6457] | 0.4483 |
| `yes_no` | 13 | 0.3314 | [0.0947, 0.6331] | 0.3846 |
| `entity` | 5 | 0.4000 | [0.0800, 0.8000] | 0.4000 |
| `date` | 1 | 0.0000 | [0.0000, 0.0000] | 0.0000 |

Throughput:

- final cached rerun: `0.4435s/query`;
- first uncached 100-query run: about `172.8s`, or `1.73s/query`;
- 5-query preflight estimate: about `2.10s/query`.

The cost gate passed. The scientific gate failed.

## Interpretation

The strongest positive signal is the simplest one:

```text
l_plus only AUC = 0.7085
```

This says evidence-world answer likelihood has some gold-vs-plausible-wrong separability. But the reversible/null-dominated part does not help:

```text
original REV AUC = 0.4266
full NREV AUC = 0.6042
```

The destructive null is actively harmful in this Day-0 construction. `l_minus` dominates the gold null in `51/100` cases, and original REV flips below random. This means deletion/replacement did not reliably reduce gold more than plausible wrong; it often penalized gold evidence at least as much as wrong evidence.

The alternative-answer null helps wrong-side suppression in some cases, but it is not enough. It dominates wrong pairs in `48/100` cases, yet full NREV remains only `0.6042` AUC. The benefit is concentrated in `person` questions and does not transfer to `work` or `yes_no` questions.

The closed-book null is not the main rescue mechanism here. Closed-book generation is correct in only `2/100` cases, and full NREV on the closed-book-wrong subset is still `0.6057`.

## Decision

Stop NREV as a Day-0-passed residual repair route.

Do not launch:

- full fixed-pool NREV candidate family;
- open-pool NREV rescue;
- threshold tuning around the current NREV nulls.

If this line is ever reopened, it needs a materially different destructive-null construction or a better native wrong-evidence trace. It should not be reopened as a simple hyperparameter sweep.

## Audit Addendum

Audit report:

- `reports/nrev/day0_audit.md`
- `reports/nrev/day0_audit.json`
- fixed same-title replacement first-30 rerun: `reports/nrev/day0_audit_rerun30/day0_sanity.md`

The audit found three important corrections.

First, the closed-book `2/100` result is prompt/parsing sensitive and should not be used as a scientific claim. With a 20-query audit:

| Prompt | Correct@F1>=0.5 | Mean F1 |
|---|---:|---:|
| original own-knowledge prompt | 4/20 | 0.1855 |
| direct short-answer prompt | 7/20 | 0.3150 |

Second, the earlier shorthand "`l_minus > l_plus` in 51/100 gold pairs" was imprecise. The `51/100` number is the count where `l_minus` is the dominant strongest null for gold. The actual raw-component count is:

```text
gold l_minus > l_plus: 39/100
gold l_minus < l_plus: 61/100
mean gold (l_minus - l_plus): -0.0877
```

Third, there was a real implementation defect in matched replacement: same-title duplicates were allowed. This made some destructive replacements non-destructive, e.g. replacing a support doc with another passage of the same title.

The bug was fixed in:

- `scripts/run_nrev_day0_sanity.py`
- `tests/dpathrag/test_nrev_day0_sanity.py`

The fixed first-30 rerun gives:

| Score | AUC | 95% CI | Paired Win Rate |
|---|---:|---:|---:|
| `l_plus` | 0.6656 | [0.5400, 0.7944] | 0.6667 |
| `rev` | 0.5278 | [0.4078, 0.6356] | 0.4667 |
| `nrev_no_alt` | 0.5222 | [0.3867, 0.6356] | 0.5000 |
| `nrev_no_l0` | 0.6600 | [0.4911, 0.8167] | 0.5667 |
| `nrev_full` | 0.6644 | [0.4744, 0.8244] | 0.6000 |

Audit decision:

```text
NREV_AUDIT_MARGINAL_TMINUS_REDESIGN_ONLY
```

Interpretation:

- The original `STOP_NREV_DAY0_FAIL` was too strong without audit.
- The implementation bug explains part of the REV/NREV collapse.
- Fixing the bug does not make NREV pass the audit keep-alive gate (`>= 0.70` on first-30).
- It does move full NREV into the marginal band, so the only defensible continuation is one bounded T-minus redesign. It is still not cleared for full fixed-pool implementation.

## Claim Update

This result contradicts the hypothesis that:

```text
reader-likelihood falsification with broken-evidence, closed-book, and alternative-answer nulls is sufficient to separate gold answer-evidence pairs from plausible wrong pairs on the strong 2Wiki substrate.
```

It strengthens the broader residual-route diagnosis:

```text
local reader-side signals exist, but converting them into a safe global answer/evidence decision remains the hard barrier.
```
