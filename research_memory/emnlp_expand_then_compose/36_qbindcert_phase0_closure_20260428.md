# Q-BindCert Phase 0 Closure - 2026-04-28

## Final Status

`Q-BindCert` is blocked at Phase 0 and should not enter Phase 1 in the current paper cycle.

This is not a generic method failure. The result is more specific:

```text
Oracle certificate search works; train-free Qwen3-8B typed extraction and query compilation do not clear the reliability floor.
```

## What Was Tested

The Phase 0 gate tested the exact premise behind Q-BindCert:

1. Can Qwen3-8B `/no_think` extract typed propositions from gold support docs?
2. Can Qwen3-8B `/no_think` compile questions into a usable program lattice?
3. Can simple entity normalization avoid obvious alias/duplicate errors?
4. If oracle propositions and programs are given, can certificate search recover the gold answer?
5. Under oracle propositions/programs, can answer-contrastive certificate search reject plausible wrong answers?

Only tests 4 and 5 passed.

## Key Numbers

| Gate | Result | Threshold | Status |
|---|---:|---:|---|
| Proposition predicate bucket proxy | `0.94` | `0.80` | pass-like |
| Proposition argument extraction proxy | `0.62` | `0.80` | fail |
| Program top-3 any-correct proxy | `0.55` | `0.75` | fail |
| Program top-1 correct proxy | `0.55` | `0.60` | fail |
| Program answer-variable correct proxy | `0.70` | `0.75` | fail |
| Oracle certificate complete | `1.00` | `0.70` | pass |
| Oracle certificate answer agrees | `1.00` | `0.65` | pass |
| Oracle gold beats plausible wrong | `1.00` | `0.65` | pass |
| Oracle wrong structurally rejected | `0.96` | `0.40` | pass |

Decision:

```text
STOP_PHASE0_FAIL
```

## Positive Finding

The certificate formulation is viable under oracle conditions.

Given correct propositions, correct query program, and oracle entity normalization, the lexicographic certificate key cleanly recovers complete certificates and separates gold from plausible wrong answers on the 50-query oracle audit.

This means the idea of answer-competitive binding certificates is not immediately falsified. The search object itself has signal.

## Blocking Finding

The train-free upstream substrate is not reliable enough.

Qwen3-8B `/no_think` outputs parseable JSON, so the failure is not a formatting failure. The failure is semantic:

- It often misses bridge arguments or bridge hops even from gold support docs.
- It sometimes extracts the terminal answer fact but misses the upstream binding fact.
- It often compiles comparison questions around the attribute being compared instead of the entity returned as the answer.
- Program lattice samples are correlated; top-3 does not provide enough diversity to rescue top-1 errors.

This directly validates the LCPS caution: typed extraction and query compilation are the bottleneck, not downstream certificate scoring.

## Paper Interpretation

Q-BindCert should be recorded as a high-upside but Phase-0-blocked method, not as another full negative implementation.

Suggested paper language:

```text
We also evaluated a query-compiled certificate-search formulation that is structurally capable of rejecting plausible wrong answers under oracle propositions and oracle programs. However, its train-free instantiation fails the prerequisite semantic compiler gates: Qwen3-8B produces parseable JSON but misses bridge arguments and miscompiles answer variables frequently enough that top-3 program-lattice correctness reaches only 0.55 against a 0.75 gate. This suggests that certificate search is promising only if the semantic compiler is trained, supervised, or otherwise constrained; it is not a reliable zero-shot extension of DAEC.
```

## Consequence For Current Route

Do not spend the 11-18 day Phase 1/2 engineering budget on Q-BindCert now.

The current paper should keep the DAEC floor as the main positive method and use Q-BindCert as a bounded diagnostic:

```text
The next level of structure helps under oracle semantics, but train-free semantic compilation is not robust enough to instantiate it.
```

This is a useful bridge from DAEC to future work: it explains what would be needed to make the method less trick-like and more end-to-end.

## Reopen Conditions

Only reopen Q-BindCert if at least one of these changes is true:

1. A trained or supervised query compiler is allowed.
2. Human-verified or dataset-derived programs are available as supervision.
3. Typed proposition extraction is replaced by a more constrained extractor with argument coverage above `0.80`.
4. Program lattice top-3 any-correct reaches at least `0.75` and top-1 reaches at least `0.60` on a held-out Phase 0 sample.
5. The goal changes from EMNLP main method to a future supervised/e2e project.

Do not reopen by prompt tuning alone. The observed errors are semantic and correlated, not just JSON-format issues.

## Canonical Artifacts

- Plan: `research_memory/emnlp_expand_then_compose/34_qbindcert_phase0_plan.md`
- Result: `research_memory/emnlp_expand_then_compose/35_qbindcert_phase0_results_20260427.md`
- Report: `reports/qbindcert/phase0_sanity.md`
- Script: `scripts/qbindcert_phase0_sanity.py`
- Tests: `tests/dpathrag/test_qbindcert_phase0.py`
- Commit: `a696294 Add Q-BindCert Phase 0 sanity gate`

