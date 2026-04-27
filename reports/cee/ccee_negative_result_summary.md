# C-CEE Negative Result Summary

Date: 2026-04-27

## Headline

**C-CEE does not pass the Day-1 decision-level gate.** The frozen Flan-T5 reader has strong semantic likelihood capacity, and oracle-answer counterfactual deltas show a weak-to-moderate signal, but non-oracle answer-cluster counterfactual admission fails to select useful edits.

This stops the C-CEE line. Do not proceed to Phase 2 (`src/cee/` implementation), held-out eval800, MuSiQue/HotpotQA expansion, or further CEE-pairwise/DAEC-feature integration unless a new answer-hypothesis mechanism is introduced.

## Raw Result Table

| Stage | Metric | Result | Gate |
|---|---:|---:|---|
| Asset audit | PropRAG rows | 1000 | pass |
| Asset audit | Dense rows | 1000 | pass |
| Asset audit | CUDA devices | 8 | pass |
| Asset audit | Qwen endpoint | unavailable | fallback to reader-only |
| Likelihood sanity | identity delta | exact 0 by construction | pass |
| Likelihood sanity | batch/single max diff | 0.0 | pass |
| Likelihood sanity | cache repeat max diff | 0.0 | pass |
| Reader semantic capacity | sample size | 163 rank-complete dev queries | sufficient |
| Reader semantic capacity | gold vs random answer AUC | 0.988332 | pass |
| Reader semantic capacity | 95% CI | [0.978735, 0.995408] | pass |
| Reader semantic capacity | paired win rate | 1.0 | pass |
| Oracle-answer edit diagnostic | beneficial vs lexical-HN AUC | 0.713375 | weak pass on point estimate |
| Oracle-answer edit diagnostic | 95% CI | [0.6355, 0.785188] | CI lower fails 0.65 |
| Oracle-answer edit diagnostic | beneficial vs random AUC | 0.76725 | diagnostic positive |
| Non-oracle decision diagnostic | beneficial vs lexical-HN AUC | 0.536875 | fail |
| Non-oracle decision diagnostic | 95% CI | [0.449813, 0.62225] | fail |
| Non-oracle decision diagnostic | beneficial vs random AUC | 0.490875 | fail |
| Non-oracle decision diagnostic | stop rate | 0.965 | overly conservative |
| Non-oracle decision diagnostic | top1 beneficial rate on rank-incomplete | 0.0 | fail |
| Non-oracle decision diagnostic | rank-complete false edit rate | 0.03681 | pass |
| Non-oracle decision diagnostic | added gold / non-gold | 0 / 7 | fail |
| Non-oracle decision diagnostic | non-gold/gold | 7.0 | fail |
| Non-oracle decision diagnostic | support-complete gain | -0.01 | fail |

Final decision:

```text
STOP_DECISION_SEPARABILITY_FAIL
```

## Key Findings

1. **Reader likelihood is semantically meaningful under known correct answers.**

On 163 rank-complete dev queries, the reader strongly separates gold answers from random distractor answers:

```text
AUC = 0.988332
mean loglik gap = +7.643073
paired win rate = 1.0
```

This rules out the weakest failure explanation: the frozen reader's token likelihood is not globally meaningless.

2. **Oracle-answer counterfactual edit deltas contain some signal, but not enough for a robust gate.**

Using the gold answer as the target:

```text
beneficial vs lexical-HN AUC = 0.713375
95% CI = [0.6355, 0.785188]
mean delta beneficial = +2.954419
mean delta lexical-HN = +0.133903
```

This suggests that if the answer hypothesis were known, reader likelihood could partially distinguish useful edits. However, the CI lower bound falls below the planned 0.65 gate, so even the oracle-answer diagnostic is not strong enough to justify Phase 2 alone.

3. **The actual non-oracle decision problem fails.**

With generated answer clusters and conservative STOP prior:

```text
beneficial vs lexical-HN AUC = 0.536875
beneficial vs random AUC = 0.490875
top1 beneficial rate on rank-incomplete = 0.0
support-complete gain = -1.0pp
```

This is the decisive failure. C-CEE cannot convert reader likelihood into useful test-time edit admission without access to the correct answer target or a better answer-hypothesis mechanism.

4. **The conservative STOP prior controls over-editing but does not recover evidence.**

The method edited only 7/200 dev queries:

```text
stop rate = 0.965
rank-complete false edit rate = 0.03681
added_gold = 0
added_non_gold = 7
```

This mirrors CEE-pairwise: a calibrated conservative admission mechanism can stop harmful edits, but it does not select missing-gold edits.

5. **Qwen hypothesis was unavailable in this run.**

The Qwen endpoint returned connection refused:

```text
http://localhost:8039/v1 -> unavailable
```

The Day-1 result is therefore a reader-only non-oracle hypothesis result. This does not invalidate the stop decision for the current route, because the Phase-1 plan explicitly allowed fallback. It does leave one narrow interpretation: C-CEE's remaining possible rescue would require a substantially better answer-hypothesis generator, not a better edit scorer.

## Relation to Earlier D-PathRAG / CEE Results

The C-CEE failure completes the progression:

| Method line | Positive signal | Failure mode | Status |
|---|---|---|---|
| D-PathRAG v1 free-form selector | support-complete +3.3pp on eval1000 | imports 1205 non-gold docs; F1 -1.9pp directionally | stopped as main method |
| Shallow CEE | Edit@1 oracle ceiling is large | threshold scorer cannot recover support gain with acceptable precision | diagnostic only |
| CEE-pairwise v0 | candidate-level calibration induces STOP | beneficial edit recall only 0.0221/0.0588; no support gain | stopped |
| C-CEE Day 1 | reader semantic likelihood is strong for known answers | non-oracle counterfactual decision fails; no added gold | stopped |

The common conclusion is not that the edit action space is wrong. The oracle remains strong:

```text
rank top-5 support_complete = 0.772
oracle Edit@1 top20 support_complete = 0.895
```

The failure is learned or inferred admission:

```text
The system cannot reliably identify which rare edit should be admitted among many lexical hard negatives without stronger supervision or a better answer target.
```

## Paper-Ready Interpretation

The negative result supports the larger DAEC diagnostic story:

> Evidence selection, hard-negative admission, and reader consumption are distinct bottlenecks. Free-form learned selectors can improve support coverage but import reader-hostile negatives. Conservative edit formulations expose large oracle headroom, but neither calibrated pairwise admission nor frozen-reader counterfactual likelihood is sufficient to realize that headroom under non-oracle answer uncertainty.

Recommended paper use:

- Main paper: keep D-PathRAG/CEE/C-CEE as a compact diagnostic subsection or appendix.
- Claim only mechanism-positive D-PathRAG evidence gains, not answer-level improvement.
- Use C-CEE as a negative control showing that reader likelihood alone cannot solve hard-negative admission.
- Do not promote C-CEE as a method contribution.

## Artifacts

Implementation:

```text
scripts/cee_day1_diagnostic.py
tests/dpathrag/test_ccee_day1.py
```

Reports:

```text
reports/cee/asset_audit.md
reports/cee/likelihood_sanity.md
reports/cee/reader_semantic_capacity.md
reports/cee/decision_separability_dev.md
reports/cee/day1_summary.md
```

Related prior reports:

```text
reports/dpathrag/proprag_kfold_eval1000_results.md
reports/dpathrag/cee_pairwise_v0_implementation_summary.md
reports/dpathrag/hard_negative_characterization_v2_proprag_kfold1000.md
reports/dpathrag/context_ablation_proprag_eval200_results.md
```

Verification:

```text
.venv-hipporag/bin/python -m pytest tests/dpathrag
50 passed, 2 warnings
```

## Final Decision

```text
D-PathRAG v1: keep as mechanism-positive diagnostic.
Shallow CEE: keep as oracle-headroom diagnostic.
CEE-pairwise: stopped.
C-CEE: stopped after Day-1 decision gate.
DAEC / Layer-1 fixed-pool composition: remains the paper floor.
```
