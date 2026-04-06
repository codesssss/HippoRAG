# MuSiQue Coverage + Alias Audit

Last updated: 2026-04-06

## Scope

This note records the offline follow-up after the MuSiQue gate-boundary audit on
the `beneficial_withheld` bucket.

Goal:
- test whether a subset of `structure_score = 0` cases are caused by small
  predicate coverage gaps in the shared factual structure graph
- stop at case-level verification before any larger smoke rerun
- separate remaining failures into:
  - predicate coverage
  - title/alias closure
  - reachability / relation-composition

This note does **not** change the paper-facing mainline. It is an audit memo.

## Fixed Context

Frozen simple selector candidate:
- `reserve3 + bridge2 + general_factual + saturation guard`

Do-not-touch constraints for this audit:
- no retrieval backbone change
- no selector score change
- no gate threshold change
- no `_apply_structure_rerank()` patch
- no reader ordering probe extension

## Part A: Predicate Coverage Audit

Implemented:
- `scripts/analyze_gate_decision_boundary.py`
- `scripts/audit_predicate_coverage_cases.py`
- `general_factual_v2` as a default-off eval-only predicate-family extension

Outputs:
- `outputs_step0_general_musique/eval_reports/musique_gate_decision_boundary_20260406.analysis.{json,md}`
- `outputs_step0_general_musique/eval_reports/musique_coverage_gap_cases_20260406.json`
- `outputs_step0_general_musique/eval_reports/musique_predicate_coverage_audit_20260406.{json,md}`
- `outputs_step0_general_musique/eval_reports/musique_coverage_patch_casecheck_20260406.{json,md}`

Main findings:
- gate-boundary split on MuSiQue-100:
  - `beneficial_withheld = 8`
  - `harmful_blocked = 11`
  - `beneficial_allowed = 1`
  - `harmful_allowed = 0`
- among `beneficial_withheld`, exactly 5 A-class cases had:
  - `best_offrank_structure_score = 0`
- raw triples existed in all 5 cases
- 4/5 cases exposed repeated, patchable rejected predicate families
- dominant repeated family:
  - `factual_part_of_or_contains`

Minimal `general_factual_v2` patch:
- add containment/reference patterns:
  - `is part of`
  - `part of`
  - `is located in`
  - `located in`
  - `neighborhood of`
  - `district of`
  - `region of`
  - `capital of`
  - `is the entry for`
- add alias/name patterns:
  - `is also known as`
  - `is officially called`
  - `was referred to as`
  - `is called`

5-case case-check result:
- `structure 0 -> nonzero = 2/5`
- `gate unblocked = 2/5`
- `top-5 gold improved = 1/5`
- `go_decision = False`

Interpretation:
- the coverage-gap hypothesis is **partially validated**
- but the patch fails the pre-registered case-level threshold
- therefore this branch is a **no-go for smoke100**

Improved cases:
- `Q42` (`Cape Verde`)
- `Q74` (`Nanjing`)

Unresolved after `general_factual_v2`:
- `Q65` (`Seria, Belait`)
- `Q69` (`Tucson, Arizona`)
- `Q85` (`Jessie Woodrow Wilson Sayre`)

## Part B: Alias / Canonicalization Audit

Implemented:
- `scripts/audit_alias_closure_cases.py`

Outputs:
- `outputs_step0_general_musique/eval_reports/musique_alias_closure_audit_20260406.{json,md}`

The alias audit was intentionally limited to the three unresolved A-class cases.

Diagnosis:

### Q65

Primary diagnosis:
- `title_alias_missing`

Observed pattern:
- best off-rank title: `Seria, Belait`
- structure entities contain bare `Seria`
- current `location_alias` probe does **not** recover `Seria, Belait -> Seria`

Conclusion:
- this is the cleanest case for a tiny default-off title-to-bare alias closure probe

### Q69

Primary diagnosis:
- `alias_present_but_additional_reachability_needed`

Observed pattern:
- current `location_alias` already derives:
  - `Tucson, Arizona -> Tucson`
- the case still remains at `structure_score = 0`

Conclusion:
- alias is not the main remaining bottleneck here
- the next missing factor is more likely reachability / hop exposure / doc coverage

### Q85

Primary diagnosis:
- `alias_plus_relation_family`

Observed pattern:
- relevant forms cluster under descriptor stripping:
  - `US President Woodrow Wilson`
  - `President Woodrow Wilson`
- this is not a title-to-bare alias problem

Conclusion:
- this case requires descriptor-level canonicalization plus relation-family support
- alias-only fixes are insufficient

## Net Takeaway

The right conclusion is **not**:
- "general_factual_v2 almost works, just add more relations"
- "the remaining failures are all alias problems"

The correct conclusion is:
- coverage patching fixed part of the A-class bucket, but not enough to justify a larger run
- the remaining three failures are already split across different layers:
  - `Q65`: pure title alias closure
  - `Q69`: alias available, but reachability still missing
  - `Q85`: descriptor canonicalization plus relation composition

Therefore:
- do **not** run smoke100 on this branch
- do **not** expand `general_factual_v2` broadly
- if continuing, only run tiny case-family-specific probes

## Code / Test Inventory

Relevant code:
- `src/hipporag/utils/causal_utils.py`
- `src/hipporag/utils/config_utils.py`
- `scripts/eval_causal_qwen3.py`
- `scripts/analyze_gate_decision_boundary.py`
- `scripts/audit_predicate_coverage_cases.py`
- `scripts/audit_alias_closure_cases.py`

Relevant tests:
- `tests/test_analyze_gate_decision_boundary.py`
- `tests/test_audit_predicate_coverage_cases.py`
- `tests/test_predicate_classifier_general_factual_v2.py`
- `tests/test_audit_alias_closure_cases.py`
