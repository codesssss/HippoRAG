# DAEC-ALR Step-1 Pre-Flight Plan - 2026-04-27

## Purpose

This document freezes the next DAEC-ALR experiment before implementation.

The Step-1 question is deliberately narrow:

```text
Can Qwen reader consistency turn DAEC from static fixed-pool composition into a safe single-edit repair operator?
```

This is a research-memory pre-registration. It is not a paper section, not an active-retrieval proposal, and not a claim that DAEC-ALR is already a method.

## Decision Boundary

Step-1 may proceed only as:

```text
single-edit repair inside the existing PropRAG top100 pool
```

Step-1 must not:

- retrieve outside the cached PropRAG top100 pool;
- run free-form iterative retrieval;
- tune gates after looking at Step-1 outcomes;
- promote reader consistency as correctness;
- continue if edits reproduce the D-PathRAG hard-negative import pattern.

The only permitted positive interpretation is:

```text
reader self-consistency can safely complement DAEC demand/binding scores for bounded in-pool repair
```

The permitted negative interpretation is:

```text
reader self-consistency has Day-0 separability but cannot support a non-destructive edit operator
```

## Fixed Inputs

Primary dataset:

- dataset: `2WikiMultiHopQA`
- split: first `200` rows from the existing PropRAG top100 DAEC export
- DAEC report: `run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json`
- pool JSON: `run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json`
- evidence budget: top-5 selected documents

Side dataset:

- dataset: `HotpotQA`
- split: first `200` rows, only if the cached files are readable
- DAEC report: `run_logs/layer1_proprag_pool_eval_fixed_20260424/hotpotqa_proprag_pool_daec_oracle.json`
- pool JSON: `run_logs/proprag_pool_exports_full1000_20260424/hotpotqa_pool100.json`
- if either file is missing or schema-incompatible, record `HOTPOT_SIDE_GATE_BLOCKED_BY_CACHE` and do not substitute a different dataset.

Reader:

- local OpenAI-compatible endpoint: `http://localhost:8043/v1`
- model: `qwen3-8b-train`
- prompt: HippoRAG `rag_qa_musique` one-shot QA template
- answer normalization: same normalization used by `scripts/run_daec_consistency_probe.py`

Primary baseline:

- co-run `no edit / DAEC only` under the same reader, prompt, decoding, and evaluation code as the edited variants
- Day-0 reconstructed DAEC values are reference checks, not the comparison baseline for stochastic Step-1 runs

## Day-0 Facts Used To Freeze Rules

Primary Qwen matched-prompt probe:

- report: `reports/daec_alr/consistency_probe.md`
- rows: `200`
- variants: `original, swap01, reverse, rotate_left, drop_last`
- original EM/F1: `0.4900 / 0.5882`
- support recall/complete: `0.9413 / 0.8550`
- `majority_fraction` AUC vs original `F1>=0.5`: `0.757184`, 95% CI `[0.687466, 0.820193]`
- `inverse_entropy` AUC vs original `F1>=0.5`: `0.759778`, 95% CI `[0.690214, 0.822619]`
- decision: `PASS_SIGNAL_PROBE`

Majority-fraction distribution from Day-0 rows:

| majority_fraction | queries | F1>=0.5 | F1<0.5 | positive rate |
|---:|---:|---:|---:|---:|
| 0.20 | 7 | 0 | 7 | 0.000 |
| 0.40 | 24 | 4 | 20 | 0.167 |
| 0.60 | 30 | 14 | 16 | 0.467 |
| 0.80 | 37 | 20 | 17 | 0.541 |
| 1.00 | 102 | 81 | 21 | 0.794 |

Frozen skip decision:

```text
skip if round0_majority_fraction == 1.0
eligible_for_edit if round0_majority_fraction < 1.0
```

Rationale:

- `majority_fraction == 1.0` skips `102/200 = 51.0%` of queries and leaves `98/200` eligible.
- A looser skip rule such as `majority_fraction >= 0.8` would skip `139/200 = 69.5%` and would remove too many potentially repairable wrong or marginal cases.
- The skip rule is fixed before Step-1 and must not be tuned after seeing edit results.

Controls and limitations:

- Qwen simple-prompt AUC is not directly comparable because the prompt changed the base error distribution.
- Flan-T5 control had weak consistency separability (`majority_fraction` AUC `0.5666`), so this signal is reader-dependent.
- Consistency is not correctness; consistent wrong answers exist.
- Day-0 validates a reader-side signal only. It does not prove that edits improve F1.

## Main Hypothesis

Main hypothesis:

```text
Among DAEC contexts where Qwen answers are not fully stable, a single replacement from the existing PropRAG top100 pool can improve final answer F1 if it passes both DAEC binding improvement and reader-consistency improvement gates.
```

Anti-hypotheses to rule out:

- The gain, if any, comes only from DAEC binding scores and consistency adds no useful admission signal.
- The gain, if any, comes only from reader consistency and DAEC binding is unnecessary.
- The operator improves consistency by making wrong answers more stable.
- The operator imports shared lexical hard negatives, reproducing D-PathRAG's failure mode.

## Non-Goals

Step-1 is not designed to:

- recover the full oracle@100 headroom;
- build a new graph-RAG substrate;
- train a verifier;
- tune a production active-retrieval policy;
- compare reader families;
- claim reader-consistency portability beyond Qwen.

If Step-1 works, it is a bounded DAEC extension or ablation. If it fails, it is a controlled diagnostic of why reader-side consistency cannot safely drive in-pool repair.

## Step-1 Operator

### Round-0 Consistency

For each query, compute DAEC top-5 consistency before any edit.

Fixed round-0 perturbation family:

```text
original
swap01
reverse
rotate_left
drop_last
```

Fixed features:

- primary consistency score: `inverse_entropy`
- secondary consistency score: `majority_fraction`
- diagnostic scores: `inverse_distinct`, `mean_pairwise_f1`, `distinct_answer_count`, `normalized_entropy`

The primary score is `inverse_entropy` because it had the strongest Day-0 AUC against original `F1>=0.5` among non-tied metrics.

### Eligibility / Skip Rule

For each query:

```text
if round0_majority_fraction == 1.0:
    skip edit and keep DAEC top5
else:
    enter candidate prefilter
```

Skipped queries still count in final metrics. They are not removed from the evaluation denominator.

### Candidate Prefilter

For each eligible query:

1. Candidate pool is `PropRAG top100 - current DAEC top5`.
2. Exclude candidates whose title duplicates any current top-5 title after normalized title matching.
3. Require at least one lexical/entity overlap with either:
   - current top-5 evidence titles/text, or
   - question anchors.
4. Score candidates by DAEC demand/binding improvement relative to the current top-5 context.
5. Keep `K_edit = 5`.

Do not evaluate all remaining top100 candidates with the reader. The reader-consistency stage only sees the five prefiltered candidates.

### Edit Construction

Each candidate edit replaces exactly one current DAEC-selected document.

Replacement target:

```text
the selected top-5 document with the lowest demand utility, subject to not removing an already satisfied high-confidence demand
```

If the implementation cannot identify a non-destructive replacement target, the query receives no edit.

At most one edit may be accepted per query.

### Binding Gate

A candidate passes the binding gate only if all conditions hold:

1. It increases DAEC demand/binding score relative to the current top-5.
2. It improves the weakest currently selected demand.
3. It does not reduce already satisfied high-confidence demands.
4. It does not duplicate an existing selected title.

If the internal DAEC score is multi-component, the implementation must log the component-level before/after scores. The decision rule must use the same frozen score composition for every query.

### Admission Consistency

Admission uses an order-based perturbation family on each candidate-edited context.

Fixed admission variants:

```text
original
swap01
reverse
rotate_left
```

Fixed admission budget:

```text
K_admit = 4
```

Admission consistency rule:

```text
admit candidate for validation only if:
    candidate passes binding gate
    and admission_inverse_entropy >= base_inverse_entropy + 0.10
```

The `+0.10` margin is fixed before execution. It is intended to avoid accepting noise-level improvements from coarse K-sample consistency.

If multiple candidates pass admission, validate only the one with the largest lexicographic tuple:

```text
(admission_delta_inverse_entropy, binding_improvement, admission_delta_majority_fraction)
```

### Independent Validation

Validation is run only for the best admitted candidate.

Validation must use a different perturbation family from admission to reduce selection bias.

Fixed validation variants:

```text
original
drop_last
drop_first
drop_weakest_selected
drop_candidate
swap01_after_drop_last
```

Fixed validation budget:

```text
K_val = 6
```

Final acceptance rule:

```text
accept edit only if:
    validation_inverse_entropy >= base_inverse_entropy
    and validation_majority_answer is not empty
```

Validation is a preservation gate, not a second opportunity to tune thresholds. It checks that the admitted edit does not rely only on the order-based perturbation family.

### Final Prediction

For accepted edits, the final context is the edited top-5. For skipped or rejected queries, the final context is the original DAEC top-5.

Final answer generation:

- use the original-order final top-5 context unless the existing DAEC reader protocol specifies a different canonical order;
- do not majority-vote final answers unless a separate ablation explicitly records that behavior;
- final F1/EM/support metrics are computed over all 200 queries.

## Pre-Registered Variants

Run exactly these variants for the primary 2Wiki eval200 block:

| Variant | Description | Purpose |
|---|---|---|
| `no_edit_daec_only` | Keep original DAEC top-5. | Baseline under the same run/eval code. |
| `binding_gate_only` | Candidate prefilter + binding gate; no reader-consistency gate. | Tests whether DAEC binding alone explains any gain. |
| `consistency_gate_only` | Reader-consistency gate over prefiltered candidates; no binding gate except duplicate-title safety. | Tests whether consistency alone is sufficient. |
| `double_gate_no_skip` | Binding gate + consistency gate for all queries. | Tests whether the skip rule is necessary to avoid over-editing. |
| `double_gate_skip` | Binding gate + consistency gate with `majority_fraction == 1.0` skip. | Main configuration. |

Main configuration:

```text
double_gate_skip
```

Interpretation constraints:

- If `binding_gate_only` is close to `double_gate_skip`, reader consistency is not contributing enough.
- If `consistency_gate_only` is close to `double_gate_skip`, DAEC binding is not contributing enough and the result conflicts with the historical binding-load-bearing finding.
- If `double_gate_no_skip` underperforms `double_gate_skip`, over-editing stable contexts is a real failure mode.
- If `double_gate_skip` is not best or statistically tied with simpler variants, do not overclaim Step-1.

## Metrics

Primary outcome metrics:

- reader F1
- reader EM
- support recall
- support complete
- paired per-query F1 delta against `no_edit_daec_only`
- paired bootstrap 95% CI over F1 delta

Edit behavior metrics:

- eligible query rate
- candidate trigger rate
- admitted-to-validation rate
- accepted edit rate over all queries
- accepted edit rate over eligible queries
- edited-subset pre-edit F1
- edited-subset post-edit F1
- wrong-to-correct flips
- correct-to-wrong flips
- no-change rate

Hard-negative import diagnostics:

- added gold support count
- added non-gold count
- removed gold support count
- removed non-gold count
- `added_non_gold / added_gold`
- title-duplicate rejection count
- support_complete before/after on edited subset

Consistency diagnostics:

- base `inverse_entropy`
- admission `inverse_entropy`
- validation `inverse_entropy`
- base `majority_fraction`
- admission `majority_fraction`
- validation `majority_fraction`
- majority answer before/after

All metrics must be logged per variant. Query-level traces must be saved so that failed edits can be audited.

## Gates

### Primary Gate: 2Wiki Eval200

The main configuration `double_gate_skip` passes only if all conditions hold:

```text
F1 >= no_edit_daec_only_F1 + 0.004 absolute
paired bootstrap 95% CI lower bound for F1 delta > 0
accepted edit rate over all queries is in [10%, 35%]
edited-subset post-edit F1 > edited-subset pre-edit F1
wrong_to_correct_flips > correct_to_wrong_flips
support_complete >= no_edit_daec_only_support_complete
added_non_gold / added_gold <= 8.0
```

If `added_gold == 0`, the hard-negative diagnostic fails regardless of the ratio convention.

### Side Gate: Hotpot Eval200

Run only if the cached Hotpot DAEC report and pool JSON are available and schema-compatible.

The main configuration passes the side gate if:

```text
F1 >= no_edit_daec_only_F1 - 0.003 absolute
accepted edit rate over all queries <= 25%
edited-subset post-edit F1 does not collapse below edited-subset pre-edit F1 by more than 0.003 absolute
support_complete does not drop by more than 0.003 absolute
```

Hotpot is a non-collapse stress test, not a required positive-gain dataset for Step-1.

### Diagnostic Gate

The edit mechanism is considered failed if either condition holds:

```text
added_gold == 0
added_non_gold / added_gold > 8.0
```

The threshold `8.0` is fixed as a conservative improvement target relative to the prior D-PathRAG hard-negative import pattern.

## Decision Rules

### GREEN

Declare `GREEN` only if:

- `double_gate_skip` passes all 2Wiki primary gates;
- Hotpot side gate passes or is explicitly blocked by missing/incompatible cache rather than negative performance;
- `double_gate_skip` is the best or statistically tied-best non-baseline variant;
- hard-negative import diagnostic passes;
- query-level audit does not reveal metric or schema artifacts.

GREEN interpretation:

```text
DAEC-ALR Step-1 is a bounded, non-destructive DAEC extension worth keeping for later larger-scale evaluation.
```

### YELLOW

Declare `YELLOW` if:

- F1 delta is positive but CI lower bound is not above zero;
- edited-subset F1 improves but overall F1 gain is below `+0.004`;
- one non-critical gate fails without destructive behavior;
- a simpler ablation matches the main config, making the mechanism ambiguous.

YELLOW interpretation:

```text
reader consistency has partial utility, but the operator is not strong enough to promote without redesign or larger-scale confirmation.
```

### RED

Declare `RED` if any condition holds:

- main F1 is below `no_edit_daec_only`;
- support_complete drops below baseline;
- accepted edit rate is outside `[10%, 35%]`;
- edited-subset post-edit F1 is not higher than pre-edit F1;
- `correct_to_wrong_flips >= wrong_to_correct_flips`;
- `added_gold == 0`;
- `added_non_gold / added_gold > 8.0`;
- query audit reveals implementation or metric invalidity.

RED interpretation:

```text
Day-0 consistency separability does not transfer into a safe single-edit repair operator.
```

## Cost Budget

Expected primary 2Wiki main-config cost:

- round-0 consistency: `200 * 5 = 1000` Qwen calls
- eligible queries from Day-0 rule: about `98/200`
- candidate admission: `98 * 5 * 4 = 1960` Qwen calls
- validation: expected `20%` to `35%` accepted over all queries, about `240` to `420` Qwen calls
- final answer generation if not reused: up to `200` Qwen calls
- expected main-config total: about `3400` to `3800` Qwen calls

Expected 2Wiki all-ablation budget:

```text
<= 20k Qwen calls with cache reuse
<= 30k Qwen calls without perfect reuse
```

Expected Hotpot side budget:

```text
<= 10k-20k Qwen calls, depending on cache reuse and accepted edit rate
```

Required cache key:

```text
(dataset, query_id, selected_doc_ids, perturbation_family, perturbation_variant, prompt_id, reader_model)
```

Do not rerun reader calls across ablations when the exact context and perturbation are already cached.

## Failure Modes To Watch

Selection bias:

- admission and validation use different perturbation families;
- validation is not allowed to tune thresholds.

Over-editing stable queries:

- `majority_fraction == 1.0` queries are skipped in the main config;
- `double_gate_no_skip` measures whether this skip is load-bearing.

K-sample noise:

- admission requires `+0.10` inverse-entropy improvement;
- validation requires preservation under subset/drop perturbations.

Hard-negative import:

- added non-gold/gold ratio is a gate, not an afterthought;
- support_complete must not drop.

Reader dependence:

- Qwen is the only reader used for Step-1;
- do not generalize to Flan-T5 or other readers from this experiment.

Consistency-stabilized wrong answers:

- wrong-to-correct and correct-to-wrong flips are mandatory;
- edited-subset F1 must improve.

## Implementation Notes

Suggested output location:

```text
reports/daec_alr/step1_single_edit_gate.md
reports/daec_alr/step1_single_edit_gate.json
reports/daec_alr/step1_single_edit_traces.jsonl
```

Suggested implementation constraints:

- reuse normalization and scoring helpers from `scripts/run_daec_consistency_probe.py` where possible;
- add tests for candidate prefilter, skip rule, consistency scoring, gate classification, and metric aggregation;
- log enough query-level state to manually inspect every accepted edit;
- fail loudly on DAEC/pool question mismatch;
- keep the no-edit baseline in the same script invocation or same cache/eval protocol as edited variants.

Minimum test coverage before running real Qwen calls:

- `majority_fraction == 1.0` skip is applied only in the main skip variant;
- duplicate-title candidates are rejected;
- `K_edit = 5` cap is enforced;
- binding-only, consistency-only, double-gate, and double-gate-skip variants produce distinct trace labels;
- RED/YELLOW/GREEN decision logic matches the frozen rules above.

## Do-Not-Change Rules

After implementation begins, do not change:

- primary dataset and row count;
- Qwen endpoint/model/prompt for primary comparison;
- `majority_fraction == 1.0` skip threshold;
- `K_edit = 5`;
- `K_admit = 4`;
- `K_val = 6`;
- admission margin `inverse_entropy >= base + 0.10`;
- validation preservation rule;
- five pre-registered variants;
- primary 2Wiki gates;
- GREEN/YELLOW/RED decision rules.

If any rule becomes technically impossible because of schema or cache issues, record the blocker and stop rather than substituting a new rule after inspecting outcomes.
