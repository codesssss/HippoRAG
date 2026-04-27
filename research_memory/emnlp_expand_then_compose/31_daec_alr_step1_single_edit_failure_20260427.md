# DAEC-ALR Step-1 Single-Edit Gate - 2026-04-27

## Purpose

This note records the executed DAEC-ALR Step-1 gate.

The tested question was:

```text
Can Qwen reader consistency turn DAEC from static fixed-pool composition into a safe single-edit repair operator?
```

The experiment followed the pre-flight plan:

- `research_memory/emnlp_expand_then_compose/30_daec_alr_step1_plan.md`

This run did not perform active retrieval. It only replaced at most one document inside the existing PropRAG top100 pool.

## Artifacts

Implementation:

- `scripts/run_daec_alr_step1.py`
- `tests/dpathrag/test_daec_alr_step1.py`

Primary outputs:

- `reports/daec_alr/step1_single_edit_gate.md`
- `reports/daec_alr/step1_single_edit_gate.json`
- `reports/daec_alr/step1_single_edit_rows.jsonl`
- `reports/daec_alr/step1_single_edit_traces.jsonl`
- `reports/daec_alr/step1_single_edit_reader_cache.jsonl`

Validation:

```text
python -m py_compile scripts/run_daec_alr_step1.py tests/dpathrag/test_daec_alr_step1.py
.venv-hipporag/bin/python -m pytest tests/dpathrag/test_daec_alr_step1.py
5 passed, 2 warnings
```

Earlier combined validation also passed:

```text
.venv-hipporag/bin/python -m pytest tests/dpathrag/test_daec_alr_step1.py tests/dpathrag/test_daec_consistency_probe.py
8 passed, 2 warnings
```

## Implementation Boundary

The exported DAEC report contains selected positions, requirements, coverage, covered requirement positions, binding candidates, and selection steps. It does not contain enough embedding state or raw match-score arrays to exactly recompute the original DAEC candidate score for every replacement candidate.

Therefore Step-1 used a fixed trace-local demand/binding proxy:

- protect already covered high-confidence requirement positions;
- choose the lowest-utility non-protected selected document as the replacement target;
- prefilter candidates from `PropRAG top100 - DAEC top5`;
- reject duplicate-title candidates;
- require lexical/entity overlap with either selected context or question anchors;
- score candidate utility from requirement anchors, binding-candidate title hits, subquery-token overlap, expected-answer-type cues, and rank prior;
- log all component scores in query-level traces.

This means the result is a valid test of the frozen exported-trace Step-1 operator, not a test of a fully reconstructed embedding-based DAEC scorer.

## Invalid 32-Token Run

An initial full run used:

```text
--max_new_tokens 32
```

That run was invalid because Day-0 used `max_new_tokens=64`. The 32-token setting truncated many Qwen answers and collapsed the no-edit baseline:

```text
invalid no_edit_daec_only F1 = 0.2034
expected Day-0-aligned F1 ~= 0.5882
```

During audit, the cache key was also fixed to include generation limits:

```text
max_new_tokens
max_docs
max_doc_chars
```

This prevents 32-token and 64-token generations from sharing cache entries.

The valid run below used:

```text
--max_new_tokens 64
```

and recovered the Day-0 baseline:

```text
no_edit_daec_only EM/F1 = 0.4900 / 0.5881
Day-0 EM/F1 = 0.4900 / 0.5882
```

## Protocol

Input:

- dataset: `2WikiMultiHopQA`
- rows: first `200`
- DAEC report: `run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json`
- pool JSON: `run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json`
- reader: local `qwen3-8b-train` at `http://localhost:8043/v1`
- prompt: HippoRAG `rag_qa_musique` one-shot QA template
- final answer budget: `max_new_tokens=64`

Frozen operator settings:

- round-0 perturbations: `original, swap01, reverse, rotate_left, drop_last`
- skip rule: edit only if round-0 `majority_fraction < 1.0`
- candidate budget: `K_edit = 5`
- admission perturbations: `original, swap01, reverse, rotate_left`
- admission rule: `admission_inverse_entropy >= base_inverse_entropy + 0.10`
- validation perturbations: `original, drop_last, drop_first, drop_weakest_selected, drop_candidate, swap01_after_drop_last`
- validation rule: `validation_inverse_entropy >= base_inverse_entropy`
- accepted edits per query: at most `1`

Pre-registered variants:

1. `no_edit_daec_only`
2. `binding_gate_only`
3. `consistency_gate_only`
4. `double_gate_no_skip`
5. `double_gate_skip`

Main configuration:

```text
double_gate_skip
```

## Main Results

| Variant | F1 | Delta F1 | CI95 Delta F1 | Support Complete | Accepted Edit Rate | Edited Subset F1 Pre/Post | W->C | C->W | Added Gold / Non-Gold | Non-Gold / Gold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_edit_daec_only` | 0.5881 | 0.0000 | [0.0000, 0.0000] | 0.8550 | 0.000 | 0.0000 / 0.0000 | 0 | 0 | 0 / 0 | n/a |
| `binding_gate_only` | 0.5245 | -0.0636 | [-0.1159, -0.0190] | 0.6450 | 1.000 | 0.5887 / 0.5245 | 9 | 25 | 13 / 187 | 14.38 |
| `consistency_gate_only` | 0.5757 | -0.0125 | [-0.0371, 0.0110] | 0.8150 | 0.230 | 0.2910 / 0.2343 | 3 | 5 | 4 / 42 | 10.50 |
| `double_gate_no_skip` | 0.5757 | -0.0125 | [-0.0371, 0.0110] | 0.8150 | 0.230 | 0.2910 / 0.2343 | 3 | 5 | 4 / 42 | 10.50 |
| `double_gate_skip` | 0.5757 | -0.0125 | [-0.0371, 0.0110] | 0.8150 | 0.230 | 0.2910 / 0.2343 | 3 | 5 | 4 / 42 | 10.50 |

Decision:

```text
RED
```

## Gate Evaluation

Pre-registered 2Wiki main gates for `double_gate_skip`:

| Gate | Required | Observed | Pass |
|---|---:|---:|---|
| F1 gain | `>= +0.004` | `-0.0125` | no |
| paired bootstrap CI lower bound | `> 0` | `-0.0371` | no |
| accepted edit rate | `[10%, 35%]` | `23.0%` | yes |
| edited-subset F1 | post > pre | `0.2343 < 0.2910` | no |
| flip balance | W->C > C->W | `3 < 5` | no |
| support_complete | `>= baseline` | `0.8150 < 0.8550` | no |
| hard-negative import | NG/G `<= 8.0` | `10.50` | no |

The main variant fails six of seven gates. The only passed gate is the edit-rate window.

## Edit-Level Diagnostics

For `double_gate_skip`:

- accepted edits: `46/200`
- support_complete drops among accepted edits: `10`
- support_complete gains among accepted edits: `2`
- support_complete unchanged among accepted edits: `34`
- F1 gains among accepted edits: `9`
- F1 losses among accepted edits: `13`
- F1 unchanged among accepted edits: `24`
- wrong-to-correct flips: `3`
- correct-to-wrong flips: `5`
- added gold supports: `4`
- added non-gold docs: `42`
- removed gold supports: `15`
- removed non-gold docs: `31`

The operator is not just low-gain. It is actively non-destructive only in many no-op cases, and destructive in the subset that matters.

## Mechanism Interpretation

### 1. Reader consistency reduces but does not solve hard-negative import

`binding_gate_only` is the clearest failure:

```text
accepted edit rate = 1.000
F1 delta = -0.0636
support_complete = 0.6450
added_non_gold / added_gold = 14.38
```

The trace-local binding proxy finds plausible replacement candidates, but most are shared lexical hard negatives. This repeats the D-PathRAG pattern: the operator can find locally relevant documents but cannot reliably distinguish gold from distractor.

The consistency gate reduces the damage:

```text
binding_gate_only F1 = 0.5245
double_gate_skip F1 = 0.5757
```

But it does not make edits safe:

```text
double_gate_skip F1 delta = -0.0125
support_complete drop = -0.0400
added_non_gold / added_gold = 10.50
```

### 2. Consistency-gated accepted edits are worse on the edited subset

The edited subset has:

```text
pre-edit F1 = 0.2910
post-edit F1 = 0.2343
```

This directly fails the diagnostic gate. The operator is selecting edits whose consistency improvement does not translate into answer correctness.

### 3. The skip rule did not rescue the operator

`double_gate_no_skip` and `double_gate_skip` are numerically identical on final F1/support.

The skip rule reduced eligibility:

```text
double_gate_no_skip eligible rate = 1.000
double_gate_skip eligible rate = 0.485
```

But the actual accepted edits were already concentrated outside fully stable contexts, so the skip did not change the accepted set. This is useful: over-editing stable queries was not the main failure in this run. The main failure is admission quality among unstable queries.

### 4. Binding gate did not add separability beyond consistency

`consistency_gate_only`, `double_gate_no_skip`, and `double_gate_skip` have identical final metrics.

This means the trace-local binding proxy did not provide meaningful additional filtering once consistency admission was applied. Two interpretations are possible:

- exported-trace binding proxy is too weak to represent full DAEC binding behavior;
- or, in this residual setting, DAEC-style binding signals are already saturated and cannot distinguish the accepted hard negatives.

The current evidence supports the first statement directly and the second only as a hypothesis.

## Relation To Day-0

Day-0 was still valid:

```text
Qwen answer self-consistency under perturbation separates correct from wrong original DAEC contexts.
```

Step-1 shows the missing link:

```text
consistency as a state-quality diagnostic does not imply consistency as an edit-admission utility.
```

The reader can tell that some DAEC contexts are unstable, but replacing one document to increase stability often stabilizes a wrong or still-insufficient context.

## Hotpot Side Gate

The Hotpot cached inputs exist, but the side gate was not launched after the 2Wiki primary gate returned RED.

Reason:

- GREEN requires passing the 2Wiki primary gate.
- The 2Wiki main variant fails F1, CI, edited-subset gain, flip balance, support preservation, and hard-negative ratio.
- Hotpot cannot rescue the decision; it can only add robustness diagnostics.

If this branch is ever reopened, Hotpot should be run only after replacing the admission object or reconstructing the exact DAEC scorer.

## Final Decision

```text
STOP_DAEC_ALR_STEP1_SINGLE_EDIT
```

DAEC-ALR Step-1 should not continue as currently formulated.

Allowed future reopening conditions:

- reconstruct the exact DAEC embedding/demand scorer for arbitrary candidate edits;
- or replace reader consistency with a different admission object that directly predicts answer-correctness improvement;
- or move away from single-edit repair and explicitly test a different intervention.

Disallowed continuation:

- tuning the inverse-entropy margin;
- changing the skip threshold;
- increasing `K_edit`;
- adding more perturbation variants;
- running active retrieval on top of this admission rule.

The useful finding is negative:

```text
Reader self-consistency is a valid diagnostic of context stability, but it is not a reliable admission signal for local evidence repair on strong PropRAG+DAEC substrates.
```
