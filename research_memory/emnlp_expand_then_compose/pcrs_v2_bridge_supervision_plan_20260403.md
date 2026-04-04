# PCRS-RAG V2 Bridge Supervision Plan

Date: 2026-04-03

## Purpose

This plan defines the next execution wave after the failed `atomic_hybrid_smoke10` run.

The working conclusion is:

> the current hybrid atomic scorer is not blocked by model capacity first;
> it is blocked by missing `bridge_support` supervision.

So the next wave should **not** expand scale and should **not** retune parser, hop budget, or set objective.

It should focus on teaching the atomic layer what `bridge_support` actually is.

## Scope

This wave changes only the atomic supervision path.

In scope:

1. bridge label definition
2. bridge probe-set construction
3. bridge-like hard negative mining
4. bridge detector training path
5. bridge-aware cache rebuild on a small slice
6. atomic and shortlist gates
7. smoke10 recheck only after upstream gates pass

Out of scope:

1. parser / compiler redesign
2. relation-hop cap tuning
3. beam / Pareto / utopia changes
4. reserve policy changes
5. reader prompt changes
6. 40-query or 100-query expansion

## Boundary Decision

For now:

1. `clean V2` stays as the stable baseline branch.
2. `hybrid atomic` stays as an experiment branch.
3. The immediate goal is **not** end-to-end QA gain.
4. The immediate goal is:

   `bridge docs that are currently invisible should become visible to atomic scoring and shortlist traces`

## Main Failure Signals

Current evidence from `pcrs_v2_atomic_hybrid_smoke10_20260403.md`:

1. train `bridge_support` labels: `1`
2. eval `bridge_support` labels: `0`
3. positive atomic items with `bridge_support_prob > 0.05`: `2 / 660`
4. hybrid changed almost all selected sets, but selector `Recall@5` dropped from `0.4917` to `0.4083`
5. all `conditional / cap2 / cap3` smoke10 runs collapsed to the same poor QA result

This means the next task is supervision repair, not architecture search.

## Execution Order

### Phase 0: Freeze Everything Else

Goal:

Keep the problem isolated to supervision.

Tasks:

1. do not modify parser/compiler logic
2. do not modify relation-hop cap logic
3. do not modify selector search logic
4. do not modify set-level objective
5. do not run larger end-to-end validation until bridge gates pass

Deliverable:

1. no code change required
2. this document becomes the active execution boundary

### Phase 1: Write the Bridge Label Spec

Goal:

Turn `bridge_support` from a vague idea into a hard supervision rule.

Deliverable:

Create a short spec note that defines:

1. what counts as `bridge_support`
2. what does **not** count as `bridge_support`
3. what counts as `bridge_like_hard_negative`
4. how `bridge_support` differs from `full_support`
5. how `bridge_support` differs from `NEI`

Required positive criteria:

1. the doc provides an intermediate entity or variable needed by an adjacent hop
2. the doc supplies one half of a chain closure that another hop can consume
3. title and body together close a variable transition even if the doc does not directly answer the unit
4. the variable chain is continuous, not just topically related

Required exclusion criteria:

1. shared topic only
2. shared question entity only
3. title similarity only
4. predicate word overlap without variable continuity
5. entity co-occurrence without hop usefulness

Target file:

1. `research_memory/emnlp_expand_then_compose/pcrs_v2_bridge_label_spec_20260403.md`

### Phase 2: Build a Bridge Probe Set

Goal:

Construct a small, high-value supervision set before any new model training.

Target size:

1. `100` to `200` `(question, need_unit, doc)` examples

Required composition:

1. `bridge_positive`
2. `bridge_like_hard_negative`
3. `clean_nei`
4. a small number of `full_support` controls

Recommended mix:

1. `40` to `60` bridge positives
2. `40` to `60` bridge-like hard negatives
3. `20` to `40` clean NEI
4. `10` to `20` full-support controls

Priority sources:

1. smoke10 regressed queries
2. query 6 style real 3-hop examples
3. docs with title anchor + body relation split
4. docs with gold usefulness but current `support_prob` near zero
5. shortlist/source traces where the needed middle hop never entered the beam

Output files:

1. `research_memory/emnlp_expand_then_compose/data/musique_bridge_probe_20260403.jsonl`
2. `research_memory/emnlp_expand_then_compose/data/musique_bridge_probe_20260403.md`

### Phase 3: Add Mining Support

Goal:

Make bridge sample collection reproducible instead of purely manual.

Implementation tasks:

1. add a small mining script or extension that exports candidate `(question, unit, doc)` rows from an existing need-unit cache
2. rank rows by likely bridge usefulness
3. expose current scores:
   - alignment
   - support
   - contradiction
   - coverage
4. expose trace usefulness hints:
   - whether the doc is in baseline top-k
   - whether the doc is in selector shortlist/source preview
   - whether the query regressed after hybrid

Preferred target:

1. extend `scripts/train_need_unit_scorer.py` or add a new helper script instead of rewriting the training pipeline

Suggested new script:

1. `scripts/mine_bridge_probe_examples.py`

Expected output schema:

1. question
2. query_index
3. unit_id
4. unit_type
5. doc_id
6. doc_title
7. doc_text_preview
8. current_atomic_scores
9. current_cache_path
10. source_reason
11. candidate_label

### Phase 4: Add Bridge-Specific Annotation

Goal:

Label the probe set with a bridge-first task, not with the full old multiclass assumption.

Recommended labels:

1. `bridge_support`
2. `not_bridge`

Optional secondary field:

1. `support_subtype`

Allowed subtype values:

1. `full_support`
2. `bridge_support`
3. `nei`
4. `contradiction`

But the primary optimization target in this wave should remain:

`is_bridge_support`

Implementation options:

1. manual annotation on the small probe set
2. LLM prelabel + manual correction

Preferred path:

1. LLM prelabel with the new bridge label spec
2. human correction on the high-ambiguity rows

Expected output file:

1. `research_memory/emnlp_expand_then_compose/data/musique_bridge_probe_labels_20260403.jsonl`

### Phase 5: Add a Bridge Detector

Goal:

Do not retrain the whole atomic system first.

Instead, add a narrow experimental head:

> given `(question, need_unit, doc)` and existing atomic features, predict whether the doc is a real bridge for that need unit.

Recommended design:

1. keep current `full_support / contradiction / nei` path as-is
2. add one `bridge detector` branch
3. apply it only when:
   - alignment is not trivially zero
   - full-support probability is not already dominant
4. expose `bridge_support_prob` as a calibrated additive signal, not as a total rewrite

Preferred first model:

1. logistic regression or HistGBDT

Reason:

1. the main risk is supervision quality, not capacity
2. the first bridge head should stay debuggable

Target files:

1. `scripts/train_need_unit_scorer.py`
2. `scripts/requirement_beam_utils.py`
3. `scripts/annotate_need_unit_support.py`
4. `tests/test_setwise_selector.py`

### Phase 6: Rebuild Only a Small Cache

Goal:

Do not rerun large cache builds.

Use the same small smoke slice and rebuild only:

1. one conditional cache first
2. then cap2 / cap3 only if upstream gates pass

Target input:

1. `research_memory/emnlp_expand_then_compose/models/musique_need_unit_cache_qdmr_v1_pool100_ann20_limit10_conditional_8043_reuseopenie.json`

Target output:

1. `..._bridgeprobe_hybrid.json`

### Phase 7: Gate on Atomic and Shortlist Behavior

Goal:

Do not look at EM/F1 first.

Primary gates:

1. bridge probe recall must rise materially
2. bridge-like hard negative false positives must stay controlled
3. known true bridge docs that were near zero should receive visibly higher bridge-aware scores
4. those docs should start appearing in:
   - `candidate_source_preview`
   - `candidate_shortlist_preview`
5. conditional and cap-based variants should become distinguishable again on real 3-hop cases

Concrete gate suggestions:

1. bridge probe recall at top score threshold improves by at least `+20` absolute points versus current hybrid
2. bridge-like hard negative precision stays at least `0.7`
3. at least `3` known bridge docs from the current regression set move from near-zero support to clearly non-zero bridge-aware support
4. at least `2` smoke10 regression queries show the needed bridge doc entering shortlist/source preview

### Phase 8: Only Then Re-run Smoke10

Goal:

Use smoke10 as a confirmation gate, not as the primary debugger.

Success conditions:

1. selector `Recall@5` should no longer drop relative to the clean report
2. shortlist/source traces should show true bridge docs entering the frontier
3. `conditional / cap2 / cap3` should no longer collapse to the same behavior
4. query 6 style examples should show bridge recovery before reader QA is considered

Only if these hold:

1. consider expanding to `40`

## Immediate Coding Order

The next implementation round should land in this order:

1. add `pcrs_v2_bridge_label_spec_20260403.md`
2. add `scripts/mine_bridge_probe_examples.py`
3. export a first bridge probe candidate set
4. label the bridge probe set
5. add a bridge-detector training path to `scripts/train_need_unit_scorer.py`
6. add bridge-detector loading and scoring support in `scripts/requirement_beam_utils.py`
7. add bridge-aware cache rebuild support in `scripts/annotate_need_unit_support.py`
8. add focused tests in `tests/test_setwise_selector.py`
9. rebuild one small conditional cache
10. run atomic and shortlist gates
11. rerun smoke10 only if gates pass

## Acceptance Criteria

This wave is successful only if all of the following hold:

1. `bridge_support` is defined tightly enough that two reviewers would label most probe cases the same way
2. the probe set contains real bridge positives and real bridge-like hard negatives
3. the bridge detector raises scores for known useful middle-hop docs
4. those docs become visible in shortlist/source traces
5. smoke10 no longer shows the current pattern of:
   - changed sets everywhere
   - lower Recall@5
   - no meaningful budget differentiation

## Stop Conditions

Stop and reassess if any of the following happens:

1. bridge positives cannot be defined without collapsing into vague usefulness
2. probe annotation disagreement stays high after the spec is written
3. bridge detector improves probe metrics but still does not change shortlist/source traces
4. smoke10 still shows `Recall@5` degradation after bridge probe gains

## One-Line Task Definition

> Repair the atomic layer as a supervision problem by building bridge-first labels, mining hard negatives, training a narrow bridge detector, and gating on atomic plus shortlist behavior before any larger end-to-end expansion.
