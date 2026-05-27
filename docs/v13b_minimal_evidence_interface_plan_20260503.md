# V13B Minimal Evidence Interface Plan

Date: 2026-05-03

This is the active plan for continuing V13B after reviewing the schema-guided
OpenIE prompt probe. It replaces the earlier "fixed evidence-frame schema" idea
as the research mainline.

## Decision

Do not make a fixed relation schema the method.

The previous evidence-frame contract was useful as a diagnostic lens: it showed
that Qwen OpenIE triples do not align well with query obligations. But turning
that diagnosis into a hand-written relation inventory would create the same
problem as the V13B-specific prompt:

```text
benchmark failures -> hand-written relation families -> improved local result
```

That is too scene-specific and too easy to criticize as prompt/schema
engineering.

## Current Evidence

Claude's V13B-specific prompt is a valid probe, not a valid mainline method.

| Dataset | Original Qwen exact frame rate | V13B prompt exact frame rate | Change |
|---|---:|---:|---:|
| 2Wiki | 42.73% | 49.09% | +6.36 |
| HotpotQA | 17.13% | 12.96% | -4.17 |
| MuSiQue | 9.49% | 10.28% | +0.79 |

Selector-side changes are also weak:

| Dataset | Original feasible | V13B prompt feasible | Original gains/losses | V13B prompt gains/losses |
|---|---:|---:|---:|---:|
| 2Wiki | 53 | 58 | 7 / 0 | 6 / 0 |
| HotpotQA | 28 | 27 | 0 / 0 | 0 / 0 |
| MuSiQue | 23 | 18 | 1 / 0 | 1 / 0 |

Interpretation:

```text
The prompt confirms that corpus-side evidence representation matters.
It does not solve the problem in a stable, dataset-agnostic way.
```

## Main Principle

Use a schema-light evidence interface, not a fixed relation ontology.

The method should require the corpus graph to preserve source-grounded evidence
units, but it should not require every relation to be mapped into a hand-written
label such as `born_on`, `country_origin`, `composer`, or `president_of`.

Allowed:

| Component | Allowed content |
|---|---|
| Evidence unit | Source sentence/span, document id, title, entity mentions, predicate span, argument spans. |
| Query demand | Anchor mentions, variable slots, predicate phrase, answer slot, dependency between demands. |
| Matching | Query-time source-grounded support decision between one demand and one evidence unit/span. |
| Selection | Minimal connected evidence cover over matched units. |
| Normalization | Deterministic title cleanup, parenthetical title cleanup, exact mention normalization. |

Forbidden as mainline:

| Component | Why forbidden |
|---|---|
| Hand-written relation inventory | Turns failure diagnosis into dataset-specific ontology engineering. |
| Relation synonym table | Hides schema failure in matching rules. |
| Benchmark-specific OpenIE prompt | Improves local failure cases but can regress other datasets. |
| Lexical fallback | Bypasses the graph method. |
| Partial obligation grounding | Makes programs appear feasible while evidence roles remain uncovered. |
| More weight tuning | The bottleneck is representation and support, not score interpolation. |

## Revised Method Sketch

The research method should be:

```text
Query Demand Graph
  -> Source-Grounded Evidence Unit Matching
  -> Minimal Connected Evidence Cover
```

Not:

```text
Query triples and corpus triples are both forced into a fixed relation schema.
```

### Corpus Side

Build a schema-light evidence graph:

| Node / field | Purpose |
|---|---|
| document | Retrieval/candidate unit. |
| sentence/span | Source-grounded evidence carrier. |
| mention | Entity or literal mention in source text. |
| predicate span | Raw relation/action phrase in source text. |
| argument span | Subject/object/complement span if available. |
| title | Document-level anchor and alias source. |

The graph may still use OpenIE triples as one way to propose spans, but it must
not depend on a fixed relation key being correct.

### Query Side

Compile the query into a demand graph:

| Field | Purpose |
|---|---|
| anchor mention | Known entity/text anchor. |
| variable slot | Unknown entity/date/place/work to bind. |
| predicate phrase | Natural-language relation demand, not fixed ontology label. |
| answer slot | Final variable or comparison target. |
| dependency edge | How one variable enables another demand. |

The compiler must avoid turning descriptions into hard endpoints when they
refer to variables. For example, `recently abdicated queen` should usually be a
description to resolve, not a corpus endpoint that must match literally.

### Matching Side

The matcher should answer:

```text
Does this source span support this query demand under the currently bound
anchors/variables?
```

It should not answer:

```text
Do these two normalized relation labels exactly match?
```

The first implementation should be diagnostic-only. It should inspect gold docs
and candidate docs and report supportability without changing retrieval.

### Selection Side

After supportable units exist, select a minimal connected evidence cover:

| Constraint | Meaning |
|---|---|
| coverage | Retrieval-critical query demands must be supported. |
| binding consistency | Shared variables must resolve to compatible mentions. |
| locality | Evidence units should be connected through document/title/mention links. |
| minimality | Prefer fewer evidence units when coverage is equal. |

This can reuse V13B's current evidence-set size and connectivity machinery
later, but it should not be tuned until source-grounded matching is validated.

## Execution Plan

### Phase 0: Freeze Probe Status

Mark these artifacts as probes, not mainline:

| Artifact | Status |
|---|---|
| `src/hipporag/prompts/templates/triple_extraction_v13b.py` | Probe only. |
| `scripts/reextract_openie_v13b.py` | Probe runner only. |
| `run_logs/v13b_structural_pool100_limit100_v13bprompt_20260503/` | Diagnostic result only. |
| `docs/v13b_qwen_evidence_frame_contract_20260503.md` | Superseded diagnostic note. |

Acceptance criterion:

```text
No final mainline result should cite V13B prompt as the proposed method.
```

### Phase 1: Source-Support Audit

Implement a diagnostic script:

```text
analyze_v13b_source_support_audit.py
```

Inputs:

| Input | Purpose |
|---|---|
| source report | Gold docs and candidate docs. |
| selector JSON | Retrieval-critical demand ids and variable bindings. |
| query obligation cache | Query-side demand graph. |
| OpenIE JSON / corpus text | Source sentences and existing fact units. |

Outputs:

| Output | Purpose |
|---|---|
| gold_source_support_available | Gold source text appears to support the demand. |
| openie_exact_available | Existing OpenIE triple already supports it. |
| openie_failed_but_source_supports | Main evidence that fixed OpenIE schema is the wrong bottleneck. |
| descriptive_endpoint_query_issue | Query compiler made a descriptive phrase into a hard endpoint. |
| unresolved_variable_issue | Upstream variable binding prevents support check. |

Acceptance criterion:

```text
We can separate "source text supports demand but OpenIE label fails" from
"source text itself does not support demand".
```

### Phase 2: Minimal Evidence Unit Builder

Implement a schema-light builder:

```text
build_minimal_evidence_units.py
```

It should produce:

| Field | Required |
|---|---|
| `doc_index` | yes |
| `title` | yes |
| `span_text` | yes |
| `mention_surfaces` | yes |
| `predicate_text` | optional |
| `argument_texts` | optional |
| `source` | `openie`, `sentence`, or `hybrid` |

No fixed relation inventory is allowed.

Acceptance criterion:

```text
The builder can represent evidence even when OpenIE emits generic or wrong
relation labels.
```

### Phase 3: Source-Grounded Demand Matching Probe

Implement a matcher probe:

```text
match_query_demands_to_evidence_units.py
```

Initial matching must be conservative:

| Requirement | Rationale |
|---|---|
| Bound anchors must occur in the source span or title. | Avoid fuzzy rescue. |
| Predicate content words must have textual support or span-level semantic support. | Avoid relation-label exactness. |
| Variable-binding candidates must come from explicit mentions. | Avoid hallucinated bindings. |
| Each match must retain source provenance. | Keep method auditable. |

The first version should be diagnostic-only and should not alter retrieval.

Acceptance criterion:

```text
On gold docs, support coverage improves over exact OpenIE frame coverage without
using hand-written relation synonyms.
```

### Phase 4: Minimal Connected Evidence Cover

Only after Phase 3 succeeds, integrate with selector:

```text
select_minimal_connected_evidence_cover.py
```

Selection objective:

```text
Cover retrieval-critical demands with a small, binding-consistent, connected
set of evidence units.
```

No score-weight fusion should be introduced. If tie-breaking is needed, use
deterministic structural order:

```text
coverage > binding consistency > connectivity > source rank > doc order
```

Acceptance criterion:

```text
The selector changes top5 because it found a more complete evidence cover, not
because a weighted reranker preferred different documents.
```

### Phase 5: Evaluation

Run in this order:

| Step | Dataset | Limit | Purpose |
|---|---|---:|---|
| 1 | 2Wiki / HotpotQA / MuSiQue | 100 | Debug support coverage and selector behavior. |
| 2 | same | 500 | Check stability before full run. |
| 3 | same | 1000 | Final comparison only after support audit passes. |

Report at least:

| Metric | Why |
|---|---|
| source_support_available | Whether source text can satisfy demands. |
| openie_failed_but_source_supports | Whether OpenIE schema is the bottleneck. |
| support_match_coverage | Whether schema-light matching helps. |
| evidence_cover_feasible | Whether full demand coverage is possible. |
| R@5 / all-gold@5 | Retrieval outcome. |
| EM / F1 | Reader outcome, only after retrieval is stable. |

## Immediate Next Step

Start with Phase 1.

Do not modify the selector yet.
Do not modify the OpenIE prompt yet.
Do not add relation aliases.

The next code change should be a source-support audit that answers:

```text
When exact OpenIE grounding fails, does the gold source text itself contain
enough anchored evidence to support the query demand?
```

If yes, V13B should move away from fixed OpenIE relation labels.
If no, the problem is query decomposition, candidate coverage, or missing source
evidence rather than schema.

## Phase 1 Results

Implemented:

```text
analyze_v13b_source_support_audit.py
```

Validation:

```text
env PYTHONDONTWRITEBYTECODE=1 /mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python -m pytest tests/v13b -q
# 140 passed
```

Limit100 deg30 original-Qwen audit:

| Dataset | Retrieval-critical obligations | OpenIE exact available | OpenIE failed but source supports | Unresolved/descriptive query issue | Main signal |
|---|---:|---:|---:|---:|---|
| 2Wiki | 220 | 127 / 57.73% | 18 / 8.18% | 22 / 10.00% | OpenIE exact often available |
| HotpotQA | 216 | 38 / 17.59% | 58 / 26.85% | 39 / 18.06% | Source support exceeds OpenIE exact gap |
| MuSiQue | 253 | 26 / 10.28% | 37 / 14.62% | 85 / 33.60% | Query compiler or binding bottleneck |

Interpretation:

```text
HotpotQA strongly supports the schema-light direction: many gold source spans
contain anchored predicate evidence even when exact OpenIE grounding fails.

MuSiQue is different: the largest visible issue is unresolved variables and
descriptive/query-side endpoint problems. For MuSiQue, better source matching
alone is unlikely to solve the bottleneck until the query demand graph/binding
chain is diagnosed.

2Wiki is already relatively compatible with exact OpenIE; source-support helps
less, which explains why V13B prompt probes improved exact frame rate but did
not translate into a robust selector gain.
```

Updated next step:

```text
Do not tune selector.
Do not continue prompt/schema specialization.
Split the next work by dataset failure mode:
1. HotpotQA: implement a diagnostic source-grounded demand matcher probe.
2. MuSiQue: implement query demand/binding-chain audit before matcher changes.
3. 2Wiki: keep as a sanity dataset; do not overfit to it.
```

## Phase 2/3 Diagnostic Probe Results

Implemented:

```text
build_minimal_evidence_units.py
match_query_demands_to_evidence_units.py
```

Validation:

```text
env PYTHONDONTWRITEBYTECODE=1 /mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python -m pytest tests/v13b -q
# 146 passed
```

The matcher is still diagnostic-only.  It does not modify retrieval, selector,
PPR, fallback, OpenIE prompts, or candidate generation.

Clean constraints:

| Constraint | Current implementation |
|---|---|
| Fixed relation schema | Not used |
| Relation synonym table | Not used |
| Selector pseudo-match seed | Disabled by default |
| LLM call | Not used |
| Ranking/fusion weight | Not used |
| Source provenance | Required in every match |

Limit100, original Qwen OpenIE, structural candidate pool deg30, gold-doc diagnostic:

| Dataset | Retrieval-critical obligations | OpenIE exact | Source-grounded matches | New over OpenIE exact | Exact or source grounded |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 220 | 127 / 57.73% | 89 / 40.45% | 19 / 8.64% | 146 / 66.36% |
| HotpotQA | 216 | 38 / 17.59% | 88 / 40.74% | 56 / 25.93% | 94 / 43.52% |
| MuSiQue | 253 | 26 / 10.28% | 58 / 22.92% | 39 / 15.42% | 65 / 25.69% |

HotpotQA candidate-doc diagnostic:

| Dataset | Retrieval-critical obligations | OpenIE exact | Source-grounded matches | New over OpenIE exact | Exact or source grounded |
|---|---:|---:|---:|---:|---:|
| HotpotQA candidate docs | 216 | 41 / 18.98% | 98 / 45.37% | 65 / 30.09% | 106 / 49.07% |

Reports:

```text
run_logs/v13b_structural_pool100_limit100_deg30_20260502/failure_reports/2wikimultihopqa_source_grounded_demand_matcher.md
run_logs/v13b_structural_pool100_limit100_deg30_20260502/failure_reports/hotpotqa_source_grounded_demand_matcher.md
run_logs/v13b_structural_pool100_limit100_deg30_20260502/failure_reports/hotpotqa_candidate_source_grounded_demand_matcher.md
run_logs/v13b_structural_pool100_limit100_deg30_20260502/failure_reports/musique_source_grounded_demand_matcher.md
```

Interpretation:

```text
HotpotQA has a real source-grounded gap: many retrieval-critical demands can
be matched to anchored source sentences even when exact OpenIE fact grounding
fails.

The signal is visible inside the current structural candidate pool, not only
on oracle gold docs. This means the next clean experiment should be a minimal
connected evidence cover over source-grounded evidence units.
```

Caveat:

```text
The current matcher proves source-support availability, not final binding
quality. Some variable bindings are broad explicit mention sets from a sentence.
The next selector-facing step must enforce binding consistency and connected
coverage before changing top5.
```

## Phase 4 Diagnostic Cover Results

Implemented:

```text
select_minimal_connected_evidence_cover.py
```

Validation:

```text
env PYTHONDONTWRITEBYTECODE=1 /mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python -m pytest tests/v13b -q
# 151 passed
```

The cover selector is still diagnostic-only.  It selects a small evidence set by
deterministic structural order:

```text
coverage > binding consistency > connectivity > source rank > doc order
```

It does not use weighted rank fusion, fallback, PPR tuning, relation schemas, or
relation synonym tables.

Limit100 diagnostic cover results:

| Dataset | Covered demand rate | Full cover feasible | Connected cover | Uses source-grounded match | Cover R@5 | V13B selector R@5 | Source R@5 | Cover all-gold@5 | V13B selector all-gold@5 | Gains/losses vs selector |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 67.27% | 52/100 | 42/100 | 22/100 | 0.9600 | 0.9550 | 0.9350 | 0.9100 | 0.9200 | 2 / 2 |
| HotpotQA | 48.15% | 29/100 | 63/100 | 52/100 | 0.9400 | 0.9300 | 0.9300 | 0.8900 | 0.8700 | 2 / 0 |
| MuSiQue | 33.20% | 21/100 | 54/100 | 43/100 | 0.6892 | 0.6825 | 0.6775 | 0.4100 | 0.4000 | 3 / 1 |

Reports:

```text
run_logs/v13b_structural_pool100_limit100_deg30_20260502/failure_reports/2wikimultihopqa_minimal_connected_evidence_cover.md
run_logs/v13b_structural_pool100_limit100_deg30_20260502/failure_reports/hotpotqa_minimal_connected_evidence_cover.md
run_logs/v13b_structural_pool100_limit100_deg30_20260502/failure_reports/musique_minimal_connected_evidence_cover.md
```

Interpretation:

```text
The cover objective is now mechanism-positive: it improves R@5 on all three
limit100 datasets relative to both source top5 and current V13B selector.

However, it is not selector-ready as a blanket replacement.  2Wiki improves R@5
but loses 1 point all-gold@5 relative to the current selector, and MuSiQue has
one regression.  The next integration must be conservative: a source-grounded
cover may override the selector only when it preserves existing full-gold
coverage or provides a strictly more complete binding-consistent evidence cover.
```

Gold is not available at inference time, so the first non-oracle admission rule
tested is:

```text
admit cover only when full_cover_feasible and connected; otherwise keep the
current V13B selector top5.
```

Certified admission diagnostic:

| Dataset | Admitted covers | Certified R@5 | Selector R@5 | Certified all-gold@5 | Selector all-gold@5 | Gains/losses vs selector |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | 23/100 | 0.9575 | 0.9550 | 0.9200 | 0.9200 | 1 / 1 |
| HotpotQA | 25/100 | 0.9350 | 0.9300 | 0.8800 | 0.8700 | 1 / 0 |
| MuSiQue | 21/100 | 0.6825 | 0.6825 | 0.4000 | 0.4000 | 0 / 0 |

This is safer than unconditional replacement, but still not final:

```text
2Wiki still has one query-level loss under the non-oracle certificate.  The next
step should inspect admitted-loss cases and strengthen the structural admission
condition, not add a score weight.
```

Admitted-loss analysis found the key failure:

```text
2Wiki query 8 asks whether two film directors are from the same country.
The query compiler produced four triples, but typed retrieval kept only the two
director-finding bridge obligations and demoted the two country constraints to
non-retrieval materialized obligations.  The cover therefore looked complete
while missing one country evidence document.
```

Materialized-obligation scope diagnostic:

| Dataset | Scope | Certified R@5 | Selector R@5 | Certified all-gold@5 | Selector all-gold@5 | Gains/losses vs selector |
|---|---|---:|---:|---:|---:|---:|
| 2Wiki | materialized | 0.9600 | 0.9550 | 0.9300 | 0.9200 | 1 / 0 |
| HotpotQA | materialized | 0.9350 | 0.9300 | 0.8800 | 0.8700 | 1 / 0 |
| MuSiQue | materialized | 0.6792 | 0.6825 | 0.3900 | 0.4000 | 0 / 1 |

Interpretation:

```text
The right fix is not a score trick.  It is query-demand scope control.

Some comparison constraints are genuinely retrieval-critical and should be
covered; blindly treating all materialized constraints as retrieval-critical
hurts MuSiQue.  The next research step is a query-demand completeness audit that
promotes only answer-bearing or comparison-binding constraints into the cover.
```
