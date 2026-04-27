# Q-BindCert Phase 0 Blocking Plan - 2026-04-27

## Decision

Run `Q-BindCert` only through a Phase 0 blocking sanity gate. Do not implement the full method unless the gate clearly passes. Marginal evidence counts as failure for engineering purposes.

## Method Snapshot

`Q-BindCert` reframes multi-hop retrieval as query-compiled evidence certificate search:

```text
O(q) = (P*, a*, theta*, S*, Pi*)
```

- `P*`: selected query program from a small program lattice.
- `a*`: answer candidate.
- `theta*`: variable binding.
- `S*`: evidence set under the reader budget.
- `Pi*`: certificate mapping demands to evidence, bindings, and answer-terminal support.

The search object is `(program, answer, binding, evidence)`, not a passage path or reader likelihood trace.

## Hard Prohibitions

- No reader likelihood inner loop.
- No verifier-guided iterative search.
- No pairwise binary admission classifier.
- No NLI global proof score.
- No cross-pool agreement as a positive signal.
- No PPR-style uniform diffusion as the main operator.
- No single free-form LLM answer candidate listing.
- No weighted score soup.

## Phase 0 Tests

### Test 1: Typed Proposition Extraction

Input: 100 2Wiki dev/cache queries, gold support docs only.

Gate:

| Check | Threshold |
|---|---:|
| Predicate bucket correct | >=80% |
| Argument extraction correct | >=80% |
| Source span/provenance correct | >=90% |
| No hallucinated proposition | >=90% |

Automation can provide a silver/proxy score against 2Wiki evidence triples, but the final gate needs manual inspection of 50-100 propositions.

### Test 2: Query Program Lattice

Input: same 100 queries, top-3 programs.

Gate:

| Check | Threshold |
|---|---:|
| At least one program correct in top-3 | >=75% |
| Top-1/single program correct | >=60% |
| Answer variable correct | >=75% |
| Dependency binding correct | >=75% |
| Question type/operator correct | >=80% |

Error modes to report:

- `variable_mismatch`
- `type_wrong`
- `dependency_wrong`
- `answer_variable_wrong`
- `operator_or_relation_wrong`
- `parse_failed`

### Test 3: Entity Normalization

Input: 200 mention pairs from gold/candidate docs.

Gate:

| Check | Threshold |
|---|---:|
| Alias merge correct | >=85% |
| False merge rate | <=10% |
| Same-title duplicate detection | >=90% |

This test must explicitly include same-title duplicate hygiene because the NREV audit found same-title replacement as a real confound.

### Test 4: Oracle Certificate Search

Input: 50 queries, gold support docs, oracle/silver-correct program.

Gate:

| Check | Threshold |
|---|---:|
| Complete certificate found | >=70% |
| Certificate answer agrees with gold | >=65% |
| Evidence set has no non-contributing duplicate | >=90% |

This isolates whether the certificate formulation works when upstream extraction/program/entity assumptions are satisfied.

### Test 5: Oracle Answer-Contrastive Certificate Audit

Input: 50 hard queries, gold answer vs plausible wrong answer.

Correction from review: this test is oracle-controlled. It should use oracle typed propositions, oracle/correct program, and oracle entity normalization so that failure is attributable to certificate search and answer competition, not upstream extraction noise.

Gate:

| Check | Threshold |
|---|---:|
| Gold certificate beats best wrong | >=65% |
| Wrong rejected by answer-terminal/binding/type | >=40% |
| Query-program error causing wrong win | <=25% |

Under oracle setting the last rate should be near zero; it is tracked to catch leakage from the program compiler into this test.

## Decision Rule

| Result | Decision |
|---|---|
| All five tests clearly pass | Proceed to Phase 1 MVP |
| Any test fails | Stop Q-BindCert and record a negative diagnostic |
| Any test is marginal | Treat as fail |
| Any manual-quality gate lacks audit evidence | Do not proceed; status is `NEEDS_MANUAL_AUDIT` |

## Tonight's Implementation Scope

Allowed:

- Implement Phase 0 scaffold and report writer.
- Run oracle/proxy tests.
- Generate manual-audit sample files.
- Probe Qwen endpoint and run limited LLM extraction/program generation if stable.
- Record positive/negative signal.

Not allowed:

- Full Q-BindCert MVP.
- Tuning thresholds after seeing results.
- Treating silver proxy scores as final manual PASS.

## Planned Artifacts

- `scripts/qbindcert_phase0_sanity.py`
- `tests/dpathrag/test_qbindcert_phase0.py`
- `reports/qbindcert/phase0_sanity.md`
- `reports/qbindcert/phase0_sanity.json`
- `reports/qbindcert/phase0_extraction_samples.jsonl`
- `reports/qbindcert/phase0_program_samples.jsonl`
- `reports/qbindcert/phase0_entity_pairs.jsonl`
- `reports/qbindcert/phase0_certificate_rows.jsonl`

