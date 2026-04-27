# Q-BindCert Phase 0 Results - 2026-04-27

## Decision

`Q-BindCert` should not enter Phase 1 under the agreed blocking rules.

The oracle-controlled certificate layer produced clear positive signal, but the Qwen3-8B `/no_think` upstream gates failed:

- Typed proposition extraction: `FAIL_PROXY`
- Query program lattice: `FAIL_PROXY`
- Entity normalization: `NEEDS_MANUAL_AUDIT`
- Oracle certificate search: `PASS`
- Oracle answer-contrastive certificate audit: `PASS`

Overall decision:

```text
STOP_PHASE0_FAIL
```

Phase 1 is not justified without a different upstream compiler/extractor design or manual correction layer.

## Run

Command:

```text
.venv-hipporag/bin/python scripts/qbindcert_phase0_sanity.py \
  --run-llm \
  --llm-base-url http://localhost:8043/v1 \
  --llm-model qwen3-8b-train \
  --llm-limit 20 \
  --limit 100 \
  --oracle-limit 50 \
  --entity-pair-limit 200 \
  --output-dir reports/qbindcert/phase0_probe20_qwen8b_v2
```

The prompts included `/no_think`. The endpoint returned parseable JSON for all 20 LLM rows.

Throughput:

```text
20 queries, 139.411 seconds, 6.971 seconds/query
```

## Metrics

| Gate | Status | Main numbers |
|---|---|---|
| Typed proposition extraction | `FAIL_PROXY` | parse ok 1.00; predicate bucket 0.94; argument extraction 0.62; provenance 1.00; no hallucination 0.9804 |
| Query program lattice | `FAIL_PROXY` | parse ok 1.00; top-3 any correct 0.55; top-1 correct 0.55; answer variable 0.70; dependency 1.00; type/operator 0.95 |
| Entity normalization | `NEEDS_MANUAL_AUDIT` | alias merge proxy 1.00; false merge proxy 0.00; same-title detection proxy 1.00, but no same-title pairs appeared in the 200-pair sample |
| Oracle certificate search | `PASS` | complete certificate 1.00; answer agrees 1.00; no duplicate evidence 1.00 |
| Oracle answer-contrastive certificate | `PASS` | gold beats wrong 1.00; wrong structurally rejected 0.96; program-error wrong win 0.00 |

## Interpretation

This is mixed but useful:

- Positive: the certificate search formulation is not falsified under oracle conditions. Given correct propositions/program/entities, the lexicographic certificate key separates gold from plausible wrong on the sampled hard cases.
- Negative: Qwen3-8B zero-shot extraction/compilation is not reliable enough to support the method end-to-end. The program lattice gate is especially weak: top-3 any-correct is 0.55 against a 0.75 gate.
- Negative: extraction produces well-formed, grounded JSON, but misses bridge arguments/hops often enough that argument coverage is only 0.62 against a 0.80 gate.

The important distinction is that Q-BindCert's search mechanism has positive signal, while the train-free upstream semantic compiler does not.

## Failure Modes

Observed extraction failures:

- Missing bridge relation when the answer document was easy to extract, e.g. extracting only `Ermengarde of Tours -> date_of_death -> 20 March 851` and missing `Lothair II -> mother -> Ermengarde of Tours`.
- Predicate bucket was often recoverable after canonicalization, but argument coverage remained low.
- Some relation direction errors remained, e.g. `child_of` arguments reversed relative to the evidence demand.

Observed program failures:

- Comparison questions often compiled the comparison attribute instead of the answer entity.
- Bridge-comparison questions missed one side of the comparison chain.
- Family-chain inference sometimes collapsed to the wrong relation, e.g. paternal grandfather becoming a spouse relation.
- Top-3 lattice did not provide enough diversity; failures were often correlated across all programs.

## Paper Implication

Q-BindCert is currently best recorded as a Phase-0 blocked ambitious direction:

```text
Oracle certificates work; zero-shot Qwen3-8B query compilation and typed proposition extraction do not clear the reliability floor.
```

This supports the earlier LCPS caution: the main risk is not certificate search, but the typed extraction/query compilation substrate.

## Artifacts

- Plan: `research_memory/emnlp_expand_then_compose/34_qbindcert_phase0_plan.md`
- Script: `scripts/qbindcert_phase0_sanity.py`
- Tests: `tests/dpathrag/test_qbindcert_phase0.py`
- Report: `reports/qbindcert/phase0_sanity.md`
- JSON: `reports/qbindcert/phase0_sanity.json`
- Samples:
  - `reports/qbindcert/phase0_extraction_samples.jsonl`
  - `reports/qbindcert/phase0_program_samples.jsonl`
  - `reports/qbindcert/phase0_entity_pairs.jsonl`
  - `reports/qbindcert/phase0_certificate_rows.jsonl`
  - `reports/qbindcert/phase0_contrastive_rows.jsonl`

## Validation

```text
.venv-hipporag/bin/python -m py_compile scripts/qbindcert_phase0_sanity.py tests/dpathrag/test_qbindcert_phase0.py
.venv-hipporag/bin/python -m pytest tests/dpathrag/test_qbindcert_phase0.py -q
5 passed, 2 warnings
```

