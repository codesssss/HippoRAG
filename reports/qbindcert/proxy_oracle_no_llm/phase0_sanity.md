# Q-BindCert Phase-0 Sanity

## Decision

- decision: `PHASE0_NOT_CLEARED`
- phase1_justified: `False`
- rows: `100`
- llm_run: `False`
- llm_rows: `0`

## Gate Summary

| Test | Status | Key metrics |
|---|---|---|
| `typed_proposition_extraction` | `NOT_RUN` | rows=0; parse_ok_rate=0.0000; predicate_bucket_correct_proxy=0.0000; argument_extraction_correct_proxy=0.0000; source_span_provenance_correct_proxy=0.0000; no_hallucinated_proposition_proxy=0.0000; oracle_propositions=0; extracted_propositions=0 |
| `query_program_lattice` | `NOT_RUN` | rows=0; parse_ok_rate=0.0000; top3_any_program_correct_proxy=0.0000; top1_program_correct_proxy=0.0000; answer_variable_correct_proxy=0.0000; dependency_binding_correct_proxy=0.0000; question_type_operator_correct_proxy=0.0000 |
| `entity_normalization` | `NEEDS_MANUAL_AUDIT` | pairs=200; alias_pairs=5; negative_pairs=195; same_title_pairs=0; alias_merge_correct_proxy=1.0000; false_merge_rate_proxy=0.0000; same_title_duplicate_detection_proxy=1.0000 |
| `oracle_certificate_search` | `PASS` | rows=50; complete_certificate_found=1.0000; certificate_answer_agrees_with_gold=1.0000; no_non_contributing_duplicate_evidence=1.0000 |
| `answer_contrastive_oracle_certificate` | `PASS` | rows=50; gold_certificate_beats_wrong=1.0000; wrong_rejected_by_answer_terminal_binding_type=0.9600; query_program_error_caused_wrong_win=0.0000 |

## Interpretation

- `PASS`/`FAIL` are used only for oracle-controlled automatic tests.
- `NEEDS_MANUAL_AUDIT` means the proxy check did not falsify the gate, but Phase 1 is still blocked.
- `FAIL_PROXY` is sufficient to stop unless a manual audit explicitly overturns the proxy.
- This run does not implement full Q-BindCert and does not use reader/verifier feedback.

## LLM Probe

- available: `False`
- raw: `<think>

</think>

`
- error: ``

## Outputs

- json: `reports/qbindcert/proxy_oracle_no_llm/phase0_sanity.json`
- markdown: `reports/qbindcert/proxy_oracle_no_llm/phase0_sanity.md`
- extraction_samples: `reports/qbindcert/proxy_oracle_no_llm/phase0_extraction_samples.jsonl`
- program_samples: `reports/qbindcert/proxy_oracle_no_llm/phase0_program_samples.jsonl`
- entity_pairs: `reports/qbindcert/proxy_oracle_no_llm/phase0_entity_pairs.jsonl`
- certificate_rows: `reports/qbindcert/proxy_oracle_no_llm/phase0_certificate_rows.jsonl`
- contrastive_rows: `reports/qbindcert/proxy_oracle_no_llm/phase0_contrastive_rows.jsonl`
