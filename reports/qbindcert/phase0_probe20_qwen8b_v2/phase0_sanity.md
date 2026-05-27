# Q-BindCert Phase-0 Sanity

## Decision

- decision: `STOP_PHASE0_FAIL`
- phase1_justified: `False`
- rows: `100`
- llm_run: `True`
- llm_rows: `20`

## Gate Summary

| Test | Status | Key metrics |
|---|---|---|
| `typed_proposition_extraction` | `FAIL_PROXY` | rows=20; parse_ok_rate=1.0000; predicate_bucket_correct_proxy=0.9400; argument_extraction_correct_proxy=0.6200; source_span_provenance_correct_proxy=1.0000; no_hallucinated_proposition_proxy=0.9804; oracle_propositions=50; extracted_propositions=51 |
| `query_program_lattice` | `FAIL_PROXY` | rows=20; parse_ok_rate=1.0000; top3_any_program_correct_proxy=0.5500; top1_program_correct_proxy=0.5500; answer_variable_correct_proxy=0.7000; dependency_binding_correct_proxy=1.0000; question_type_operator_correct_proxy=0.9500 |
| `entity_normalization` | `NEEDS_MANUAL_AUDIT` | pairs=200; alias_pairs=5; negative_pairs=195; same_title_pairs=0; alias_merge_correct_proxy=1.0000; false_merge_rate_proxy=0.0000; same_title_duplicate_detection_proxy=1.0000 |
| `oracle_certificate_search` | `PASS` | rows=50; complete_certificate_found=1.0000; certificate_answer_agrees_with_gold=1.0000; no_non_contributing_duplicate_evidence=1.0000 |
| `answer_contrastive_oracle_certificate` | `PASS` | rows=50; gold_certificate_beats_wrong=1.0000; wrong_rejected_by_answer_terminal_binding_type=0.9600; query_program_error_caused_wrong_win=0.0000 |

## Interpretation

- `PASS`/`FAIL` are used only for oracle-controlled automatic tests.
- `NEEDS_MANUAL_AUDIT` means the proxy check did not falsify the gate, but Phase 1 is still blocked.
- `FAIL_PROXY` is sufficient to stop unless a manual audit explicitly overturns the proxy.
- This run does not implement full Q-BindCert and does not use reader/verifier feedback.

## LLM Probe

- available: `True`
- raw: `<think>

</think>

OK`
- error: ``

## Outputs

- json: `reports/qbindcert/phase0_probe20_qwen8b_v2/phase0_sanity.json`
- markdown: `reports/qbindcert/phase0_probe20_qwen8b_v2/phase0_sanity.md`
- extraction_samples: `reports/qbindcert/phase0_probe20_qwen8b_v2/phase0_extraction_samples.jsonl`
- program_samples: `reports/qbindcert/phase0_probe20_qwen8b_v2/phase0_program_samples.jsonl`
- entity_pairs: `reports/qbindcert/phase0_probe20_qwen8b_v2/phase0_entity_pairs.jsonl`
- certificate_rows: `reports/qbindcert/phase0_probe20_qwen8b_v2/phase0_certificate_rows.jsonl`
- contrastive_rows: `reports/qbindcert/phase0_probe20_qwen8b_v2/phase0_contrastive_rows.jsonl`
