# Propose-Verify Single-Swap Controller

Date: `2026-04-14`

## Goal

Implement a narrower action controller after the failure of the zero-shot global utility controllers.

The controller keeps the existing actionized skeleton:

- `bridge_append`
- baseline-only scaffold
- single slot-preserving swap
- `keep ∪ {d ↔ i}`

But it changes the controller role:

- `dryrun` proposes the top-1 relaxed legal action
- verifier only checks:
  - whether the candidate supports the earliest missing claim
  - whether the replacee is the unique supporter of any earlier claim

This is a verifier, not a chooser.

## Files

- implementation: [scripts/eval_causal_qwen3.py](/mnt/nvme/code/HippoRAG/scripts/eval_causal_qwen3.py)
- prompt parser / message builder: [scripts/run_swap_utility_judge.py](/mnt/nvme/code/HippoRAG/scripts/run_swap_utility_judge.py)
- selector tests: [tests/test_setwise_selector.py](/mnt/nvme/code/HippoRAG/tests/test_setwise_selector.py)
- verifier parser tests: [tests/test_swap_utility_judge.py](/mnt/nvme/code/HippoRAG/tests/test_swap_utility_judge.py)
- offline audit: [scripts/audit_action_swap_propose_verify.py](/mnt/nvme/code/HippoRAG/scripts/audit_action_swap_propose_verify.py)

## Implemented Pieces

### New assemble mode

- `action_swap_propose_verify`

Wired into the existing `bridge_append` action-controller patch point:

- after CE `ranking_rows` are available
- before `materialize_reader_top_positions(...)`

### New helpers

- `build_propose_verify_claims(...)`
  - heuristic tiers first
  - forced flat fallback
  - max 2 claims
- `compute_claim_ce_witness_details(...)`
  - CE chooses the best witness unit per `(claim, doc)`
  - witness units reuse:
    - `full_doc`
    - `title_plus_1sent`
    - `title_plus_2sent`
- `_select_action_swap_proposal_job(...)`
  - relaxed legality
  - dryrun-style top-1 proposal
- `_run_claim_support_verifier_jobs(...)`
  - binary claim support prompt
  - supports injected cached results for tests
- `score_action_swap_propose_verify_jobs(...)`
  - computes earliest unsupported claim
  - runs gain and preservation checks
- `select_action_swap_propose_verify(...)`
  - executes only the proposed action
  - otherwise keeps

### New trace fields

- `claims`
- `claim_mode`
- `proposal_action_present`
- `proposal_candidate_pool_position`
- `proposal_replace_pool_position`
- `earliest_unsupported_claim_id`
- `earliest_unsupported_claim_text`
- `gain_verifier_verdict`
- `gain_verifier_reason`
- `preservation_unique_support_claim_ids`
- `claim_supports_before`

## Decision Rule

Let `a_prop = (d, i)` be the dryrun proposal.

The controller swaps iff:

1. `d` supports the earliest unsupported claim
2. `i` is not the unique supporter of any earlier claim

Otherwise it keeps.

There is no global utility score, no noisy-or aggregation, and no action search in the verifier.

## Current Validation

### Static checks

- `python -m py_compile scripts/eval_causal_qwen3.py scripts/run_swap_utility_judge.py tests/test_setwise_selector.py tests/test_swap_utility_judge.py scripts/audit_action_swap_propose_verify.py`

### Unit tests

- `pytest tests/test_swap_utility_judge.py -q`
- `pytest tests/test_setwise_selector.py -q -k 'propose_verify'`

The tests cover:

- heuristic tier claim extraction
- flat fallback claim extraction
- gain/preservation scoring at action level
- successful execution when both checks pass
- preservation veto
- no-unsupported-claim keep path

## Offline Audit Plan

The intended first evaluation is offline verifier audit, not reader smoke.

Audit command template:

```bash
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy \
.venv-hipporag/bin/python scripts/audit_action_swap_propose_verify.py \
  --dataset musique \
  --oracle_relaxed_report run_logs/musique_action_swap_oracle_relaxed_20260413.json \
  --query_report outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260409smoke.json \
  --dryrun_report outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_action_swap_v0_dryrun_qatopk5_20260413impl.json \
  --judge_report outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_action_swap_v0_judge_qatopk5_20260413impl.json \
  --output_json run_logs/musique_action_swap_propose_verify_audit_20260414.json \
  --output_md run_logs/musique_action_swap_propose_verify_audit_20260414.md \
  --output_csv run_logs/musique_action_swap_propose_verify_audit_20260414.csv \
  --ce_device cuda:0 \
  --setwise_late_rerank_judge_backend responses \
  --setwise_late_rerank_judge_model gpt-5.4-mini \
  --setwise_late_rerank_judge_base_url http://localhost:8000/v1
```

Expected audit readout:

- oracle-positive pass rate
- oracle-negative pass rate
- dryrun-selected pass rate
- judge-selected pass rate
- reject reason counts
- sample oracle-positive rejections

## Intended Next Step

Only if the offline audit shows separation:

- positive swaps pass materially more often than negative swaps
- dryrun proposals are improved by verifier veto

Then run small online smoke:

- MuSiQue 100
- 2Wiki 100

Otherwise, stop this controller line and record it as another bounded negative result.
