# Propose-Verify `/no_think` Fix And MuSiQue Subset20 Result

Date: `2026-04-14`

## Scope

This note records the narrow follow-up after implementing `action_swap_propose_verify`.

The goal was not to expand experiments. The goal was to answer one specific question:

> Was the initial MuSiQue failure caused by the verifier interface, or by the method itself?

## What Was Broken First

The first MuSiQue full audit for `action_swap_propose_verify` was not cleanly interpretable.

Observed failure mode:

- `gain_verifier_verdict = None`
- `parse_succeeded = False`
- `finish_reason = length`
- raw responses were truncated inside `<think> ...`
- parser saw `missing_verdict` / `missing_reason`

This meant the initial all-reject result mixed together:

- genuine semantic rejects
- output-format failures from local `qwen3-8b-train`

So the first full MuSiQue audit could not be used as a method conclusion.

## Fix Applied

The project already had an existing local-Qwen control pattern:

- `SETWISE_LLM_NO_THINK_PREFIX = "/no_think"`

The claim-support verifier prompt was updated to reuse that exact pattern.

Implementation change:

- [scripts/run_swap_utility_judge.py](/mnt/nvme/code/HippoRAG/scripts/run_swap_utility_judge.py)

Changes:

- prepend `/no_think` to the claim-support user prompt
- explicitly forbid `<think>` and analysis text
- require exactly two lines:
  - `VERDICT: supported / not_supported`
  - `REASON: ...`

Validation:

- [tests/test_swap_utility_judge.py](/mnt/nvme/code/HippoRAG/tests/test_swap_utility_judge.py)
- `pytest tests/test_swap_utility_judge.py -q`
- result: `7 passed`

## Failure Slice Check

Before rerunning any larger audit, a 5-case MuSiQue oracle-positive failure slice was inspected.

After `/no_think`:

- responses no longer died in long `<think>` chains
- `finish_reason` became `stop`
- parser successfully extracted structured verdicts

Representative post-fix behavior:

- some cases now cleanly returned `VERDICT: not_supported`
- some cases surfaced `earliest unsupported = None`

This showed the interface bug was real and was fixed, but also exposed a second-layer method problem.

## MuSiQue Subset20 Audit

To avoid rerunning the full action set again, a subset audit was run:

- 10 oracle-positive legal swaps
- 10 oracle-negative legal swaps
- dryrun-selected and judge-selected actions were preferentially retained when possible

Artifacts:

- [musique_action_swap_propose_verify_subset20_20260414.json](/mnt/nvme/code/HippoRAG/run_logs/musique_action_swap_propose_verify_subset20_20260414.json)
- [musique_action_swap_propose_verify_subset20_20260414.md](/mnt/nvme/code/HippoRAG/run_logs/musique_action_swap_propose_verify_subset20_20260414.md)
- [musique_action_swap_propose_verify_subset20_20260414.csv](/mnt/nvme/code/HippoRAG/run_logs/musique_action_swap_propose_verify_subset20_20260414.csv)

## Main Result

After the `/no_think` fix, the method still produced:

- total actions: `20`
- oracle-positive: `10`
- oracle-negative: `10`
- proposal pass rate: `0.0%`
- oracle-positive pass rate: `0/10`
- oracle-negative pass rate: `0/10`
- dryrun-selected pass rate: `0/16`
- judge-selected pass rate: `0/2`

So the clean post-fix result is still:

> `action_swap_propose_verify` has no action-level separation on MuSiQue subset20.

## Reject Breakdown

Post-fix reject reasons:

- `gain_verifier_reject`: `11`
- `no_unsupported_claim`: `9`

This is the important decomposition.

### Oracle-positive rows

Among the `10` oracle-positive rows:

- `7` were rejected as `no_unsupported_claim`
- `3` were rejected as `gain_verifier_reject`

### Oracle-negative rows

Among the `10` oracle-negative rows:

- `8` were rejected as `gain_verifier_reject`
- `2` were rejected as `no_unsupported_claim`

## Interpretation

This separates two distinct failure modes.

### 1. Claim construction failure

`no_unsupported_claim` on oracle-positive rows means:

- the current scaffold was judged as already satisfying the claim layer
- but the oracle still says a swap helps

So the current claim layer is too coarse or misaligned.

It fails to express the missing information that the positive swap is actually repairing.

### 2. Witness + verifier failure

`gain_verifier_reject` on oracle-positive rows means:

- a missing claim was identified
- but the proposed candidate was still judged `not_supported`

So even when the controller does see a gap, the current:

- heuristic claim text
- CE-selected witness unit
- binary support verifier

still fails to lift positive swaps.

## What This Means

The earlier all-reject MuSiQue run was partly polluted by verifier formatting failure.

That is now fixed.

After fixing the interface, the method still fails.

So the correct final conclusion is:

> The initial failure was not purely a `/no_think` issue.  
> `/no_think` was necessary to make the audit interpretable, but after the fix, `propose_verify` still fails on MuSiQue because both claim-gap detection and claim-support verification remain misaligned with oracle-positive swaps.

## Project Decision

Do not continue this controller line with more prompt tweaks or another full dataset run.

Recommended status:

- keep the implementation as a bounded experiment artifact
- record the result as a meaningful negative finding
- do not run 2Wiki fullscale for this controller version

## Recommended Paper Framing

This should not be written as:

- “the verifier model was weak”
- “the prompt needs more tuning”

The cleaner framing is:

> A narrowed `proposal + verifier` controller removes the global-utility burden, but a zero-shot claim layer still fails to align scaffold gaps with oracle-positive replacement actions. Even after fixing local verifier output control, the controller collapses through a mixture of false `no_unsupported_claim` decisions and false `not_supported` gain rejections.

## Commands Used

Verifier fix validation:

```bash
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy \
.venv-hipporag/bin/pytest tests/test_swap_utility_judge.py -q
```

Subset audit generation:

The subset audit was generated from a one-off inline Python runner over:

- [musique_action_swap_oracle_relaxed_20260413.json](/mnt/nvme/code/HippoRAG/run_logs/musique_action_swap_oracle_relaxed_20260413.json)
- [width_match_bridge_append_plus_ce_qatopk5_20260409smoke.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260409smoke.json)
- [width_match_bridge_append_plus_action_swap_v0_dryrun_qatopk5_20260413impl.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_action_swap_v0_dryrun_qatopk5_20260413impl.json)
- [width_match_bridge_append_plus_action_swap_v0_judge_qatopk5_20260413impl.json](/mnt/nvme/code/HippoRAG/outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_action_swap_v0_judge_qatopk5_20260413impl.json)

