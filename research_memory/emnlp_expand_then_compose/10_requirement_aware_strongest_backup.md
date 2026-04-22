# Requirement-Aware Strongest Backup Memo

Last updated: 2026-04-17

## Status

This note preserves `Requirement-Aware Strongest` as a **usable backup method line**.

Current status:
- keep the paper-facing mainline on `Expand-then-Compose`
- keep `PCRS-RAG` as the main parallel requirement branch
- keep `Requirement-Aware Strongest` as a lighter graph-side reserve option
- do not promote it to the default paper story without new evidence

Short name:
- `RAS`

Runnable approximation already in code:
- `clean GBC strongest sidecar`

## Why Keep This Line

This branch is worth preserving because it targets a different failure mode than the current simple selector line.

Main hypothesis:
- the graph retriever can already reach the relevant region
- the remaining failure is often not missing connectivity
- the failure is that the added graph-near evidence does not match the query's unmet requirement slots
- this creates a retrieval-reader gap:
  - graph-side support recall can improve
  - reader-side QA can still get worse

This makes `RAS` a coherent backup story if the current `Expand-then-Compose` selector line stops moving.

## Core Story

Recommended one-sentence framing:

> In graph-augmented multi-hop retrieval, the main bottleneck is not only connectivity deficiency. It is requirement-misaligned evidence completion: newly added graph-near passages do not necessarily satisfy the query's still-missing reasoning requirements.

Implication:
- retrieval should not simply maximize graph mass or local graph proximity
- it should preserve already-grounded anchors
- it should only reward candidates that cover unmet requirements
- it should suppress candidates that introduce sibling-relation conflicts
- it should preserve reader-compatible ordering unless coverage strictly improves

## Method Definition

The intended full method is:

1. Start from the validated `legacy_fact_graph` baseline.
2. Apply strongest-style candidate formation on a widened pool.
3. Extract a small set of query requirements:
   - anchor entity / event
   - relation or role
   - requested attribute or answer type
4. Protect top anchor passages that already satisfy grounded requirements.
5. Score frontier candidates by unmet-requirement coverage, not only by graph proximity.
6. Downweight candidates that satisfy a wrong sibling relation for the same anchor.
7. Only allow aggressive top-5 reordering when requirement coverage strictly improves.

This is the intended paper-style version.

## Current Runnable Approximation

The current codebase already contains a practical approximation of this line via the migrated strongest sidecar plus `clean GBC`.

Key code paths:
- `src/hipporag_ext/strongest/types.py`
- `src/hipporag_ext/strongest/runtime.py`
- `src/hipporag_ext/strongest/gbc.py`
- `src/hipporag_ext/strongest/shadow_entry.py`
- `scripts/eval_causal_qwen3.py`
- `tests/test_strongest_equivalence.py`
- `tests/test_strongest_shadow_regression.py`

Operational interpretation:
- `standard strongest` is the raw graph-side reorderer
- `clean GBC` is the guardrail version
- `RAS` should be treated as the next conceptual step beyond `clean GBC`

Recommended runnable config skeleton:
- selector front-end:
  - `--setwise_selector bridge_append`
  - `--setwise_pool_k 20`
  - `--expand_base_k 5`
  - `--append_max_docs 0`
  - `--append_policy bridge`
  - `--assemble_mode none`
- strongest sidecar:
  - `--strongest_shadow_enabled true`
  - `--strongest_shadow_apply_to_pool true`
  - `--strongest_candidate_k 20`
  - `--strongest_final_k 10`
  - `--strongest_hippo_head_k 10`
  - `--strongest_smoothed_union_k 10`
  - `--strongest_gamma 0.15`
- guardrail version:
  - `--strongest_rerank_mode gbc`
  - `--strongest_gbc_protected_anchor_k 2`
  - `--strongest_gbc_head_coverage_k 5`
  - `--strongest_gbc_top_passage_pool_k 20`
  - `--strongest_gbc_frontier_bonus_k 6`
  - `--strongest_gbc_bonus_weight 1.0`

## Evidence So Far

### Positive signal on 2Wiki

Report note:
- `outputs_step0_general_2wikimultihopqa/eval_reports/strongest_bridge_append_none_applypool_smoke40_20260416.md`

Summary:
- control `EM/F1 = 0.3500 / 0.4170`
- strongest apply-to-pool `EM/F1 = 0.4000 / 0.4824`
- delta `EM/F1 = +0.0500 / +0.0654`
- `Recall@5 = 0.7937 -> 0.8375`

Interpretation:
- strongest can help as a front-end pool correction layer on a structurally harder dataset
- the gain pattern is consistent with the idea that graph-side evidence exposure can matter

### Mixed signal on HotpotQA

Audit note:
- `outputs_step0_general_hotpotqa/eval_reports/strongest_gbc_e2e_audit_hotpotqa20_20260416.md`

Summary:
- baseline `EM/F1 = 0.5500 / 0.6568`
- standard strongest `EM/F1 = 0.4500 / 0.5568`
- clean GBC `EM/F1 = 0.5000 / 0.6068`

Interpretation:
- raw strongest is too aggressive
- `clean GBC` is a meaningful stabilizer
- but the line is still below baseline end-to-end on the shallow dataset

### Current diagnosis

What the two smokes jointly suggest:
- graph-side strengthening can help retrieval
- but retrieval gain does not automatically convert to QA gain
- the remaining problem is not obviously lack of graph access
- the dominant residual issue looks like requirement alignment and reader compatibility

## Why This Is A Backup, Not The Mainline

Reasons not to promote it today:
- the current best evidence is still smoke-scale
- the `HotpotQA` end-to-end result is stabilizing, not winning
- the method story is principled but not yet validated across datasets
- the paper already has a cleaner mainline in `Expand-then-Compose`
- `PCRS-RAG` remains the main explicit requirement-oriented branch

Practical position:
- preserve this line as a reserve option
- do not let it replace the current paper story by default
- reuse it when a graph-specific backup method is needed

## Relationship To PCRS-RAG

These two lines overlap in spirit but are not the same.

`PCRS-RAG`:
- changes the selector objective directly around requirement support and counterfactual leakage
- is the main requirement-first branch

`RAS`:
- keeps the graph-side strongest machinery
- adds requirement-aware completion constraints on top of it
- is lighter-weight as an engineering fallback

So `RAS` should be treated as:
- not the replacement for `PCRS-RAG`
- not the replacement for the frozen paper mainline
- a reserve graph-aware branch that can be revived quickly

## Promotion Gates

Promote this line only if all of the following look credible:

1. `HotpotQA` reaches at least baseline parity in end-to-end `EM/F1`.
2. `2Wiki` keeps a positive gain.
3. `MuSiQue` shows at least a non-destructive trend.
4. Query-level audits show fewer relation-conflict failures than `standard strongest`.
5. The added complexity can still be narrated as a clean requirement-aware completion objective.

## Minimal Next Implementation If Revisited

If this branch is reopened, the next implementation should be:

1. Add a lightweight requirement extractor on top of the strongest pool.
2. Replace pure boundary bonus with unmet-requirement-conditioned completion bonus.
3. Add sibling-relation conflict suppression.
4. Add a strict prefix-preservation gate for shallow datasets.
5. Re-run the existing `HotpotQA smoke20` audit before any larger sweep.

## Safe Claim

The strongest safe claim for this line today is:

> `clean GBC` shows that graph-side strongest needs reader-aware guarding, and the next principled step is a requirement-aware evidence completion objective rather than a more aggressive graph-only reranker.
