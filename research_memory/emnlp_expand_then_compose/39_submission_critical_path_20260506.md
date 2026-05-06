# DAEC Submission Critical Path - 2026-05-06

This note consolidates the current paper-facing experiment order for the
`DAEC-LLM + CTL` line. It supersedes the older broad TODO priority while the
submission blocker experiments are running.

## Positioning

Primary paper object:

```text
DAEC is the method. Expand-then-Compose is the problem framing.
```

Safe wording:

- use "no in-domain supervision" or "no task-specific fine-tuning";
- avoid using "train-free" as an unqualified slogan because DAEC still uses an
  LLM for decomposition/entity extraction;
- call the iterative baseline "IRCoT-style (local)", never "official IRCoT
  reproduction";
- keep limit100 results as pilot/appendix evidence; use full1000 numbers in
  main tables.

Do not claim:

- DAEC beats official IRCoT;
- Expand itself is a method contribution;
- graph/binding verifier variants are positive methods;
- DAEC preserves PropRAG's original online LLM-free profile after adding
  DAEC-LLM + CTL.

## Critical Path Order

### P0. Reviewer Baselines: Run First

Run and finish these before opening any additional heavy experiment.

| Order | Experiment | Purpose | Status | Artifacts |
|---:|---|---|---|---|
| 1 | IRCoT-style (local) full1000 | Answers "why not iterative retrieval?" | Running | `run_logs/ircot_style_full1000_20260506/` |
| 2 | LLM-direct-select PropRAG full1000 | Answers "why not let the LLM pick top-5?" | Queued after IRCoT-style | `run_logs/llm_direct_select_proprag_full1000_20260506/` |

Launcher:

- `run_logs/launch_reviewer_baselines_full1000_20260506.sh`
- tmux session: `reviewer_baselines_full1000_20260506`

Protocol constraints:

- Use the same 1000-query subset, reader, final `qa_top_k=5`, Qwen3-8B
  endpoints, and NV-Embed endpoint as DAEC full1000.
- Do not report mixed limit100/full1000 rows in the main table.
- If LLM-direct `snippet128` is strong or wins on answer metrics, preserve the
  story by reporting cost: DAEC is then a cost-efficient structured composer,
  not a claim that embedding/noisy-OR always beats LLM judgment.

### P0.5. Full1000 Consolidation: Do Immediately After P0

Once all full1000 JSON files land, build these tables from full1000 only.

| Task | Required comparisons | Output |
|---|---|---|
| Main comparison | Top5, DAEC-LLM+CTL, IRCoT-style (local), LLM-direct-title, LLM-direct-snippet128 | EM, F1, Support R@5 |
| Paired uncertainty | DAEC vs Top5, DAEC vs IRCoT-style, DAEC vs both LLM-direct variants | paired bootstrap CI for EM/F1/Support R@5 deltas |
| Cost/calls | DAEC-LLM+CTL, IRCoT-style, LLM-direct variants | selector calls/q, retrieval calls/q, latency/q, token usage where available |
| Protocol caveats | IRCoT-style and LLM-direct | explicit caveat text for local protocol and cost asymmetry |

Canonical DAEC full1000 cost seed:

- `reports/daec_cost_review_20260503/cost_review_draft.md`

Required main-table rules:

- Main table: full1000 only.
- Limit100: appendix/pilot/diagnostic only.
- Use "Support R@5" for evidence recall to avoid conflict with raw retrieval
  metric names stored in JSON.

### P0.5. Current-Version Nobinding Refresh

Run this after P0 baselines finish and before freezing paper ablations.

Reason:

```text
Dependency binding is the clearest method novelty, but old Layer-1 nobinding
ablations do not exactly match the 20260503 DAEC-LLM+CTL protocol.
```

Minimum refresh:

| Dimension | Value |
|---|---|
| Pool | PropRAG pool100 |
| Datasets | 2Wiki, HotpotQA, MuSiQue |
| Limit | 1000 |
| Variant | current DAEC-LLM+CTL with dependency binding disabled |
| Baseline | current DAEC-LLM+CTL full1000 from `run_logs/daec_llm_wiki_title_proprag_full1000_20260503/` |

Expected cost:

- should be cheaper than full DAEC-LLM+CTL because no LLM binding extraction is
  needed;
- run only after P0 baselines finish to avoid starving reviewer baselines.

Paper use:

- If negative, it supports binding as load-bearing and moves to main ablation.
- If small/noisy, keep old broader Layer-1 nobinding as historical support and
  report the current refresh honestly as a boundary.

## P1. Useful But Not Blocking

These should not interrupt P0/P0.5.

| Experiment | Reason to defer |
|---|---|
| Answer-spotting / leakage controls | Important, but LLM-direct-select already addresses the strongest "LLM can just pick" objection more directly. Run only after P0/P0.5 if time remains. |
| Demand-only extraction / doc-only extraction | Useful appendix controls, not necessary before reviewer baselines and same-version binding ablation. |
| Additional retriever pools for IRCoT-style or LLM-direct | Expands the matrix too much; run only if PropRAG full1000 leaves an ambiguity. |
| Bridge/router experiments | Separate story; do not mix with the DAEC main method table. |

## P2. Do Not Run Now

| Experiment | Decision |
|---|---|
| Official IRCoT reproduction | Do not run for the current DAEC claim. It answers a different question and introduces Elasticsearch/data/prompt/HP confounds. |
| More binding posterior / structural verifier variants | Stop. Existing posterior/softcompat/verifier probes are negative and should be limitation/future work. |
| Graph-prior expansion beyond pilot | Stop unless it becomes a paper section; current signal is weak and not central. |
| Full LLM-direct over Dense/HippoRAG pools | Do not run before PropRAG full1000 interpretation. |

## Narrative Gates

Use these gates when full1000 results arrive.

| Outcome | Paper response |
|---|---|
| DAEC beats IRCoT-style and LLM-direct on F1/Support R@5 | Main story remains demand-aware fixed-pool composition. |
| LLM-direct-snippet128 beats DAEC on EM/F1 but costs much more | Reframe DAEC as a cost-efficient structured alternative with no per-query top100 LLM reading. |
| IRCoT-style beats DAEC | Shrink claim: DAEC is a fixed-pool composer, not a replacement for iterative retrieval; move IRCoT-style to main comparison and analyze where iteration helps. |
| Current-version nobinding hurts DAEC | Binding is main method novelty; include ablation in main paper. |
| Current-version nobinding is flat | Treat binding contribution as dataset/pool-dependent; use old cross-pool nobinding only as supporting diagnostic. |

## Active Run Snapshot

As of 2026-05-06:

- `IRCoT-style (local) full1000` is running in
  `reviewer_baselines_full1000_20260506`.
- `LLM-direct-select PropRAG full1000` is queued in the same launcher and starts
  only after IRCoT-style exits successfully.
- Do not start the nobinding refresh until that queue completes.

