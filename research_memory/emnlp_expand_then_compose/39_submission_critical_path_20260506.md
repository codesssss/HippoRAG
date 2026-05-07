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
- call the SetR comparison "SetR-style IRI selector (local) + rank-order
  fallback", never "official SetR reproduction";
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

SetR-style full1000 is already complete and should be treated as the strongest
LLM set-selection baseline, not as a future experiment.

| Order | Experiment | Purpose | Status | Artifacts |
|---:|---|---|---|---|
| 0 | SetR-style IRI full1000 | Answers the strongest "why not LLM set selection?" attack | Done | `reports/setr_full1000_20260503/`, `run_logs/setr_full1000_20260503/` |
| 0.5 | PropRAG nobinding full1000 refresh | Tests whether DAEC's binding dimension is the load-bearing difference from SetR-style flat IRI selection | Running / launched after reviewer queue | `run_logs/daec_nobinding_proprag_full1000_20260506/` |
| 1 | IRCoT-style (local) full1000 | Answers "why not iterative retrieval?" | Done | `run_logs/ircot_style_full1000_20260506/` |
| 2 | LLM-direct-select PropRAG full1000 | Simpler title/snippet LLM selector control | Done | `run_logs/llm_direct_select_proprag_full1000_20260506/` |

Launcher:

- `run_logs/launch_reviewer_baselines_full1000_20260506.sh`
- tmux session: `reviewer_baselines_full1000_20260506`

SetR-style canonical memo:

- `research_memory/emnlp_expand_then_compose/40_setr_style_full1000_positioning_20260506.md`

Protocol constraints:

- Use the same 1000-query subset, reader, final `qa_top_k=5`, Qwen3-8B
  endpoints, and NV-Embed endpoint as DAEC full1000.
- Do not report mixed limit100/full1000 rows in the main table.
- SetR-style must be labeled as local prompt-based IRI selection with rank-order
  fallback. Its prompt can select fewer than five documents; the adapter fills
  remaining reader slots from original rank order.
- Selection-depth audit shows SetR-style is shallow on 2Wiki/HotpotQA
  (80-86% of selected positions come from the original top5) and deeper on
  MuSiQue (57% from original top5). Use this as interpretation, not as a claim
  that the implementation is buggy.
- R@5 must be sourced from one named metric field per table. Do not mix
  `overall_recomputed`, DAEC selector metrics, source-pool payload recall, and
  ad-hoc title recomputations.
- If LLM-direct `snippet128` is strong or wins on answer metrics, preserve the
  story by reporting cost: DAEC is then a cost-efficient structured composer,
  not a claim that embedding/noisy-OR always beats LLM judgment.
- Do not claim DAEC generally beats LLM set selection. Existing SetR-style
  full1000 wins PropRAG HotpotQA/MuSiQue on F1 and support R@5.

### P0.5. Full1000 Consolidation: Do Immediately After P0

All reviewer-baseline full1000 JSON files have landed. Build these tables from full1000 only.

| Task | Required comparisons | Output |
|---|---|---|
| Main comparison | Top5, DAEC-LLM+CTL, SetR-style IRI, IRCoT-style (local), LLM-direct-title, LLM-direct-snippet128 | EM, F1, Support R@5 |
| Paired uncertainty | DAEC vs Top5, DAEC vs SetR-style IRI, DAEC vs IRCoT-style, DAEC vs both LLM-direct variants | paired bootstrap CI for EM/F1/Support R@5 deltas |
| Cost/calls | DAEC-LLM+CTL, SetR-style IRI, IRCoT-style, LLM-direct variants | selector calls/q, retrieval calls/q, latency/q, token usage where available |
| Protocol caveats | SetR-style, IRCoT-style, and LLM-direct | explicit caveat text for local protocol, fallback behavior, and cost asymmetry |

Canonical DAEC full1000 cost seed:

- `reports/daec_cost_review_20260503/cost_review_draft.md`

Reviewer-baseline cost/calls table:

- script: `scripts/analyze_reviewer_baseline_costs.py`
- report: `reports/reviewer_baseline_costs_20260506/cost_table.md`
- CSV: `reports/reviewer_baseline_costs_20260506/cost_table.csv`

Reviewer-baseline paired CI:

- script: `scripts/analyze_reviewer_baseline_paired_ci.py`
- report: `reports/reviewer_baseline_ci_20260506/paired_ci.md`
- CSV: `reports/reviewer_baseline_ci_20260506/paired_ci.csv`
- hard-slice script: `scripts/analyze_reviewer_baseline_hard_slices.py`
- hard-slice report:
  `reports/reviewer_baseline_ci_20260506/hard_slice_gold_doc_count.md`

Main F1 CI result:

| Dataset | Comparison | dF1 | Paired 95% CI | Status |
|---|---|---:|---:|---|
| 2Wiki | DAEC-selective - Top5 | +0.0661 | [+0.0444, +0.0883] | significant |
| 2Wiki | DAEC-selective - SetR-style k20 | +0.0182 | [-0.0016, +0.0378] | mixed / not significant |
| 2Wiki | DAEC-selective - IRCoT-style | +0.0825 | [+0.0572, +0.1083] | significant |
| HotpotQA | DAEC-selective - Top5 | +0.0246 | [+0.0110, +0.0381] | significant |
| HotpotQA | DAEC-selective - SetR-style k20 | -0.0079 | [-0.0222, +0.0065] | mixed / not significant |
| HotpotQA | DAEC-selective - IRCoT-style | +0.0394 | [+0.0205, +0.0583] | significant |
| MuSiQue | DAEC-selective - Top5 | +0.0282 | [+0.0076, +0.0488] | significant |
| MuSiQue | DAEC-selective - SetR-style k20 | -0.0212 | [-0.0445, +0.0019] | mixed / not significant |
| MuSiQue | DAEC-selective - IRCoT-style | +0.0294 | [+0.0028, +0.0563] | significant |

Gold-support-count hard slice, DAEC-selective vs SetR-style k20:

| Dataset | Slice | N | dF1 | F1 95% CI | dSupport R@5 | R@5 95% CI | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| 2Wiki | `gold_doc_count>=3` (= 4-doc subset) | 235 | +0.0454 | [+0.0014, +0.0894] | +0.0106 | [-0.0106, +0.0319] | DAEC-selective wins answer F1 on the deep subset; support is tied. |
| HotpotQA | `gold_doc_count>=3` | 0 | -- | -- | -- | -- | No deep-support slice; treat as shallow/saturated contrast. |
| MuSiQue | `gold_doc_count>=3` | 482 | -0.0083 | [-0.0431, +0.0260] | -0.0003 | [-0.0201, +0.0197] | Tied; SetR-style is not overturned on ambiguous long-chain questions. |

Additional slice note: on MuSiQue `gold_doc_count=2`, SetR-style is
significantly higher on answer F1 (`dF1=-0.0333`, 95% CI
`[-0.0639, -0.0031]`). The hard-slice result should therefore be framed as a
2Wiki deep-composition win plus MuSiQue tie, not as a universal SetR defeat.

Current cost interpretation:

- DAEC is not lower-call than one-shot LLM selectors: cold logical selector
  cost is one decomposition call plus about `6-7` binding calls/query.
- Normalized prompt-budget comparison: `prompt_tokens_per_q` is already a
  per-query aggregate over recorded binding calls, so do not multiply it by
  call count. DAEC binding extraction records `~1.0k-1.25k` prompt tokens/query
  but `7-8` logical calls/query plus an uninstrumented decomposition call;
  SetR-style k20 uses one selector call and `~1.9k-2.8k` estimated prompt
  tokens/query (`prompt_chars/4`). DAEC is not cheaper in calls/latency than
  SetR-style; the value claim is structure/control/auditability.
- DAEC still uses no extra retrieval over the fixed PropRAG pool, while
  IRCoT-style uses `3` follow-up retrieval calls/query.
- DAEC-selective is a robustness patch, not a cost patch: cold-equivalent cost
  equals DAEC because the gate runs after binding extraction.
- Token comparisons need caveats: DAEC token fields cover binding extraction
  only; SetR historical rows use reconstructed `chars/4` token estimates rather
  than API-reported token usage.
- Unified title-multiset support R@5 CI has been regenerated in
  `reports/reviewer_baseline_ci_20260506/paired_ci.md`; use that table for
  paper-facing support uncertainty and avoid mixing stored aggregate fields.

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
SetR-style already covers flat information-requirement selection; current DAEC
needs a same-protocol binding ablation to show what the extra binding dimension
buys.
```

Minimum refresh:

| Dimension | Value |
|---|---|
| Pool | PropRAG pool100 |
| Datasets | 2Wiki, HotpotQA, MuSiQue |
| Limit | 1000 |
| Variant | current DAEC-LLM+CTL with DAEC binding mode disabled via `daec_noisyor_nobind` |
| Baseline | current DAEC-LLM+CTL full1000 from `run_logs/daec_llm_wiki_title_proprag_full1000_20260503/` |
| Launcher | `run_logs/launch_daec_nobinding_proprag_full1000_20260506.sh` |
| Output | `run_logs/daec_nobinding_proprag_full1000_20260506/` |

Launch status:

| Field | Value |
|---|---|
| Status | Done |
| Start time | 2026-05-06T18:59:11+08:00 |
| End time | 2026-05-06T19:33:42+08:00 |
| tmux session | `daec_nobinding_proprag_full1000_20260506` |
| Ports | `2wikimultihopqa:8041`, `hotpotqa:8042`, `musique:8043` |
| Dataset status files | `run_logs/daec_nobinding_proprag_full1000_20260506/*_full1000.status` |

Results:

| Dataset | DAEC EM/F1/R@5 | Nobinding EM/F1/R@5 | dF1 | Interpretation |
|---|---:|---:|---:|---|
| 2Wiki | 0.642 / 0.7118 / 0.9410 | 0.548 / 0.6120 / 0.8610 | -0.0998 | Binding is strongly load-bearing. |
| HotpotQA | 0.620 / 0.7473 / 0.9605 | 0.616 / 0.7387 / 0.9545 | -0.0086 | Shallow/saturated; binding effect is small. |
| MuSiQue | 0.337 / 0.4359 / 0.7269 | 0.343 / 0.4458 / 0.7297 | +0.0099 | Boundary case; binding is not uniformly beneficial. |

Expected cost:

- should be cheaper than full DAEC-LLM+CTL because no LLM binding extraction is
  needed;
- run only after P0 baselines finish to avoid starving reviewer baselines.

Paper use:

- Move this ablation into the main paper, but phrase it carefully:
  binding is load-bearing on the deep 2Wiki setting and weak/boundary on
  HotpotQA/MuSiQue.
- Use the 2Wiki result to distinguish DAEC from SetR-style flat IRI selection;
  do not claim binding is universally positive.

### P0.5. Selective Binding Consolidation

Status: done.

Artifacts:

- script: `scripts/analyze_daec_selective_binding_phase1.py`
- report: `reports/daec_selective_binding_phase1_20260506/phase1_report.md`
- paired bootstrap CI:
  `reports/daec_selective_binding_phase1_20260506/phase1_paired_bootstrap_ci.csv`
- sanity audit:
  `reports/daec_selective_binding_phase1_20260506/phase1_sanity_audit.md`
- result registry:
  `research_memory/emnlp_expand_then_compose/04_result_registry.md`

Frozen rule:

```text
if bind_conf_title_unique >= 0.88:
    use DAEC binding
else:
    abstain to nobinding
```

Main result:

| Dataset | DAEC F1 | DAEC-selective F1 | dF1 | Paired 95% CI |
|---|---:|---:|---:|---:|
| 2Wiki | 0.7118 | 0.7118 | +0.0000 | [0.0000, 0.0000] |
| HotpotQA | 0.7473 | 0.7473 | +0.0000 | [0.0000, 0.0000] |
| MuSiQue | 0.4359 | 0.4548 | +0.0189 | [+0.0076, +0.0309] |

Interpretation:

- The title-uniqueness gate passes the fresh-vs-offline consistency check:
  all fresh F1 deviations are within the pre-set ±0.005 gate.
- It repairs MuSiQue over-constrained binding without hurting 2Wiki/HotpotQA.
- The exact zero delta on 2Wiki/HotpotQA is explained by selection equality:
  the gate does abstain (`null_rate=0.302/0.343`), but all abstained queries
  have identical DAEC and Nobind top-5 titles, so the reader input is unchanged.
- Phase-1 reused the base DAEC binding-extraction cache because the selective
  method differs only by a post-binding query-level abstention gate. This
  isolates the gate from LLM extraction stochasticity; selector decisions and
  reader outputs were produced by the selective run and audited against base
  reports.
- On MuSiQue, the gate abstains on `62.9%` of queries, but DAEC-selective does
  not collapse to Nobind: Nobind F1 is `0.4458`, while DAEC-selective reaches
  `0.4548`. The extra `+0.0090` over Nobind comes from the `37.1%` of queries
  where binding is retained.
- It does not justify a claim that DAEC beats SetR-style generally:
  full-dataset DAEC-selective vs SetR-style F1 CIs cross zero on all three
  datasets, although the pre-specified 2Wiki 4-doc slice is a positive
  significant DAEC-selective case.
- Use as an abstention-aware robustness patch if method space allows; otherwise
  report as a calibrated ablation/extension.

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
| New SetR-style prompt implementation | Do not run. Existing full1000 SetR-style IRI baseline is complete. |
| Official SetR reproduction | Do not run before submission blocking items. The current controlled local baseline is enough for the "why not LLM set selection?" attack; official reproduction adds checkpoint/data/prompt confounds. |
| Official IRCoT reproduction | Do not run for the current DAEC claim. It answers a different question and introduces Elasticsearch/data/prompt/HP confounds. |
| More binding posterior / structural verifier variants | Stop. Existing posterior/softcompat/verifier probes are negative and should be limitation/future work. |
| Graph-prior expansion beyond pilot | Stop unless it becomes a paper section; current signal is weak and not central. |
| Full LLM-direct over Dense/HippoRAG pools | Do not run before PropRAG full1000 interpretation. |

## Narrative Gates

Use these gates when full1000 results arrive.

| Outcome | Paper response |
|---|---|
| DAEC beats SetR-style, IRCoT-style, and LLM-direct on F1/Support R@5 | Main story remains demand-aware fixed-pool composition. |
| SetR-style beats DAEC on some datasets | Do not overclaim objective superiority. Reframe around explicit dependency binding, interpretability, cost, and dataset-dependent tradeoffs. |
| LLM-direct-snippet128 beats DAEC on EM/F1 but costs much more | Reframe DAEC as a cost-efficient structured alternative with no per-query top100 LLM reading. |
| IRCoT-style beats DAEC | Shrink claim: DAEC is a fixed-pool composer, not a replacement for iterative retrieval; move IRCoT-style to main comparison and analyze where iteration helps. |
| Current-version nobinding hurts DAEC | Binding is main method novelty; include ablation in main paper. |
| Current-version nobinding is flat | Treat binding contribution as dataset/pool-dependent; use old cross-pool nobinding only as supporting diagnostic. |
| Selective-binding Phase-0 passes offline gate | Optional P1 method improvement: implement a frozen title-uniqueness abstention router and validate with a fresh reader run. |

## Active Run Snapshot

As of 2026-05-06 18:45 CST:

- `SetR-style IRI full1000` is complete:
  `reports/setr_full1000_20260503/`.
- `IRCoT-style (local) full1000` is complete:
  `run_logs/ircot_style_full1000_20260506/`.
- `LLM-direct-select PropRAG full1000` is complete:
  `run_logs/llm_direct_select_proprag_full1000_20260506/`.
- `PropRAG nobinding full1000 refresh` is the next active run:
  `run_logs/daec_nobinding_proprag_full1000_20260506/`.

Reviewer baseline full1000 takeaway:

| Dataset | DAEC F1 | IRCoT-style F1 | LLM-direct-title F1 | LLM-direct-snippet128 F1 |
|---|---:|---:|---:|---:|
| 2Wiki | 0.7118 | 0.6293 | 0.5697 | 0.6535 |
| HotpotQA | 0.7473 | 0.7079 | 0.6478 | 0.6544 |
| MuSiQue | 0.4359 | 0.4254 | 0.3582 | 0.3850 |

Neither IRCoT-style nor LLM-direct beats DAEC under the controlled full1000 protocol. SetR-style remains the stronger LLM set-selection baseline and should stay in the main table.

Selective binding Phase-0 is complete:

- script: `scripts/analyze_daec_selective_binding_phase0.py`
- report: `reports/daec_selective_binding_phase0_20260506/phase0_report.md`
- verdict: `go_phase1`
- robust exploratory rule: use DAEC binding when `bind_conf_title_unique >= 0.88`,
  otherwise abstain to nobinding.
- offline all-split simulation:
  - 2Wiki: DAEC `0.7118` F1 -> selective `0.7090` F1 (`-0.0029`).
  - HotpotQA: DAEC `0.7473` F1 -> selective `0.7493` F1 (`+0.0020`).
  - MuSiQue: DAEC `0.4359` F1 -> selective `0.4553` F1 (`+0.0194`).
- Treat this as exploratory separability evidence, not final method evidence;
  any implementation must freeze the rule before a fresh reader run.

Selective binding Phase-1 fresh run is complete:

- implementation: `scripts/dtc_embed_utils.py`, `scripts/eval_causal_qwen3.py`
- launcher: `run_logs/launch_daec_selective_titleuniq_proprag_full1000_20260506.sh`
- results: `run_logs/daec_selective_titleuniq_proprag_full1000_20260506/`
- report: `reports/daec_selective_binding_phase1_20260506/phase1_report.md`
- frozen rule: `bind_conf_title_unique >= 0.88` -> bind, else abstain to nobinding.

Fresh full1000 result:

| Dataset | DAEC F1 | Nobind F1 | DAEC-selective F1 | dF1 vs DAEC | Phase0 Predicted F1 | Fresh-Phase0 |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.7118 | 0.6120 | 0.7118 | +0.0000 | 0.7090 | +0.0028 |
| HotpotQA | 0.7473 | 0.7387 | 0.7473 | +0.0000 | 0.7493 | -0.0020 |
| MuSiQue | 0.4359 | 0.4458 | 0.4548 | +0.0189 | 0.4553 | -0.0005 |

Interpretation:

- Phase-1 passes the ±0.005 fresh-vs-Phase0 consistency gate.
- Selective binding is useful as an abstention-aware robustness patch: it preserves 2Wiki/HotpotQA and fixes much of MuSiQue's binding over-constraint.
- It should not be framed as a universal binding verifier. The mechanism is a confidence gate over binding extraction title uniqueness.
- For main paper positioning, use this as the final DAEC variant only if the method section can afford the extra rule. Otherwise report it as a calibrated extension/ablation showing that ambiguous binding should abstain.

Selective dense/hipporag cross-pool refresh launched:

- launcher:
  `run_logs/launch_daec_selective_titleuniq_dense_hipporag_full1000_20260506.sh`
- output:
  `run_logs/daec_selective_titleuniq_dense_hipporag_full1000_20260506/`
- tmux session:
  `daec_selective_titleuniq_dense_hipporag_full1000_20260506`
- status: launched 2026-05-06T23:20:09+08:00.
- protocol: same frozen `bind_conf_title_unique >= 0.88` gate, dense and
  HippoRAG pool100, full1000, `wiki_title`, `qa_top_k=5`,
  `qa_doc_max_chars=2048`.
- cache policy: seeded from the 20260503 dense/hipporag DAEC-LLM+CTL binding
  caches, since selective differs only by the post-binding abstention gate.
- parallelism: three dataset workers run against `8041/8042/8043`; each worker
  runs dense first, then HippoRAG.
