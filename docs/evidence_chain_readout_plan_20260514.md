# Evidence-Chain Readout Plan

**Problem**: ETv4 clean 在 MuSiQue 上没有达到预期，主要不是 graph pool 缺候选，而是 top5 readout 没有稳定选出 answer-tail gold。

**Method Thesis**: 在 query-local document graph 内，final top5 应该优先形成可闭合的 evidence chain，而不是简单采用 graph-neighbor / source-backbone ordering。

**Date**: 2026-05-14

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: Evidence-chain-aware readout can recover MuSiQue answer-tail evidence without adding dense fallback or dataset routing. | 当前 clean 的主缺口是 pool has gold but readout miss；如果 readout objective 对，MuSiQue title all-gold@5 应该上升。 | MuSiQue limit100 title all-gold@5 相比 clean +1 point or more，且 exact/title R@5 不下降。 | B1, B2 |
| C2: The improvement comes from chain closure, not from reintroducing removed tricks. | 论文不能退回 source-prior guard / repair-replacement，否则主 claim 变弱。 | 新 policy 不使用 source-prior guard、repair、replacement、gold labels、dataset switch、BM25 rerank。trace 明确记录。 | B2, B3 |
| Anti-claim: The issue can be solved by bigger candidate pool, deeper closure, more roots, or textual rerank. | 这些方向已被 limit100 probe 基本否定；继续堆会陷入局部工程最优。 | pool300 / closure-depth / BM25 / path-cover 的负结果写进 plan，并作为不再主推的排除证据。 | B0 |

## Paper Storyline

- Main paper must prove: GraphRAG E2E 的 query-local document graph 有效，且 final top5 selection 应按 evidence-chain objective 而不是弱邻接 objective。
- Appendix can support: pool size、closure depth、BM25/textual rerank、raw adjacency、multi-anchor precision 等负向/弱向 ablation。
- Experiments intentionally cut: dataset-specific hop=3 routing、gold-aware repair、LLM query schema、weighted score fusion、source-prior trick rollback。

## Current Evidence

| Finding | Evidence | Interpretation |
|---|---|---|
| MuSiQue 是主要瓶颈。 | ETv4 clean MuSiQue title all@5 0.4720，PropRAG 0.4760；2Wiki 强，HotpotQA retrieval 接近饱和。 | 优先优化 MuSiQue readout。 |
| 候选池不是唯一问题。 | Prop-only pool all@200 有 67 queries，但 Prop top5-only 中 80/99 是 ETv4 pool 和 dense 都已有 gold 却 readout miss。 | 关键问题在 top5 selection。 |
| pool300 不解决 top5。 | limit100 pool all 提升 0.89 -> 0.92，但 top5 metrics 不变。 | 不应只加 pool budget。 |
| BM25/textual rerank 有害。 | BM25 pool title all@5 0.16，clean3+BM25 0.32，均低于 clean 0.42。 | 文本显著性会抓单实体，破坏 full evidence set。 |
| path-cover 过激。 | limit100 path_cover title all@5 0.39，低于 clean 0.42。 | 不能用全局 endpoint coverage 替代稳定 reader context。 |
| 3-hop loss 机制明确。 | loss 中 tail gold drop 96.55%，weak-graph inserted 61.54%。 | 需要只针对 tail slot 选择弱邻接替换，而不是全局重排。 |

## Proposed Method: Evidence-Chain-Aware Readout

### Definition

`readout` 是 GraphRAG E2E pipeline 的 final document selection stage：

```text
query-local document graph + admitted candidate universe -> top5 reader documents
```

它不是 dense rerank、不是 reader prompt、不是 OpenIE、不是 candidate generation。

### Objective

Given clean readout order `B` and query-local graph `G`, select `S` of size `k=5`:

1. Preserve the stable evidence prefix from clean readout.
2. Re-score only tail slots using graph evidence-chain contribution.
3. Prefer candidates that strengthen closed evidence chains with already selected documents.
4. Penalize weak topic-neighbor substitutions unless they add query-grounded evidence and strong transitions.
5. Fall back to clean order for ties and for all non-improving cases.

### Closed-Form Tail Score

For a candidate `d` and current selected prefix `S`:

```text
score(d | S) =
  strong_edge_to_selected
  + relation/query-token support
  + new query endpoint / token coverage
  + variable-flow or role-transition support
  - weak-only connectivity
  - disruption from clean order
```

Implementation should be lexicographic, not weighted score fusion:

```text
1. candidate has any fact-witnessed edge to selected
2. candidate has sentence_grounded_transition / role_bridge to selected
3. candidate adds uncovered query endpoint
4. candidate adds uncovered query token
5. candidate is not weak-only source_endpoint_incidence
6. candidate has lower clean/source-prior rank
7. deterministic doc id tie-break
```

### Critical Constraint

This is not a fallback:

- No `dataset == musique`.
- No `hop_count == 3`.
- No gold labels.
- No dense rank threshold as final guard.
- No BM25/textual rerank.
- No source-prior guard / repair / replacement.
- No LLM-generated query schema.

## Experiment Blocks

### Block 0: Negative Evidence Freeze

- Claim tested: prior fixes are not the right direction.
- Why this block exists: avoid repeating pool/depth/textual/path-cover loops.
- Dataset / split / task: MuSiQue limit100.
- Compared systems: clean, pool300, closure-depth, BM25, path-cover.
- Metrics: exact/title R@5, exact/title all-gold@5, top5 diff, gain/loss.
- Success criterion: documented as frozen negative evidence.
- Failure interpretation: if future code changes invalidate these numbers, rerun before claiming them.
- Priority: MUST-RUN documentation only.

### Block 1: Evidence-Chain Readout Sanity

- Claim tested: tail chain scoring changes the intended docs without destabilizing the prefix.
- Why this block exists: catch no-op or over-aggressive readout before full run.
- Dataset / split / task: MuSiQue limit100, Qwen32B clean index, candidate_pool_k=200, top_k=5.
- Compared systems: clean_mainline vs evidence_chain.
- Metrics: exact/title R@5, exact/title all-gold@5, top5 changed, gain/loss, prefix preservation rate, inserted edge strength distribution.
- Setup details: same index, same candidate generator, no reader needed for first gate.
- Success criterion: title all@5 >= clean +0.01 and exact/title R@5 not below clean.
- Failure interpretation: objective still misses answer-tail gold or is too conservative.
- Table / figure target: internal gate table; paper appendix only if positive.
- Priority: MUST-RUN.

### Block 2: Mechanism Audit

- Claim tested: gains come from evidence-chain closure, not weak-neighbor substitution.
- Why this block exists: reviewer will ask why this is not another trick.
- Dataset / split / task: MuSiQue limit100 changed queries.
- Compared systems: clean top5 vs evidence_chain top5.
- Metrics: inserted docs with fact-witnessed edge to selected, strong edge ratio, weak-only ratio, tail-gold recovery/loss.
- Setup details: reuse graph_payload trace and gold titles only for analysis, not selection.
- Success criterion: recovered docs have higher strong-edge support and lower weak-only ratio than clean loss insertions.
- Failure interpretation: if weak-only ratio remains high, objective is not evidence-chain-aware enough.
- Priority: MUST-RUN if Block 1 positive or neutral.

### Block 3: Full1000 and Cross-Dataset Safety

- Claim tested: MuSiQue gain generalizes and does not damage 2Wiki/HotpotQA.
- Why this block exists: a method change cannot be justified on limit100 only.
- Dataset / split / task: 2Wiki, MuSiQue, HotpotQA full1000 retrieval; reader only after retrieval gate.
- Compared systems: clean_mainline, evidence_chain, PropRAG/HippoRAG same protocol table.
- Metrics: R@5/R@20/R@100/R@200, all-gold@5, title-level equivalents, EM/F1 after GPT-4o-mini reader.
- Success criterion: MuSiQue title all@5 improves >= +0.01 full1000; 2Wiki/HotpotQA no material drop; EM/F1 non-negative or explainable.
- Failure interpretation: keep as diagnostic, do not rename main method.
- Priority: RUN ONLY AFTER Block 1 gate.

## Implementation Plan

| Step | Change | File | Gate |
|---|---|---|---|
| I1 | Add plan/tracker docs. | `docs/evidence_chain_readout_plan_20260514.md`, `run_logs/evidence_chain_readout_tracker_20260514.md` | Files exist and define stop/go gates. |
| I2 | Add readout policy CLI option. | `evidence_transition_graphragv4_fact_witnessed_sto/run_fresh_e2e.py` | `--readout-policy evidence_chain` accepted. |
| I3 | Add clean-mainline ablation branch. | `source_authorized_vocab_strict_retrieval/e2e_pipeline.py` | Trace says `evidence_chain_readout`; clean default unchanged. |
| I4 | Add `order_local_sto_evidence_chain_readout`. | `agsto/local_graph.py` | Function is deterministic, no dataset/gold/LLM use. |
| I5 | Run syntax/import sanity. | Python compile or smoke import. | No syntax/import errors. |
| I6 | Run MuSiQue limit100 retrieval. | run_logs output root | Metrics generated and compared to clean. |

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | Freeze plan and implement policy. | local patch + syntax check | no clean default change | < 30 min | dirty worktree conflicts |
| M1 | MuSiQue limit100 retrieval gate. | `evidence_chain`, pool200, top5 | title all@5 >= clean +0.01 and R@5 no drop | ~15-60 min depending cache | policy no-op or hurts |
| M2 | Mechanism audit. | changed-query diagnostic | strong-edge insertions up, weak-only down | ~30 min | trace lacks needed fields |
| M3 | Full1000 retrieval. | MuSiQue first, then 2Wiki/HotpotQA | full MuSiQue positive, others safe | hours | limit100 overfits |
| M4 | Reader QA. | GPT-4o-mini top5 reader | EM/F1 non-negative or retrieval gain clearly separated | hours/API | reader noise masks retrieval |

## Decision Gates

- Gate A: If MuSiQue limit100 title all@5 drops below clean, stop and audit selected docs; do not run full1000.
- Gate B: If title all@5 unchanged but weak-only substitution decreases, run a second conservative variant only after documenting why.
- Gate C: If title all@5 improves but R@5 drops, reject objective as too all-gold-biased.
- Gate D: If full1000 MuSiQue improves but HotpotQA drops, isolate HotpotQA reader/evidence-ordering separately; do not make evidence_chain default.

## Final Checklist

- [ ] Mainline clean result remains reproducible.
- [ ] New policy is opt-in via `--readout-policy evidence_chain`.
- [ ] No source-prior guard, repair, replacement, dataset routing, BM25 rerank, gold labels, or LLM query schema.
- [ ] MuSiQue limit100 gate result recorded.
- [ ] Mechanism audit result recorded before any paper claim.
