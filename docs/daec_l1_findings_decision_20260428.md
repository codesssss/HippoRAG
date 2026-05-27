# DAEC 2026-04-28 Decision Memo: Archive DAPG, Reframe Around DAEC-L1

Date: 2026-04-28

Status: current execution decision after Phase 2.5 mechanism gates and DAEC-L1 full1000 gain diagnostics.

This memo supersedes `docs/daec_dapg_proposal_b_demand_lifted_absorption_20260428.md`
for execution decisions. The old DAPG proposal is retained as a negative-ablation
record, not as the active method plan.

Related exploratory proposal:

- `docs/arec_rag_proposal_c_train_free_evidence_closure_20260428.md` proposes
  AREC-RAG, a train-free closed-loop evidence closure route. Its MuSiQue
  limit=100 smoke is now a red-light result: silver obligations show mostly a
  projection/selector signal, while generated obligations degrade SC@5 and
  recall. It is not promoted to the active execution plan.
- `docs/beep_rag_proposal_d_bridge_edge_evidence_path_20260428.md` proposes
  BEEP-RAG, a corpus-grounded bridge-edge evidence-chain route. Its Day -1
  gates are now red: BridgeRAG overlap risk is high, and MuSiQue 4-hop manual
  audit failed the entity-bridge ceiling gate. It is not promoted to smoke.

## Executive Decision

Stop the DAEC-DAPG / demand-lifted absorption line as a main method.

Do not continue with:

- full-corpus DAPG / Phase 3 graph retrieval;
- phi calibration sweeps;
- binding-conditioned transition masks;
- new channel-routing variants;
- delayed noisy-OR aggregation as a main contribution.
- BEEP entity-only bridge-edge implementation or smoke runs.

The only currently defensible positive line is DAEC-L1 as a train-free
fixed-pool evidence projection method, with a narrower Findings/short-paper
framing:

```text
Strong retriever top-100 pool
  -> demand-aware fixed-pool projection
  -> reader top-k context with fewer distractors and more complete support
```

The graph / absorption results should be written as controlled negative
ablations and methodological evidence, not as the proposed method.

## What The Gates Showed

### Gate 1: DAEC-L1 Has A Real Positive Signal On 2Wiki

DAEC-L1 full1000 gain diagnostics:

| Setting | Baseline EM/F1 | DAEC-L1 EM/F1 | Delta EM/F1 | Wrong->Correct | Correct->Wrong | Support Recall Delta | Support Complete Delta | Noise Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki dense pool100 | 0.4550 / 0.4991 | 0.4980 / 0.5541 | +0.0430 / +0.0551 | 95 | 52 | +0.0617 | +0.1150 | -0.0292 |
| 2Wiki PropRAG pool100 | 0.5790 / 0.6503 | 0.6130 / 0.6828 | +0.0340 / +0.0325 | 68 | 34 | +0.0187 | +0.0380 | -0.0071 |
| MuSiQue dense pool100 | 0.2980 / 0.3896 | 0.2900 / 0.3818 | -0.0080 / -0.0078 | 57 | 65 | -0.0034 | -0.0140 | +0.0237 |

Interpretation:

- On 2Wiki, DAEC-L1 gains come from actual context changes, not reader noise.
- Wrong-to-correct transitions exceed correct-to-wrong transitions.
- Improved-F1 buckets add gold evidence, improve support completeness, and reduce noise.
- On MuSiQue, DAEC-L1 is not stable: correct-to-wrong exceeds wrong-to-correct and support/noise deltas go in the wrong direction.

Main positive claim that remains viable:

```text
In bridge-style QA pools such as 2Wiki, a train-free demand-aware fixed-pool
projection can replace distractors with missing support documents and improve
reader EM/F1.
```

Claims that are not supported:

```text
DAEC-L1 is a universal multi-hop retriever.
DAEC-L1 solves MuSiQue long-chain composition.
DAEC-L1 is an end-to-end graph retrieval method.
```

### Gate 2: Dense Query Is The Retrieval Ceiling In The Local Harness

Full1000 Phase 2.5 dense-only hard gate:

| Dataset / Hop | Dense Query Cosine | Query Absorption | Dense Demand Union | Union Absorption |
| --- | ---: | ---: | ---: | ---: |
| 2Wiki 2-hop | 0.7771 / 0.5542 | 0.7346 / 0.4810 | 0.7235 / 0.4523 | 0.7000 / 0.4170 |
| 2Wiki 4-hop | 0.5500 / 0.0213 | 0.5330 / 0.0213 | 0.5436 / 0.0213 | 0.5277 / 0.0128 |
| MuSiQue 2-hop | 0.7751 / 0.5695 | 0.7336 / 0.4942 | 0.6950 / 0.4189 | 0.6737 / 0.3764 |
| MuSiQue 3-hop | 0.6772 / 0.2658 | 0.6456 / 0.2373 | 0.6350 / 0.1741 | 0.6097 / 0.1804 |
| MuSiQue 4-hop | 0.4418 / 0.0422 | 0.3991 / 0.0301 | 0.4332 / 0.0301 | 0.4061 / 0.0241 |

Each cell is support recall / support complete at top-5.

Interpretation:

- Direct NV-Embed question cosine is strongest in every full1000 bucket.
- Query-level absorption is consistently worse than direct query cosine.
- Demand-union absorption is consistently worse than dense demand-union cosine.
- Demand union does not beat raw question embedding.
- Local graph absorption is not a positive retrieval operator in the current substrate.

This rules out `Demand-Union Absorption` as a main method name or claim.

### Gate 3: Multi-Channel Delayed Aggregation Failed

Across Phase 2.5 mechanism tests:

- `multi_channel_sum` and `multi_channel_noisy_or` select the same documents at raw scale.
- Calibration can force selection differences, but only by pushing phi into a saturated regime.
- At larger scales, noisy-OR sometimes beats channel-sum relatively, but both remain below single-channel dense/query controls.
- Multi-channel split is consistently weaker than single-channel or union-source controls.

The proposal's mathematical warning became the no-go condition:

```text
If multi-channel + noisy-OR does not beat channel-sum and query-level PPR,
delayed aggregation is not a main contribution.
```

That no-go condition has been met.

## Why DAPG Should Be Archived

The evidence now rules out the original DAPG story:

| Proposed DAPG Claim | Current Evidence | Decision |
| --- | --- | --- |
| Demand-conditioned tensor is the core improvement | Multi-channel variants are weaker than dense query and union controls | Reject as main claim |
| Delayed noisy-OR aggregation is essential | Sum and noisy-OR are identical at raw scale | Demote to ablation |
| Local absorption improves retrieval | Absorption is consistently below dense-only controls | Reject current operator |
| Demand union improves over query-level retrieval | Dense demand union is below dense query in most buckets | Reject as main claim |
| Full-corpus DAPG is worth implementing | Fixed-pool absorption is already negative | Stop Phase 3 |

This is not just a calibration failure. The local absorption operator loses
signal before projection:

- graph propagation introduces entity/proposition noise;
- `kappa * hit` suppresses high-cosine correct documents when hit mass is diffuse;
- transient node text is less query-friendly than full document text;
- splitting a query into isolated channels discards joint semantic context.

## Active Paper Direction

The active route should be a narrower DAEC-L1 paper:

```text
Train-free demand-aware fixed-pool evidence projection for strong retriever pools.
```

Recommended positioning:

1. Strong retrievers already retrieve many useful candidates in top-100, but top-5 reader contexts remain noisy or incomplete.
2. DAEC-L1 decomposes the question into retrieval-active demands and selects a top-k context by demand coverage rather than independent document relevance.
3. On 2Wiki dense and PropRAG pools, DAEC-L1 improves EM/F1 because it adds missing support and removes distractors.
4. On MuSiQue long-chain questions, the same method is not robust; this is a limitation and motivates trained or reader-aware evidence utility.
5. Controlled negative ablations show that train-free local graph absorption, demand union, multi-channel propagation, and delayed noisy-OR do not improve over strong dense embedding baselines.

This is a Findings/short-paper scale contribution, not a main-method graph RAG
claim.

## CE Baseline Clarification

Do not summarize the CE evidence as "CE rerank is dead" without qualification.
The current artifacts contain two different CE paths:

1. `--cross_encoder_rerank` over the original baseline context/window is
   negative on the available 2Wiki-100 run:
   `0.4600 / 0.5188 -> 0.3900 / 0.4694` for alpha=0.7 and
   `0.4600 / 0.5188 -> 0.4100 / 0.4850` for alpha=0.9.
2. `bridge_append + assemble_mode=cross_encoder` is positive in the 100-query
   runs and should be treated as a strong assemble-stage baseline. For example,
   2Wiki k=5 `append3 CE` gives
   `0.3600 / 0.4008 -> 0.4300 / 0.4728`, and MuSiQue k=7 `append3 CE` gives
   `0.3200 / 0.3924 -> 0.3800 / 0.4385`.

Correct statement:

```text
Pointwise CE over the original top-k is not sufficient, but CE after structured
expansion is a strong assemble-stage baseline.
```

Paper implication:

- `bridge_append + cross_encoder` must be included as a matched strong baseline
  for DAEC-L1 / Expand-then-Compose.
- If DAEC-L1 only matches assemble-stage CE, frame it as a diagnostic /
  negative-study contribution rather than a method-novelty paper.
- If DAEC-L1 beats assemble-stage CE in full1000 or hard buckets, the stronger
  claim is that structure-aware set selection improves over a strong CE
  assembly baseline under the same expanded pool.

Audit artifact:

- `reports/dpathrag/ce_rerank_vs_assemble_stage_audit_20260429.md`

## AREC-RAG Smoke Outcome

AREC-RAG was tested as a train-free answer-residual evidence closure route after
the DAPG no-go results. The smoke did not support promoting it to the active
paper plan.

Residual retrieval smoke on MuSiQue limit=100:

| Stage | AREC Missing Hit | Raw Missing Hit | CoT Missing Hit | Initial SC@5 | Final SC@5 | Initial Recall@5 | Final Recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Day 0 silver oracle | 0.0320 | 0.0080 | 0.0220 | 0.2900 | 0.5000 | 0.6458 | 0.7675 |
| Day 2 generated | 0.0160 | 0.0080 | 0.0220 | 0.2900 | 0.2400 | 0.6458 | 0.4892 |

Fixed-pool control:

| Source | Projection Pool | Recall@5 | SC@5 | Initial Recall@5 | Initial SC@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| generated | top-20 | 0.4933 | 0.2300 | 0.6458 | 0.2900 |
| generated | top-100 | 0.3042 | 0.1300 | 0.6458 | 0.2900 |
| silver oracle | top-20 | 0.7417 | 0.4500 | 0.6458 | 0.2900 |
| silver oracle | top-100 | 0.6767 | 0.3700 | 0.6458 | 0.2900 |

Decision:

- Silver-oracle closure has signal, but top-20 fixed-pool projection already
  explains most of the apparent Day 0 improvement. Full residual retrieval
  reaches SC@5 0.50, while silver top-20 fixed-pool projection reaches 0.45.
- Generated obligations are actively harmful under projection. They push
  support recall and support completeness below the initial dense top-5.
- Generated residual retrieval does not beat the CoT-style query control.
- AREC should not be treated as a retrieval paper at this stage. If anything,
  closure scoring is an exploratory selector signal to compare against DAEC-L1.

Audit artifacts:

- `reports/dpathrag/arec_rag_fixed_pool_projection_musique_limit100_20260428.md`
- `reports/dpathrag/arec_rag_obligation_failure_audit_musique_limit100_20260428.md`

Do not run full1000 AREC, K=3 multi-hypothesis, or official IRCoT replication
until a manual obligation audit shows that generated-obligation failures are
prompt-fixable and not inherent wrong-answer amplification.

## Suggested Paper Contributions

Use three modest contributions:

1. **DAEC-L1 projection.**
   A train-free fixed-pool evidence projection method that selects a reader-ready context by demand coverage over a retriever top-100 pool.

2. **Mechanistic gain analysis.**
   A query-paired analysis showing that 2Wiki gains come from support completion and distractor pruning, not merely reader variance.

3. **Negative control battery.**
   A controlled evaluation suite showing that strong dense question embeddings dominate train-free local graph absorption and demand-split delayed aggregation in this setting.

Do not claim:

- end-to-end train-free graph retrieval;
- DAPG as a working retriever;
- delayed aggregation as the empirical mechanism;
- general gains on MuSiQue long-chain QA.

## Required Figures / Tables

Main paper tables should include:

1. DAEC-L1 main results:
   - 2Wiki dense pool100;
   - 2Wiki PropRAG pool100;
   - MuSiQue dense pool100 as a limitation row;
   - Hotpot dense pool100 if still available and verified.

2. Gain source table:
   - wrong->correct vs correct->wrong;
   - support recall delta;
   - support complete delta;
   - noise delta.

3. Negative ablation table:
   - dense query cosine;
   - query absorption;
   - dense demand union;
   - demand-source union absorption;
   - multi-channel sum;
   - multi-channel noisy-OR.

4. Hop-bucket limitation table:
   - 2-hop / 3-hop / 4-hop;
   - especially MuSiQue long-chain buckets.

## Execution Decisions

Proceed:

- write DAEC-L1 proposal / paper outline;
- preserve DAEC-DAPG results as negative ablation evidence;
- run significance tests for DAEC-L1 on 2Wiki dense and PropRAG;
- optionally audit Hotpot DAEC-L1 gain source with the same diagnostic script.

Stop:

- DAPG Phase 3 full retrieval;
- local absorption operator development;
- calibration sweeps;
- channel-routing variants;
- binding-mask graph transitions;
- any framing that depends on graph retrieval being positive.

## Key Artifacts

Implementation:

- `scripts/daec_l1_gain_diagnostics.py`
- `scripts/daec_dapg_phase25_dense_grounding.py`
- `src/dpathrag/daec_dapg/`
- `tests/dpathrag/test_daec_dapg_scripts.py`
- `tests/dpathrag/test_daec_dapg_core.py`

DAEC-L1 gain reports:

- `reports/dpathrag/daec_l1_gain_2wiki_dense_full1000_20260428.json`
- `reports/dpathrag/daec_l1_gain_2wiki_proprag_full1000_20260428.json`
- `reports/dpathrag/daec_l1_gain_musique_dense_full1000_20260428.json`

Dense-only hard-gate reports:

- `reports/dpathrag/daec_dapg_phase25_2wiki_dense_api_denseonly_full1000_20260428.json`
- `reports/dpathrag/daec_dapg_phase25_musique_dense_api_denseonly_full1000_20260428.json`

Earlier negative diagnostics:

- `reports/dpathrag/daec_dapg_phase25_2wiki_dense_api_calibration_limit100_20260428.json`
- `reports/dpathrag/daec_dapg_phase25_musique_dense_api_calibration_limit100_20260428.json`
- `reports/dpathrag/daec_dapg_phase25_2wiki_dense_api_crosstalk_limit100_20260428.json`
- `reports/dpathrag/daec_dapg_phase25_musique_dense_api_crosstalk_limit100_20260428.json`

## One-Sentence Current Position

DAEC-DAPG should be archived as a negative train-free graph retrieval attempt;
the viable paper is DAEC-L1 as a fixed-pool demand-aware context projection
method, supported by 2Wiki gains and by a controlled negative ablation suite
showing why graph absorption and delayed channel aggregation are not the right
mechanism under strong dense embeddings.
