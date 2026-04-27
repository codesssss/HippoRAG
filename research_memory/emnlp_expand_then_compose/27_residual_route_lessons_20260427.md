# Residual Route Lessons - 2026-04-27

## Purpose

This note records the accumulated lessons from the recent residual-method exploration around DAEC, BSGS, D-PathRAG/CEE/C-CEE, CAPS, CPAG, and Answer-Contrastive Verifier v0.

This is not a paper-writing plan. It is a neutral checkpoint for future continuation.

## Current State

The strongest positive line remains `DtC/DAEC` as a retriever-agnostic fixed-pool evidence compositor.

The recent additional routes were tested to see whether the remaining headroom after strong retrieval could be recovered by a stronger selector, proof scorer, agreement operator, or lightweight learned verifier. The overall result is:

```text
DAEC remains the positive floor.
The tested residual routes expose useful failure mechanisms, but none is currently strong enough to replace the main line.
```

## Positive Floor: DAEC

DAEC improves frozen-reader utility across both PropRAG and dense retrieval pools.

| Pool | Dataset | Base F1 | DAEC F1 | Delta F1 | Oracle F1 | Approx Gap Recovery |
|---|---:|---:|---:|---:|---:|---:|
| PropRAG top100 | 2Wiki | 0.6457 | 0.6810 | +0.0353 | 0.7311 | 41.3% |
| PropRAG top100 | HotpotQA | 0.7227 | 0.7348 | +0.0121 | 0.7697 | 25.7% |
| PropRAG top100 | MuSiQue | 0.4266 | 0.4404 | +0.0138 | 0.6019 | 7.9% |
| Dense top100 | 2Wiki | 0.4984 | 0.5628 | +0.0644 | 0.6396 | 45.6% |
| Dense top100 | HotpotQA | 0.7106 | 0.7326 | +0.0220 | 0.7669 | 39.1% |
| Dense top100 | MuSiQue | 0.3896 | 0.4138 | +0.0242 | 0.5521 | 14.9% |

Significance status:

- All six F1 deltas have bootstrap 95% CI above zero.
- All six F1 deltas have paired sign-flip p-value below 0.05.
- MuSiQue remains the weakest positive setting because it has the largest oracle gap and lowest recovery.

Component status:

| Dataset | Pool | Full DAEC Delta F1 | NoBinding Delta F1 | Binding Contribution |
|---|---:|---:|---:|---:|
| 2Wiki | PropRAG | +0.0353 | +0.0142 | +0.0211 |
| 2Wiki | Dense | +0.0644 | +0.0100 | +0.0544 |
| HotpotQA | PropRAG | +0.0121 | +0.0009 | +0.0112 |
| MuSiQue | PropRAG | +0.0138 | +0.0055 | +0.0083 |
| MuSiQue | Dense | +0.0242 | -0.0003 | +0.0245 |

Dependency binding is the most defensible load-bearing component so far. Rank prior and repair typing are not yet supported as necessary components.

## Route Outcomes

| Route | Tested Hypothesis | Outcome | Current Status |
|---|---|---|---|
| BSGS | Graph-native belief-state propagation can outperform fixed-pool composition | Oracle-slot validation still weak | Stop as main route |
| D-PathRAG v1 | Free-form selector can improve PropRAG top-5 evidence | Support improves but F1 drops | Keep as diagnostic |
| CEE pairwise | Pairwise learned admission can recover oracle edit headroom | Beneficial edit recall remains very low | Stop |
| C-CEE | Reader counterfactual likelihood can decide edits | Non-oracle separability near random | Stop |
| CAPS | NLI proof scoring can rerank answer candidates | Local NLI works, answer-level proof ranking fails | Stop current form |
| CPAG/RRF | Cross-pool proposition/entity agreement can assemble evidence | Agreement over-rewards shared distractors | Stop current form |
| Answer-Contrastive v0 | Cached scalar candidate/proof features can learn residual answer ranking | Top-k improves, hard AUC weak | Partial, not mainline |

## Failure Mechanisms

### BSGS

Observed:

- Oracle MuSiQue slot order was used to remove decomposition quality as the main confounder.
- Full support covered only 54/1000 in the oracle-slot mechanism validation.

Lesson:

```text
Node-marginal belief states lose joint set/path coherence.
```

The graph has useful associations, but local node beliefs do not preserve which evidence items form one coherent multi-hop chain.

### D-PathRAG / CEE / C-CEE

Observed:

- D-PathRAG v1 support-complete improves from 0.7720 to 0.8050.
- Reader F1 drops from 0.4824 to 0.4634.
- Selector adds 76 gold docs but also 1205 non-gold docs.
- CEE pairwise beneficial edit recall is only 0.0221 / 0.0588.
- C-CEE non-oracle beneficial-vs-lexical-HN AUC is 0.536875.

Lesson:

```text
Oracle edit headroom exists, but practical admission is the hard part.
```

Free-form selection can import gold supports, but it imports too many reader-hostile hard negatives. Pairwise preference and reader likelihood do not become calibrated edit decisions under non-oracle answer uncertainty.

### CAPS

Observed:

- Day-0 NLI verifier sanity passes with AUC 0.903.
- Day-1 candidate recall is weak: Recall@5/10/20 = 0.670 / 0.740 / 0.750.
- Day-1.5 LLM + reader + string union improves Recall@20 to 0.840 but Recall@5 remains 0.635.
- Day-2 oracle-obligation proof ranking fails:
  - conditional top1/top3 = 0.339286 / 0.583333
  - gold-vs-best-wrong AUC = 0.398136
  - mean gold proof score = 0.383771
  - mean best-wrong proof score = 0.530366

Lesson:

```text
Local obligation-level NLI does not automatically aggregate into answer-level proof ranking.
```

The NLI verifier is not useless; it works on the local sanity check. The failure is the aggregation/proof object.

### CPAG / RRF

Observed:

- Cross-pool document presence has local gold-vs-non-gold AUC 0.772391.
- CPAG support-complete drops from 0.705 to 0.470.
- CPAG reader F1 drops from 0.4720 to 0.3897.
- CPAG adds only 3 gold supports while adding 87 non-gold docs relative to PropRAG top5.
- Audit passed; RRF/CPAG failure is not an implementation artifact.
- RRF pushes PropRAG top1 below rank 3 in 12/200 queries because shared distractors receive two reciprocal-rank contributions.

Lesson:

```text
Cross-pool redundancy is not specificity.
```

Agreement is a real local signal, but it over-rewards shared hard negatives and high-degree neighbor hubs.

### Answer-Contrastive Verifier v0

Observed:

- Uses 4000 candidate rows from 200 dev queries.
- No new LLM calls.
- No DeBERTa/Qwen fine-tuning.
- Best model: HistGradientBoosting over scalar cached candidate/proof/source features.

| Model | Gold-vs-Best-Wrong AUC | 95% CI | Top1 Cond | Top3 Cond |
|---|---:|---:|---:|---:|
| proof_score | 0.398136 | [0.366337, 0.425542] | 0.339286 | 0.583333 |
| hist_gradient_boosting | 0.570826 | [0.514952, 0.624433] | 0.595238 | 0.827381 |

Per-type pattern:

| Type | Gold-vs-Best-Wrong AUC | Top1 Cond |
|---|---:|---:|
| bridge_comparison | 0.759452 | 0.847826 |
| comparison | 0.740100 | 0.725490 |
| compositional | 0.322869 | 0.350877 |
| inference | 0.229592 | 0.285714 |

Lesson:

```text
Scalar candidate/proof/source features can recover top-k placement, but they do not solve hard same-query separability.
```

The result is partial, not green. It shows signal exists, but mostly on comparison-style questions and easy negatives.

## Cross-Route Pattern

The repeated pattern is:

```text
local signal exists
but global decision fails
```

More specifically:

| Local Signal | Failed Global Target |
|---|---|
| graph node score | coherent evidence path/set |
| support selector score | reader-robust top-5 context |
| pairwise edit preference | calibrated edit admission |
| reader likelihood | true edit usefulness |
| NLI entailment | answer-level proof ranking |
| cross-pool agreement | specific gold support assembly |
| scalar answer features | same-query gold-vs-best-wrong separation |

This is the most important accumulated lesson:

```text
After a strong substrate like PropRAG, the remaining errors are not mostly about finding any local signal. They are about safely composing weak local signals into a globally correct evidence or answer decision under hard-negative competition.
```

## Training-Free + LLM Lesson

Training-free + LLM is not rejected.

The useful distinction is:

```text
good paradigm:
  LLM performs local extraction
  robust operator works on noisy output
  operator matches the benchmark bottleneck

bad residual pattern:
  LLM or local extraction produces plausible objects
  operator cannot distinguish gold from shared hard negatives
  local signal collapses under global selection
```

HippoRAG / PropRAG succeeded because their representations and operators matched a large missing-support bottleneck. Our residual setting is different: PropRAG already gives a strong substrate, so the remaining problem is subtle gold-vs-distractor discrimination.

CPAG is the cleanest local test of the training-free LLM idea in this cycle. It used cached Qwen OpenIE extraction plus a robust graph-style agreement operator, but the agreement object was not specific enough.

LCPS-style typed program execution remains untested and should not be treated as disproven. The reason it was not run is risk assessment: it requires high-precision typed extraction, query compilation, predicate canonicalization, and variable binding, which is a stronger assumption than the existing Qwen/CAPS/PCRS evidence supports.

## Current Practical Stop Conditions

Do not continue the current forms of:

- BSGS node-marginal belief propagation.
- D-PathRAG free-form selector tuning.
- CEE pairwise admission.
- C-CEE reader likelihood admission.
- CAPS NLI noisy-OR proof scoring.
- CPAG agreement-closure tuning.
- Answer-Contrastive scalar-feature rankers.

Only reopen a route if the core object changes, not by threshold or weight tuning.

Reasonable reopen conditions:

- A new proof object that preserves joint variable binding rather than noisy-OR local obligations.
- A trained semantic answer verifier with real text snippets and hard negative mining, treated as a new learned method line.
- A claim/proposition specificity model that can penalize shared distractor hubs.
- A measured typed extraction/query compilation benchmark before attempting symbolic execution.
- A reader-robust objective trained directly on final answer correctness.

## Useful Short Phrases For Future Notes

- Support recall is not reader utility.
- Agreement is not specificity.
- Local entailment is not global proof.
- Pairwise preference is not calibrated admission.
- Reader likelihood is not non-oracle edit usefulness.
- Candidate-level separability is not same-query best-wrong separability.
- Training-free does not mean no learning; it shifts learning into pretrained models, offline indexing, and structural priors.
- On strong substrates, residual errors are local-to-global failures rather than missing-local-signal failures.

## Pointers

Primary positive memos:

- `research_memory/emnlp_expand_then_compose/16_layer1_retriever_agnostic_composition_20260424.md`
- `research_memory/emnlp_expand_then_compose/17_layer1_followup_taxonomy_significance_20260424.md`

Negative/partial diagnostic memos:

- `research_memory/emnlp_expand_then_compose/19_bsgs_negative_diagnostic_20260425.md`
- `research_memory/emnlp_expand_then_compose/20_dpathrag_cee_ccee_negative_diagnostic_20260427.md`
- `research_memory/emnlp_expand_then_compose/21_caps_day0_nli_sanity_20260427.md`
- `research_memory/emnlp_expand_then_compose/22_caps_day1_candidate_recall_failure_20260427.md`
- `research_memory/emnlp_expand_then_compose/23_caps_day1_5_candidate_v2_failure_20260427.md`
- `research_memory/emnlp_expand_then_compose/24_caps_day2_proof_separability_20260427.md`
- `research_memory/emnlp_expand_then_compose/25_cpag_day1_agreement_failure_20260427.md`
- `research_memory/emnlp_expand_then_compose/26_answer_contrastive_verifier_20260427.md`

