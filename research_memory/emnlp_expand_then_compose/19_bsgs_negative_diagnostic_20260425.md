# BSGS Negative Diagnostic - 2026-04-25

## Context

This note records the outcome of the **BSGS (Belief-State Graph Scratchpad)** Week-0/Week-1 validation branch.

The branch tested whether a graph-native multi-step belief operator could become a stronger alternative to the DAEC fixed-pool composer:

- `DAEC`: one-step evidence-set posterior composition over a fixed retriever pool.
- `BSGS`: tractable node-level marginal filtering over proposition nodes.

The validation was intentionally bounded. Week 1 used oracle MuSiQue slot order so that slot generation quality would not hide the operator behavior.

## Artifacts

Method and plan:

- `docs/bsgs_week0_week1_plan_20260425.md`
- `research_memory/emnlp_expand_then_compose/18_bsgs_week0_week1_plan_20260425.md`

Implementation:

- `src/bsgs/`
- `scripts/bsgs_*.py`
- `tests/bsgs/`

Reports:

- `reports/week0/split_audit.md`
- `reports/week0/qwen_slot_quality.md`
- `reports/week0/qwen_slot_quality_posthoc.md`
- `reports/week0/nli_calibration.md`
- `reports/week1/operator_validation.md`
- `reports/week1/operator_mechanism_analysis.md`
- `reports/week1/bsgs_oracle_diagnostic_musique200.md`

Validation:

- `pytest tests/bsgs`
- Result: `11 passed`

## Week-0 Findings

### MuSiQue split audit

The local evaluation file is:

- `reproduce/dataset/musique.json`
- `1000` answerable examples
- local `MuSiQue-Ans` subset, not the official full `2417`-example dev split

This is acceptable for internal operator diagnostics because it matches the local DAEC Layer-1 protocol, but it should not be presented as the official full MuSiQue dev result without a split-alignment note.

### Qwen slot quality

Raw Qwen slot quality over the local MuSiQue-1000 subset:

| Metric | Value |
|---|---:|
| slot precision | `0.6551` |
| slot recall | `0.6769` |
| slot F1 | `0.6552` |
| exact variable grounding | `0.0019` |

Post-hoc diagnostics avoid brittle exact variable-name matching and instead check position/dependency structure:

| Metric | Value |
|---|---:|
| avg gold slots | `2.6479` |
| avg predicted slots | `2.7834` |
| slot count exact | `0.6309` |
| slot count within one | `0.9428` |
| dependency presence accuracy | `0.6908` |
| position-chain accuracy | `0.5575` |

Interpretation:

- Qwen usually predicts the right number of slots.
- Slot text quality is near the Week-2 threshold but does not pass the original `0.70` recall gate.
- Dependency wiring is weak, so latent-slot BSGS should remain diagnostic-only.

### Verifier calibration

Formal DeBERTa-NLI calibration did not run successfully in this branch. The available report marks the model as unavailable, so the semi-absorbing transition remains an ablation/fallback rather than a main operator component.

## Week-1 Oracle-Slot Operator Result

BSGS oracle-slot validation was run on the same local MuSiQue-1000 protocol:

| Metric | Value |
|---|---:|
| supporting paragraph recall | `0.2268` |
| avg belief entropy | `2.4719` |
| avg duplicate title rate | `0.2654` |

Support coverage bins:

| Bin | Count | Fraction | Avg Recall | Avg Entropy |
|---|---:|---:|---:|---:|
| full support covered | `54` | `0.0540` | `1.0000` | `2.7557` |
| partial support covered | `400` | `0.4000` | `0.4319` | `2.5009` |
| no support covered | `546` | `0.5460` | `0.0000` | `2.4226` |

The result is negative for the full BSGS route.

The important point is that this failure occurs under oracle slot order. The branch already removes Qwen decomposition quality as the main confounder. The remaining failure is operator/substrate-level.

## Diagnosis

### 1. Node marginal state loses set/path coherence

BSGS tracks:

```text
b_t(p) = sum_{S_t contains p} P(S_t | q, o_{1:t})
```

This marginal is tractable, but it drops the joint information needed for multi-hop QA:

- which previous evidence item produced a binding;
- whether a downstream evidence item uses the same entity;
- whether the selected evidence forms one coherent chain.

DAEC's strongest validated component is dependency binding across the evidence set. BSGS removes the set-level binding constraint and tries to recover it through node-level belief mass. The Week-1 result suggests that this approximation is too lossy for MuSiQue.

### 2. HippoRAG graph edges are not reliable evidence-chain transitions

BSGS assumes graph propagation can approximate "move from current evidence to next-hop evidence." The current graph substrate contains many edges whose semantics are weaker than that:

- passage-phrase containment;
- synonym/alias links;
- co-occurrence-style associations;
- noisy OpenIE relations.

The belief operator therefore diffuses over association structure rather than stable reasoning transitions. This is consistent with the earlier QBF negative result: replacing fixed-pool composition with graph-native directional flow is not robust on the current HippoRAG graph.

### 3. Verifier quality is not the main explanation

The available Week-1 runner used lexical fallback likelihoods because formal DeBERTa calibration was unavailable. A better verifier might improve local likelihoods, but it does not restore the joint set/path coherence lost by node marginals, nor does it change the graph substrate semantics.

### 4. Slot generation is not the root cause

Qwen latent slots are not strong enough for a main path, but oracle slots were already used in the operator validation. Since the oracle-slot result is weak, moving to latent slots would make the route harder rather than easier.

## Decision

Stop BSGS as a main method line.

Do not continue with:

- full latent-slot BSGS;
- soft binding;
- DPP over posterior nodes;
- full graph-native BSGS retrieval;
- additional tuning of the current node-marginal operator.

Keep BSGS as a negative diagnostic for the DAEC paper.

## Paper Implication

This result is useful as a "why not graph-native belief propagation?" section:

> We explored a multi-step graph belief filtering alternative under oracle slot ordering. Even with gold-decomposed slot sequences, the operator covered all supporting paragraphs in only `5.4%` of MuSiQue-1000 examples and covered none in `54.6%`. This suggests a substrate mismatch: current HippoRAG graph edges encode lexical and co-occurrence associations rather than reliable evidence-chain transitions, while node-marginal belief drops the set-level binding coherence that fixed-pool composition explicitly maintains.

This supports the DAEC framing:

- the robust intervention point is fixed-pool evidence composition;
- graph-native retrieval/operator replacement is not automatically better, even under oracle decomposition;
- BSGS and QBF are negative controls that narrow the paper's claim boundary.

## MuSiQue-200 Oracle Diagnostic Follow-Up

A small oracle diagnostic was run after the main negative result to sharpen the failure attribution:

- output JSON: `reports/week1/bsgs_oracle_diagnostic_musique200.json`
- output report: `reports/week1/bsgs_oracle_diagnostic_musique200.md`

The diagnostic compares four variants:

| Variant | Purpose |
|---|---|
| `base` | Reproduce the minimal oracle-slot BSGS operator. |
| `+ oracle pool` | Test whether gold support is present but not selected. |
| `+ oracle pool + oracle binding rewrite` | Test whether unresolved `#1/#2` variables are the main bottleneck. |
| `+ oracle pool + oracle binding + oracle likelihood` | Test whether even perfect likelihoods fail under graph/state transition. |

Results on local MuSiQue-200:

| Variant | Legacy title recall | Paragraph-idx recall | Full / Partial / None by idx | Avg entropy | Duplicate-title rate |
|---|---:|---:|---:|---:|---:|
| `base` | `0.2050` | `0.1608` | `2 / 71 / 127` | `2.5407` | `0.2930` |
| `oracle_pool` | `0.2454` | `0.2113` | `5 / 86 / 109` | `2.6282` | `0.3360` |
| `oracle_pool_binding` | `0.4958` | `0.4758` | `32 / 141 / 27` | `2.5464` | `0.3480` |
| `oracle_pool_binding_likelihood` | `0.5854` | `0.5746` | `43 / 157 / 0` | `3.3023` | `0.7170` |

Interpretation:

- `oracle_pool` only improves paragraph-idx recall from `0.1608` to `0.2113`, so sentence truncation / candidate exposure is not the primary bottleneck.
- `oracle_pool_binding` is the largest jump (`0.2113` to `0.4758`), confirming that unresolved variable binding is a real failure source.
- `oracle_pool_binding_likelihood` removes no-support failures but still fully covers support in only `43 / 200` examples and creates high duplicate-title concentration. Even perfect slot-level likelihood does not recover coherent evidence sets.
- The remaining failure is therefore a combination of node-marginal set-coherence loss and graph/state transition mismatch, not just slot quality, pool exposure, or verifier calibration.

This follow-up should be treated as diagnostic only. It does not reopen the full BSGS method route. Its paper value is a sharper negative-control table for the "why not graph-native belief propagation?" section.

## Current IRCoT Note

The formal `IRCoT MuSiQue-1000` run did not finish cleanly in the current session. It exited before writing:

- `reports/week0/ircot_musique1000.json`

Only the smoke/cache warmup output exists:

- `/tmp/bsgs_ircot_smoke.json`

This does not affect the BSGS negative operator diagnosis, because the negative result is based on oracle-slot mechanism metrics, not answer F1 against IRCoT.
