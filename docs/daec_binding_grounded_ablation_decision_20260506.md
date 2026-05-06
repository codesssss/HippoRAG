# DAEC Binding-Grounded Ablation Decision

Date: 2026-05-06

## Scope

This note records the decision from the limit100 ablation of binding-grounded DAEC controls.

The experiment should not be read as a new DAEC headline result. It is a method diagnostic for four optional controls:

- `typeguard`: type filter for LLM-extracted binding entities.
- `softcompat`: partial compatibility when a bound entity is mentioned in document body text but does not exactly match the title.
- `grounded`: binding reranking by selection-independent upstream phi support for the assigned entity document.
- `full`: `typeguard + softcompat + grounded + 1-swap`.

## Artifacts

- Launcher: `run_logs/launch_daec_binding_grounded_limit100_20260506.sh`
- Run directory: `run_logs/daec_binding_grounded_limit100_20260506/`
- Auto summary: `run_logs/daec_binding_grounded_limit100_20260506/summary_daec_binding_grounded_limit100.md`
- CSV: `run_logs/daec_binding_grounded_limit100_20260506/daec_binding_grounded_limit100_results.csv`
- Code commits:
  - `cc562fd Add binding-grounded DAEC controls`
  - `f137f4d Allow partial DAEC ablation dataset launches`

## Configuration

```text
Pool:              HippoRAG aligned legacy_fact_graph pool100
Datasets:          2Wiki, HotpotQA, MuSiQue
Limit:             100 queries per dataset
Reader/LLM:        Qwen3-8B train
Thinking mode:     /no_think
Embedding:         NV-Embed-v2
Selector:          daec_noisyor_llm
Binding mode:      wiki_title
QA top-k:          5
Decomposition:     llm
Causal rerank:     disabled
Structure rerank:  disabled
```

Variant definitions:

```text
+------------+------------+------------+-----------+--------+
| Variant    | TypeGuard  | SoftCompat | Grounding | Swap   |
+------------+------------+------------+-----------+--------+
| base       | false      | 0.0        | false     | false  |
| typeguard  | true       | 0.0        | false     | false  |
| softcompat | true       | 0.5        | false     | false  |
| grounded   | true       | 0.0        | true      | false  |
| full       | true       | 0.5        | true      | true   |
+------------+------------+------------+-----------+--------+
```

## Code Locations

- Type filter in LLM binding path: `scripts/dtc_embed_utils.py:1650`
- Soft body-mention compatibility: `scripts/dtc_embed_utils.py:624`
- Binding grounding score: `scripts/dtc_embed_utils.py:919`
- 1-swap refinement helper: `scripts/dtc_embed_utils.py:974`
- 1-swap integration in DAEC loop: `scripts/dtc_embed_utils.py:2165`
- Selector/eval argument plumbing: `scripts/eval_causal_qwen3.py:5342`, `scripts/eval_causal_qwen3.py:7902`
- Launcher variant matrix: `run_logs/launch_daec_binding_grounded_limit100_20260506.sh:39`

## Main Results

```text
+----------+------------+--------+--------+--------+--------+--------+---------+---------+----------+
| Dataset  | Variant    | EM     | F1     | R@5    | dEM    | dF1    | TypeRej | GroundC | SwapStep |
+----------+------------+--------+--------+--------+--------+--------+---------+---------+----------+
| 2Wiki    | base       |  0.530 |  0.595 |  0.900 | +0.000 | +0.000 |       0 |       0 |        0 |
| 2Wiki    | typeguard  |  0.530 |  0.595 |  0.900 | +0.000 | +0.000 |       3 |       0 |        0 |
| 2Wiki    | softcompat |  0.540 |  0.609 |  0.900 | +0.010 | +0.013 |       3 |       0 |        0 |
| 2Wiki    | grounded   |  0.490 |  0.535 |  0.863 | -0.040 | -0.060 |       3 |      13 |        0 |
| 2Wiki    | full       |  0.510 |  0.558 |  0.868 | -0.020 | -0.037 |       3 |      13 |        1 |
| HotpotQA | base       |  0.600 |  0.706 |  0.945 | +0.000 | +0.000 |       0 |       0 |        0 |
| HotpotQA | typeguard  |  0.600 |  0.706 |  0.945 | +0.000 | +0.000 |       4 |       0 |        0 |
| HotpotQA | softcompat |  0.600 |  0.706 |  0.945 | +0.000 | +0.000 |       4 |       0 |        0 |
| HotpotQA | grounded   |  0.600 |  0.706 |  0.945 | +0.000 | +0.000 |       4 |      10 |        0 |
| HotpotQA | full       |  0.600 |  0.706 |  0.945 | +0.000 | +0.000 |       4 |      10 |        0 |
| MuSiQue  | base       |  0.340 |  0.438 |  0.691 | +0.000 | +0.000 |       0 |       0 |        0 |
| MuSiQue  | typeguard  |  0.330 |  0.428 |  0.683 | -0.010 | -0.010 |      13 |       0 |        0 |
| MuSiQue  | softcompat |  0.350 |  0.435 |  0.683 | +0.010 | -0.003 |      13 |       0 |        0 |
| MuSiQue  | grounded   |  0.330 |  0.425 |  0.682 | -0.010 | -0.013 |      13 |      10 |        0 |
| MuSiQue  | full       |  0.320 |  0.403 |  0.676 | -0.020 | -0.036 |      13 |      10 |        1 |
+----------+------------+--------+--------+--------+--------+--------+---------+---------+----------+
```

`dEM` and `dF1` are deltas against the DAEC base variant on the same dataset and same pool.

## Answer Transition Diagnostics

Base vs grounded:

```text
+----------+---------+---------+----------------+----------+
| Dataset  | 1 to 0  | 0 to 1  | BindingChanged | Net dEM  |
+----------+---------+---------+----------------+----------+
| 2Wiki    |       4 |       0 |             13 | -0.040   |
| HotpotQA |       0 |       0 |             10 | +0.000   |
| MuSiQue  |       2 |       1 |             10 | -0.010   |
+----------+---------+---------+----------------+----------+
```

Base vs softcompat:

```text
+----------+---------+---------+-----------------------+------------+
| Dataset  | 1 to 0  | 0 to 1  | SelectedTitlesChanged | Net dEM    |
+----------+---------+---------+-----------------------+------------+
| 2Wiki    |       0 |       1 |                    22 | +0.010     |
| HotpotQA |       0 |       0 |                     7 | +0.000     |
| MuSiQue  |       1 |       2 |                    16 | +0.010     |
+----------+---------+---------+-----------------------+------------+
```

2Wiki grounded regressions:

```text
+-------+-----------------+-----------+--------------------+-----------+----------+
| Query | Base Binding    | Grounding | Grounded Binding   | Grounding | EM       |
+-------+-----------------+-----------+--------------------+-----------+----------+
| 16    | Frank Lloyd     | 0.183888  | Claude Autant-Lara | 0.718878  | 1 to 0   |
| 71    | Bostjan Hladnik | 0.303777  | Ian Barry          | 0.526284  | 1 to 0   |
| 80    | Oskar Roehler   | 0.263657  | Claude Weisz       | 0.740694  | 1 to 0   |
| 93    | Yvan Chiffre    | 0.468814  | Yonfan             | 0.857706  | 1 to 0   |
+-------+-----------------+-----------+--------------------+-----------+----------+
```

The common pattern is that upstream phi prefers the more prominent or semantically rich entity page, not the factually correct binding entity. For example, both the correct and wrong binding can saturate the noisy-OR objective at 1.0, so the grounding multiplier becomes the discriminator. That discriminator is still embedding similarity, not factual entailment.

## Interpretation

1. `typeguard` is a code-consistency fix, not a method contribution.

   The LLM binding path now has the same kind of compatibility guard as the frozen binding path. It rejects only a small number of entities: 3 on 2Wiki, 4 on HotpotQA, 13 on MuSiQue. The performance effect is zero or slightly negative.

2. `softcompat` is weak and not yet a reliable method signal.

   It gives +0.010 EM on 2Wiki and MuSiQue, and no change on HotpotQA. The query-level transitions are tiny. On MuSiQue, some gains answer correctly without recovering all gold titles, so the signal is not clean evidence-selection improvement.

3. `grounded` should not be promoted.

   The implementation is faithful to the proposed selection-independent posterior idea, but the signal is wrong for this job. It uses upstream phi support for the assigned entity document. Since phi is embedding similarity, it favors salient but wrong entities in several 2Wiki cases and causes a clear -0.040 EM / -0.060 F1 regression.

4. `full` is pulled down by `grounded`.

   The combined variant is worse than base on 2Wiki and MuSiQue and unchanged on HotpotQA. It should not be expanded to full1000 in this form.

5. `swap_refinement` is inert in this setting.

   It triggers once across all 15 runs. This suggests that greedy noisy-OR optimization is not the current bottleneck.

## Decision

```text
+--------------------+------------------------+---------------------------------------------+
| Feature            | Status                 | Rationale                                   |
+--------------------+------------------------+---------------------------------------------+
| type_filter        | Keep as implementation | Restores a missing LLM binding guard.        |
| soft_compat        | Keep optional/off      | Small noisy signal; needs better diagnosis.  |
| binding_grounding  | Stop this version      | Embedding-derived grounding is misleading.   |
| swap_refinement    | Keep optional/off      | Theoretical harmlessness, no practical gain. |
| full variant       | Do not use             | Grounding component dominates negatively.    |
+--------------------+------------------------+---------------------------------------------+
```

Default behavior should remain backward compatible: base DAEC keeps all these controls off.

## Method Lesson

The failure is not caused by an obvious implementation bug. It is a signal-design failure:

```text
Embedding similarity is not binding entailment.
```

Any binding-selection signal derived from the same embedding space that builds phi risks amplifying the original phi bias. The current grounded posterior is therefore circular in practice, even though it no longer directly conditions on the selected document set.

If binding selection is revisited, it needs an independent factual signal, such as:

- text evidence that the upstream source document states the relation to the candidate entity;
- OpenIE or graph edge support connecting the upstream demand to the assigned entity;
- a dedicated verifier/NLI signal, if the train-free constraint is relaxed or the verifier can be justified as inference-only.

## Paper Positioning

Do not write this as a positive DAEC improvement.

Recommended positioning:

- DAEC remains a train-free decomposition-aware evidence selector.
- Its limitation is latent binding quality: when the binding hypothesis is wrong, noisy-OR coverage can still saturate.
- The failed grounded posterior is a useful negative diagnostic: embedding-derived self-verification prefers salient wrong entities and is not sufficient for factual binding validation.
- The near-inert swap result suggests optimization quality is less important than phi/binding quality.

## Next Actions

1. Do not run `grounded` or `full` on full1000.
2. Keep `typeguard` code, but report it only as an implementation fix.
3. Leave `softcompat` as an optional diagnostic; do not claim it improves DAEC without stronger per-query support.
4. If continuing on binding, build a separate verifier probe based on relation/text/graph evidence rather than phi similarity.
