# DAEC Saturation-Aware Binding Verifier Probe

Date: 2026-05-06

## Purpose

This probe tests whether a conservative structural binding verifier is worth integrating into DAEC.

The policy is intentionally conservative:

```text
Let b0 be the base DAEC-selected binding.
Let T be the set of bindings whose coverage objective is within epsilon of b0.

If |T| = 1:
    keep b0
If b0 has structural support:
    keep b0
If b0 lacks support and a tied alternative has support:
    flip to the first supported tied alternative
Otherwise:
    keep b0
```

This is a post-hoc audit. It does not rerun the reader and does not change selector outputs.

## Artifacts

- Probe script: `scripts/audit_daec_binding_verifier.py`
- Input DAEC traces: `run_logs/daec_binding_grounded_limit100_20260506/evals/*_base_qwen8b_hipporag_pool100_limit100.json`
- Input pools: `run_logs/hipporag_pool_exports_full1000_20260503/*_hipporag_pool100.json`
- Generated probe reports:
  - `reports/daec_binding_verifier_20260506/`
  - `reports/daec_binding_verifier_20260506_selected_companion/`
  - `reports/daec_binding_verifier_20260506_selected_companion_eps001/`
  - `reports/daec_binding_verifier_20260506_selected_companion_eps01/`

## Support Modes

```text
+----------------------------------+--------------------------------------------------------------+
| Mode                             | Structural support condition                                  |
+----------------------------------+--------------------------------------------------------------+
| extraction_or_companion          | Candidate appears in an upstream extraction doc or companion  |
| selected_extraction_or_companion | Extraction doc must also be selected for that binding         |
| selected_companion               | Candidate appears in selected companion evidence only         |
+----------------------------------+--------------------------------------------------------------+
```

The first mode is broader but less safe. The last mode is the strictest and best matches the intended "independent co-evidence" idea.

## Results

### Broad support: `extraction_or_companion`, epsilon = `1e-9`

```text
+----------+------+-------+---------+---------+----------+----------+----------+
| Dataset  | Rows | Tied  | BaseSup | Flips   | SC 1->0  | SC 0->1  | dRecall  |
+----------+------+-------+---------+---------+----------+----------+----------+
| 2Wiki    |  100 |    21 |      87 |       3 |        1 |        1 |  +0.0000 |
| HotpotQA |  100 |    21 |      72 |       2 |        0 |        0 |  +0.0000 |
| MuSiQue  |  100 |    24 |      61 |       4 |        0 |        0 |  +0.0000 |
+----------+------+-------+---------+---------+----------+----------+----------+
```

This setting is too loose. It finds one real-looking 2Wiki support-complete rescue, but also introduces one 2Wiki support-complete regression. The regression is instructive: the verifier follows a cross-document mention to a wrong one-hop parent entity.

### Strict support: `selected_companion`, epsilon = `1e-9`

```text
+----------+------+-------+---------+---------+----------+----------+----------+
| Dataset  | Rows | Tied  | BaseSup | Flips   | SC 1->0  | SC 0->1  | dRecall  |
+----------+------+-------+---------+---------+----------+----------+----------+
| 2Wiki    |  100 |    21 |      82 |       1 |        0 |        0 |  +0.0000 |
| HotpotQA |  100 |    21 |      70 |       0 |        0 |        0 |  +0.0000 |
| MuSiQue  |  100 |    24 |      57 |       3 |        0 |        0 |  +0.0025 |
+----------+------+-------+---------+---------+----------+----------+----------+
```

This setting is safer but mostly inert. It produces no support-complete rescue on any dataset. The only positive signal is a small MuSiQue average support-recall gain from one partial-support improvement.

### Strict support with wider tie thresholds

`selected_companion`, epsilon = `0.001`:

```text
+----------+------+-------+---------+---------+----------+----------+----------+
| Dataset  | Rows | Tied  | BaseSup | Flips   | SC 1->0  | SC 0->1  | dRecall  |
+----------+------+-------+---------+---------+----------+----------+----------+
| 2Wiki    |  100 |    21 |      82 |       1 |        0 |        0 |  +0.0000 |
| HotpotQA |  100 |    21 |      70 |       0 |        0 |        0 |  +0.0000 |
| MuSiQue  |  100 |    26 |      57 |       7 |        1 |        0 |  -0.0050 |
+----------+------+-------+---------+---------+----------+----------+----------+
```

`selected_companion`, epsilon = `0.01`:

```text
+----------+------+-------+---------+---------+----------+----------+----------+
| Dataset  | Rows | Tied  | BaseSup | Flips   | SC 1->0  | SC 0->1  | dRecall  |
+----------+------+-------+---------+---------+----------+----------+----------+
| 2Wiki    |  100 |    22 |      82 |       2 |        0 |        0 |  +0.0000 |
| HotpotQA |  100 |    22 |      70 |       1 |        0 |        0 |  +0.0000 |
| MuSiQue  |  100 |    29 |      57 |       8 |        1 |        0 |  -0.0050 |
+----------+------+-------+---------+---------+----------+----------+----------+
```

Widening the tie threshold increases flips without producing support-complete rescues. On MuSiQue it introduces a support-complete regression.

## Flip Examples

The broad mode has one useful-looking 2Wiki rescue:

```text
Query 56: Which country the composer of film Thunder On The Hill is from?
Gold:    Hans J. Salter, Thunder on the Hill
Base:    Walter Ulfig, Thunder on the Hill, Bert Grund, Abe Meyer, Henri Verdun
Verify:  Hans J. Salter, Walter Ulfig, Thunder on the Hill, Bert Grund, Abe Meyer
Recall:  0.5 -> 1.0
```

But the same broad mode also regresses:

```text
Query 14: Where was the place of death of Maurice, Prince Of Orange's father?
Gold:    Maurice, Prince of Orange, William the Silent
Base:    Maurice, Prince of Orange, ..., William the Silent
Verify:  William I, Count of Nassau-Dillenburg, Maurice, Prince of Orange, ...
Recall:  1.0 -> 0.5
```

The strict mode avoids that regression but loses the rescue.

## Interpretation

1. Saturation-aware gating is conceptually cleaner than unconditional posterior reranking.

   It avoids broad damage: the exact-tie strict verifier flips only 1 to 3 queries per dataset.

2. The lexical structural signal is too weak for a method contribution.

   The broad mode can find a real relation-looking rescue, but it also follows wrong one-hop mentions. The strict mode is safer but mostly inert.

3. The current signal does not pass the threshold for selector integration.

   The desired pattern was low regression plus nonzero support-complete rescue. The probe shows low regression only when the method is so conservative that it has no complete rescue.

4. A relation-aware verifier would be needed to make this work.

   Mention co-evidence alone cannot distinguish "the candidate is the answer to the upstream demand" from "the candidate is merely related to a selected document."

## Decision

Do not integrate this verifier into DAEC now.

Recommended status:

```text
+-------------------------------+-------------------------------+
| Candidate                     | Decision                      |
+-------------------------------+-------------------------------+
| saturation-aware trigger      | Keep as a useful diagnostic    |
| selected-companion support    | Too inert for selector use     |
| extraction-or-companion mode  | Too unsafe for selector use    |
| wider objective epsilon       | Not helpful                    |
| reader rerun for this probe   | Not justified by support stats |
+-------------------------------+-------------------------------+
```

## Method Lesson

The previous grounded posterior failed because embedding similarity is not binding entailment. This structural probe shows the next constraint:

```text
Entity co-mention is not relation entailment.
```

The next viable binding verifier must score whether the candidate entity satisfies the upstream relation, not merely whether it appears near an upstream support document.

## Next Step

If this line is revisited, the next probe should be relation-aware and still conservative:

- infer a coarse relation phrase from the upstream requirement;
- look for a local text window connecting the anchor and candidate entity;
- allow override only in objective-saturated cases;
- require no support-complete regressions on limit100 before any reader rerun.
