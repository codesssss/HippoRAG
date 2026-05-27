# ETv4 Binding Feasibility Audit

Scope: MuSiQue limit100, 4-doc slice, frozen ETv3 retrieval plus baseline-stable DBEC.
No new LLM calls or reader runs were made.

## Headline

- 4-doc queries: `22`; stable DBEC residual answer failures: `18`.
- Strict state-binding-feasible residual rate: `0.0%`; relaxed binding/objective-signal rate: `11.1%`.
- Stable DBEC 4-doc F1: `0.2000`; strict-A fixed ceiling: `0.2000`; relaxed-A fixed ceiling: `0.2909`.
- Proceed gate: strict-A pass = `False`, relaxed-A pass = `False`.

## Bucket Counts

| Bucket | Count |
|---|---:|
| A_partial_binding_or_objective_signal | 2 |
| B_calibration_or_unmodeled_state | 1 |
| B_wrong_binding_or_state_extraction | 5 |
| C_missing_from_etv3_candidate200 | 5 |
| C_outside_dbec_pool100_inside_candidate200 | 5 |

## F1 Ceilings

| Quantity | Value |
|---|---:|
| ETv3 4-doc F1 | 0.1545 |
| Stable DBEC 4-doc F1 | 0.2000 |
| Conservative diagnostic DBEC 4-doc F1 | 0.2455 |
| If strict-A residuals are all fixed | 0.2000 |
| If relaxed-A residuals are all fixed | 0.2909 |
| If all support-incomplete residuals are fixed | 1.0000 |

## Residual Failure Table

| qid | bucket | stable F1 | missing gold | state-bindable | positive DBEC signal | wrong-binding proxy | stable reader@10 F1 | selected non-gold |
|---:|---|---:|---:|---|---|---|---:|---|
| 5 | C_missing_from_etv3_candidate200 | 0.0000 | 2 | no | no | no | 0.0000 | Israel // History of Israel |
| 6 | C_outside_dbec_pool100_inside_candidate200 | 0.0000 | 3 | no | no | yes | 0.0000 | The Hague City Hall // Colorado River (Texas) // Gulf of Mexico // Stockholm Public Library |
| 7 | C_missing_from_etv3_candidate200 | 0.0000 | 3 | yes | no | yes | 0.0000 | American Idol // Prysmian Group // Francisco de Orellana // Biltmore Records |
| 11 | C_outside_dbec_pool100_inside_candidate200 | 0.0000 | 2 | no | no | no | 0.0000 | Somalis |
| 30 | B_wrong_binding_or_state_extraction | 0.0000 | 2 | no | no | yes | 0.0000 | Missouri // Paris // Indiana |
| 33 | B_wrong_binding_or_state_extraction | 0.0000 | 2 | no | no | yes | 0.0000 | Middle East // Nilo Syrtis |
| 49 | C_missing_from_etv3_candidate200 | 0.0000 | 3 | no | no | yes | 0.0000 | Burning Heart Records // Jive Records // Miami // Nanjing |
| 51 | B_wrong_binding_or_state_extraction | 0.0000 | 2 | no | no | yes | 0.0000 | United States // Oklahoma City // Logan Gomez |
| 55 | C_outside_dbec_pool100_inside_candidate200 | 0.0000 | 3 | no | no | no | 0.0000 | Kingdom of Israel (united monarchy) // Judea (Roman province) // Israel // History of Israel |
| 56 | A_partial_binding_or_objective_signal | 0.0000 | 2 | yes | no | yes | 0.0000 | Saint Paul, Minnesota // Oh Yeah (Charles Mingus album) // 2018 Indianapolis 500 |
| 59 | B_calibration_or_unmodeled_state | 0.0000 | 2 | no | no | no | 0.0000 | Ambroise-Marie Carré // Henry Scott Holland // John Kerry |
| 69 | C_outside_dbec_pool100_inside_candidate200 | 0.0000 | 2 | no | no | yes | 0.0000 | United States // Oklahoma City // Saint Paul, Minnesota |
| 72 | C_missing_from_etv3_candidate200 | 0.4000 | 2 | no | no | yes | 0.0000 | Detroit // Washington (state) // Mecklenburg County, North Carolina |
| 75 | C_outside_dbec_pool100_inside_candidate200 | 0.0000 | 2 | no | no | no | 0.0000 | A Lim // Somalis |
| 77 | B_wrong_binding_or_state_extraction | 0.0000 | 2 | no | no | yes | 0.0000 | Josip Broz Tito // The White Suit |
| 83 | B_wrong_binding_or_state_extraction | 0.0000 | 2 | no | no | yes | 0.0000 | Arizona // Mayor of Jersey City, New Jersey // Richmond, Virginia |
| 87 | C_missing_from_etv3_candidate200 | 0.0000 | 3 | no | no | yes | 1.0000 | Capital punishment in the United States // Germany |
| 94 | A_partial_binding_or_objective_signal | 0.0000 | 3 | yes | no | yes | 1.0000 | Sports league ranking // World Series // World Series Most Valuable Player Award // Major League Baseball All-Star Game  |

## Interpretation Guardrails

- `state_bindable_proxy` is conservative: the missing gold must appear in DBEC's LLM binding candidates or selected binding titles/docs.
- `positive_dbec_signal` means the missing gold appears in the DBEC full rebuild set or a safe swap-in step.
- `wrong_binding_proxy` is not ground truth; it flags non-gold selected bindings while gold evidence remains missing.
- These numbers decide whether ETv4-binding is worth implementing; they are not evidence that ETv4-binding already works.
