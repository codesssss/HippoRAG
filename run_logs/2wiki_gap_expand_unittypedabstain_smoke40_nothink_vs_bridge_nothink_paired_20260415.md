# Gap Expand Paired Audit

Shared queries: 40

## Overall

- F1 win/tie/lose: 0 / 40 / 0 (mean delta 0.0)
- EM win/tie/lose: 0 / 40 / 0 (mean delta 0.0)
- Selected-gap subset: 1 queries, F1 win/tie/lose 0 / 1 / 0
- Changed-evidence subset: 2 queries, F1 win/tie/lose 0 / 2 / 0
- Unchanged-evidence subset: 38 queries, F1 win/tie/lose 0 / 38 / 0

## By Bucket

- 2_doc: count=32, F1 win/tie/lose=0 / 32 / 0, mean delta=0.0, selected_gap_rate=0.0
- 4_doc: count=8, F1 win/tie/lose=0 / 8 / 0, mean delta=0.0, selected_gap_rate=0.125

## Gap Trace

- gap_type_distribution: {'target_attribute': 14, 'abstain': 12, 'role_relation': 14}
- gap_slot_distribution: {'death_date': 3, 'director': 14, 'birthplace': 6, 'nationality': 3, 'education': 2}
- fallback_used_count: 12
- selected_gap_count: 1 (rate 0.025)
- by_stop_reason: {'no_eligible_gap_units': {'count': 10, 'mean_delta_f1': 0.0, 'win': 0, 'lose': 0}, 'structure_below_threshold': {'count': 29, 'mean_delta_f1': 0.0, 'win': 0, 'lose': 0}, 'unit_alternate_added': {'count': 1, 'mean_delta_f1': 0.0, 'win': 0, 'lose': 0}}
- top selected-gap titles in regressions: []
- top selected-gap titles in improvements: []

## By Gap Type

- abstain: count=12, F1 win/tie/lose=0 / 12 / 0, mean delta=0.0, selected_gap_rate=0.0
- role_relation: count=14, F1 win/tie/lose=0 / 14 / 0, mean delta=0.0, selected_gap_rate=0.0714
- target_attribute: count=14, F1 win/tie/lose=0 / 14 / 0, mean delta=0.0, selected_gap_rate=0.0

## By Gap Slot

- <empty>: count=12, F1 win/tie/lose=0 / 12 / 0, mean delta=0.0, selected_gap_rate=0.0
- birthplace: count=6, F1 win/tie/lose=0 / 6 / 0, mean delta=0.0, selected_gap_rate=0.0
- death_date: count=3, F1 win/tie/lose=0 / 3 / 0, mean delta=0.0, selected_gap_rate=0.0
- director: count=14, F1 win/tie/lose=0 / 14 / 0, mean delta=0.0, selected_gap_rate=0.0714
- education: count=2, F1 win/tie/lose=0 / 2 / 0, mean delta=0.0, selected_gap_rate=0.0
- nationality: count=3, F1 win/tie/lose=0 / 3 / 0, mean delta=0.0, selected_gap_rate=0.0

## Top Improvements

None.

## Top Regressions

None.

## Regressions With Selected Gap Candidate

None.
