# Gap Expand Paired Audit

Shared queries: 40

## Overall

- F1 win/tie/lose: 1 / 37 / 2 (mean delta -0.0125)
- EM win/tie/lose: 1 / 38 / 1 (mean delta 0.0)
- Selected-gap subset: 5 queries, F1 win/tie/lose 1 / 2 / 2
- Changed-evidence subset: 6 queries, F1 win/tie/lose 1 / 3 / 2
- Unchanged-evidence subset: 34 queries, F1 win/tie/lose 0 / 34 / 0

## By Bucket

- 2_doc: count=32, F1 win/tie/lose=0 / 31 / 1, mean delta=-0.0156, selected_gap_rate=0.0938
- 4_doc: count=8, F1 win/tie/lose=1 / 6 / 1, mean delta=0.0, selected_gap_rate=0.25

## Gap Trace

- gap_type_distribution: {'target_attribute': 14, 'abstain': 12, 'role_relation': 14}
- gap_slot_distribution: {'death_date': 3, 'director': 14, 'birthplace': 6, 'nationality': 3, 'education': 2}
- fallback_used_count: 12
- selected_gap_count: 5 (rate 0.125)
- by_stop_reason: {'alternate_only_complete': {'count': 7, 'mean_delta_f1': -0.0714, 'win': 1, 'lose': 2}, 'append_cap_reached': {'count': 2, 'mean_delta_f1': 0.0, 'win': 0, 'lose': 0}, 'no_typed_alternate': {'count': 4, 'mean_delta_f1': 0.0, 'win': 0, 'lose': 0}, 'structure_below_threshold': {'count': 26, 'mean_delta_f1': 0.0, 'win': 0, 'lose': 0}, 'typed_alternate_below_threshold': {'count': 1, 'mean_delta_f1': 0.0, 'win': 0, 'lose': 0}}
- top selected-gap titles in regressions: [{'title': 'Roman Polanski', 'count': 2}]
- top selected-gap titles in improvements: [{'title': 'Kenneth Branagh', 'count': 1}]

## By Gap Type

- abstain: count=12, F1 win/tie/lose=0 / 12 / 0, mean delta=0.0, selected_gap_rate=0.0
- role_relation: count=14, F1 win/tie/lose=1 / 11 / 2, mean delta=-0.0357, selected_gap_rate=0.2857
- target_attribute: count=14, F1 win/tie/lose=0 / 14 / 0, mean delta=0.0, selected_gap_rate=0.0714

## By Gap Slot

- <empty>: count=12, F1 win/tie/lose=0 / 12 / 0, mean delta=0.0, selected_gap_rate=0.0
- birthplace: count=6, F1 win/tie/lose=0 / 6 / 0, mean delta=0.0, selected_gap_rate=0.1667
- death_date: count=3, F1 win/tie/lose=0 / 3 / 0, mean delta=0.0, selected_gap_rate=0.0
- director: count=14, F1 win/tie/lose=1 / 11 / 2, mean delta=-0.0357, selected_gap_rate=0.2857
- education: count=2, F1 win/tie/lose=0 / 2 / 0, mean delta=0.0, selected_gap_rate=0.0
- nationality: count=3, F1 win/tie/lose=0 / 3 / 0, mean delta=0.0, selected_gap_rate=0.0

## Top Improvements

- Question: Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Bråk?
  F1 0.0 -> 1.0 (delta 1.0); EM 0.0 -> 1.0 (delta 1.0)
  bucket=4_doc, stop_reason=alternate_only_complete, selected_gap=True, selected_gap_titles=['Kenneth Branagh']
  gap_micro_queries=["Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Bråk?", "Find the director of film and god s gift to women. Question: Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Bråk?"]
  added_titles=['Kenneth Branagh']
  removed_titles=['Ingmar Bergman']

## Top Regressions

- Question: Which film has the director born later, Christ Walking On The Water or 45 Fathers?
  F1 1.0 -> 0.0 (delta -1.0); EM 1.0 -> 0.0 (delta -1.0)
  bucket=4_doc, stop_reason=alternate_only_complete, selected_gap=True, selected_gap_titles=['Roman Polanski']
  gap_micro_queries=['Which film has the director born later, Christ Walking On The Water or 45 Fathers?', 'Find the director of christ walking on the water and 45 fathers. Question: Which film has the director born later, Christ Walking On The Water or 45 Fathers?']
  added_titles=['Roman Polanski']
  removed_titles=['James Tinling']
- Question: What is the place of birth of the director of film The Return Of Swamp Thing?
  F1 0.5 -> 0.0 (delta -0.5); EM 0.0 -> 0.0 (delta 0.0)
  bucket=2_doc, stop_reason=alternate_only_complete, selected_gap=True, selected_gap_titles=['Roman Polanski']
  gap_micro_queries=['What is the place of birth of the director of film The Return Of Swamp Thing?', 'Find the director of the return of swamp thing and swamp thing. Question: What is the place of birth of the director of film The Return Of Swamp Thing?']
  added_titles=['Roman Polanski']
  removed_titles=['Jim Wynorski']

## Regressions With Selected Gap Candidate

- Question: Which film has the director born later, Christ Walking On The Water or 45 Fathers?
  F1 1.0 -> 0.0 (delta -1.0); EM 1.0 -> 0.0 (delta -1.0)
  bucket=4_doc, stop_reason=alternate_only_complete, selected_gap=True, selected_gap_titles=['Roman Polanski']
  gap_micro_queries=['Which film has the director born later, Christ Walking On The Water or 45 Fathers?', 'Find the director of christ walking on the water and 45 fathers. Question: Which film has the director born later, Christ Walking On The Water or 45 Fathers?']
  added_titles=['Roman Polanski']
  removed_titles=['James Tinling']
- Question: What is the place of birth of the director of film The Return Of Swamp Thing?
  F1 0.5 -> 0.0 (delta -0.5); EM 0.0 -> 0.0 (delta 0.0)
  bucket=2_doc, stop_reason=alternate_only_complete, selected_gap=True, selected_gap_titles=['Roman Polanski']
  gap_micro_queries=['What is the place of birth of the director of film The Return Of Swamp Thing?', 'Find the director of the return of swamp thing and swamp thing. Question: What is the place of birth of the director of film The Return Of Swamp Thing?']
  added_titles=['Roman Polanski']
  removed_titles=['Jim Wynorski']
