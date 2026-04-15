# Gap Expand Paired Audit

Shared queries: 40

## Overall

- F1 win/tie/lose: 5 / 27 / 8 (mean delta -0.022)
- EM win/tie/lose: 3 / 32 / 5 (mean delta -0.05)
- Selected-gap subset: 1 queries, F1 win/tie/lose 1 / 0 / 0
- Changed-evidence subset: 29 queries, F1 win/tie/lose 4 / 19 / 6
- Unchanged-evidence subset: 11 queries, F1 win/tie/lose 1 / 8 / 2

## By Bucket

- 2_doc: count=32, F1 win/tie/lose=2 / 23 / 7, mean delta=-0.1124, selected_gap_rate=0.0
- 4_doc: count=8, F1 win/tie/lose=3 / 4 / 1, mean delta=0.3393, selected_gap_rate=0.125

## Gap Trace

- gap_type_distribution: {'target_attribute': 14, 'abstain': 12, 'role_relation': 14}
- gap_slot_distribution: {'death_date': 3, 'director': 14, 'birthplace': 6, 'nationality': 3, 'education': 2}
- fallback_used_count: 12
- selected_gap_count: 1 (rate 0.025)
- by_stop_reason: {'no_eligible_gap_units': {'count': 10, 'mean_delta_f1': -0.0286, 'win': 0, 'lose': 1}, 'structure_below_threshold': {'count': 29, 'mean_delta_f1': -0.055, 'win': 4, 'lose': 7}, 'unit_alternate_added': {'count': 1, 'mean_delta_f1': 1.0, 'win': 1, 'lose': 0}}
- top selected-gap titles in regressions: []
- top selected-gap titles in improvements: [{'title': 'Ingmar Bergman', 'count': 1}]

## By Gap Type

- abstain: count=12, F1 win/tie/lose=1 / 8 / 3, mean delta=-0.2163, selected_gap_rate=0.0
- role_relation: count=14, F1 win/tie/lose=4 / 8 / 2, mean delta=0.1939, selected_gap_rate=0.0714
- target_attribute: count=14, F1 win/tie/lose=0 / 11 / 3, mean delta=-0.0714, selected_gap_rate=0.0

## By Gap Slot

- <empty>: count=12, F1 win/tie/lose=1 / 8 / 3, mean delta=-0.2163, selected_gap_rate=0.0
- birthplace: count=6, F1 win/tie/lose=0 / 5 / 1, mean delta=-0.0555, selected_gap_rate=0.0
- death_date: count=3, F1 win/tie/lose=0 / 2 / 1, mean delta=-0.1111, selected_gap_rate=0.0
- director: count=14, F1 win/tie/lose=4 / 8 / 2, mean delta=0.1939, selected_gap_rate=0.0714
- education: count=2, F1 win/tie/lose=0 / 1 / 1, mean delta=-0.1666, selected_gap_rate=0.0
- nationality: count=3, F1 win/tie/lose=0 / 3 / 0, mean delta=0.0, selected_gap_rate=0.0

## Top Improvements

- Question: Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Bråk?
  F1 0.0 -> 1.0 (delta 1.0); EM 0.0 -> 1.0 (delta 1.0)
  bucket=4_doc, stop_reason=unit_alternate_added, selected_gap=True, selected_gap_titles=['Ingmar Bergman']
  gap_micro_queries=["Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Bråk?", "Find the director of film and god s gift to women. Question: Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Bråk?"]
  added_titles=[]
  removed_titles=[]
- Question: Which film whose director is younger, Dangerously They Live or Salad By The Roots?
  F1 0.0 -> 1.0 (delta 1.0); EM 0.0 -> 1.0 (delta 1.0)
  bucket=4_doc, stop_reason=structure_below_threshold, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=['Which film whose director is younger, Dangerously They Live or Salad By The Roots?', 'Find the director of dangerously they live and salad by the roots. Question: Which film whose director is younger, Dangerously They Live or Salad By The Roots?']
  added_titles=['Georges Lautner']
  removed_titles=['Lasse Hallström']
- Question: Which film whose director was born first, El Tonto or The Heart Of Doreon?
  F1 0.0 -> 1.0 (delta 1.0); EM 0.0 -> 1.0 (delta 1.0)
  bucket=4_doc, stop_reason=structure_below_threshold, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=['Which film whose director was born first, El Tonto or The Heart Of Doreon?', 'Find the director of the heart of doreon and in the heart of doreon. Question: Which film whose director was born first, El Tonto or The Heart Of Doreon?']
  added_titles=['Leopoldo Torre Nilsson']
  removed_titles=['Arne Toonen']
- Question: What nationality is the director of film Blood Street?
  F1 0.0 -> 0.6667 (delta 0.6667); EM 0.0 -> 0.0 (delta 0.0)
  bucket=2_doc, stop_reason=structure_below_threshold, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=['What nationality is the director of film Blood Street?', 'Find the director of film and blood street. Question: What nationality is the director of film Blood Street?']
  added_titles=['Leo Fong', 'Terror Is a Man']
  removed_titles=['Jackie Kong', 'The Camp on Blood Island']
- Question: What is the place of birth of the performer of song Changed It?
  F1 0.5455 -> 0.75 (delta 0.2045); EM 0.0 -> 0.0 (delta 0.0)
  bucket=2_doc, stop_reason=structure_below_threshold, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=['What is the place of birth of the performer of song Changed It?']
  added_titles=['Frank Sinatra']
  removed_titles=['Alicia Keys']

## Top Regressions

- Question: What is the place of birth of Lisbeth Palme's husband?
  F1 1.0 -> 0.0 (delta -1.0); EM 1.0 -> 0.0 (delta -1.0)
  bucket=2_doc, stop_reason=structure_below_threshold, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=["What is the place of birth of Lisbeth Palme's husband?"]
  added_titles=[]
  removed_titles=[]
- Question: Who lived longer, Ludwig Elsbett or Pamela Ann Rymer?
  F1 1.0 -> 0.0 (delta -1.0); EM 1.0 -> 0.0 (delta -1.0)
  bucket=2_doc, stop_reason=structure_below_threshold, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=['Who lived longer, Ludwig Elsbett or Pamela Ann Rymer?']
  added_titles=['Pamela A. Barker']
  removed_titles=['Carol Rymer Davis']
- Question: Who is the father-in-law of Sisowath Kossamak?
  F1 0.8 -> 0.0 (delta -0.8); EM 0.0 -> 0.0 (delta 0.0)
  bucket=2_doc, stop_reason=structure_below_threshold, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=['Who is the father-in-law of Sisowath Kossamak?']
  added_titles=['Sirikit']
  removed_titles=['Ogawa Mataji']
- Question: Where was the director of film The Private Life Of Cinema born?
  F1 0.6667 -> 0.0 (delta -0.6667); EM 0.0 -> 0.0 (delta 0.0)
  bucket=2_doc, stop_reason=structure_below_threshold, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=['Where was the director of film The Private Life Of Cinema born?', 'Find the director of the private life of cinema and alexander korda. Question: Where was the director of film The Private Life Of Cinema born?']
  added_titles=['The Private Life of Louis XIV', 'Diane Kurys']
  removed_titles=['Kim Ki-young', 'Roger Corman']
- Question: Where did Theodore Salisbury Woolsey's father study?
  F1 1.0 -> 0.6667 (delta -0.3333); EM 1.0 -> 0.0 (delta -1.0)
  bucket=2_doc, stop_reason=structure_below_threshold, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=["Where did Theodore Salisbury Woolsey's father study?", "Find where theodore dwight woolsey studied. Question: Where did Theodore Salisbury Woolsey's father study?"]
  added_titles=['George Lessey']
  removed_titles=['Edward Wingfield, 2nd Viscount Powerscourt']
- Question: Where was the composer of film Billy Elliot born?
  F1 1.0 -> 0.6667 (delta -0.3333); EM 1.0 -> 0.0 (delta -1.0)
  bucket=2_doc, stop_reason=structure_below_threshold, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=['Where was the composer of film Billy Elliot born?', 'Find the birthplace of stephen warbeck. Question: Where was the composer of film Billy Elliot born?']
  added_titles=[]
  removed_titles=[]
- Question: Why did John Middleton Murry's wife die?
  F1 1.0 -> 0.6667 (delta -0.3333); EM 1.0 -> 0.0 (delta -1.0)
  bucket=2_doc, stop_reason=structure_below_threshold, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=["Why did John Middleton Murry's wife die?", "Find when katherine mansfield died. Question: Why did John Middleton Murry's wife die?"]
  added_titles=['J. W. N. Sullivan']
  removed_titles=['Eddie Murray (rugby league)']
- Question: Which film has the director who was born later, Playing It Wild or I'Ll Be Going Now?
  F1 0.5714 -> 0.2857 (delta -0.2857); EM 0.0 -> 0.0 (delta 0.0)
  bucket=4_doc, stop_reason=no_eligible_gap_units, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=["Which film has the director who was born later, Playing It Wild or I'Ll Be Going Now?", "Find the director of playing it wild. Question: Which film has the director who was born later, Playing It Wild or I'Ll Be Going Now?"]
  added_titles=['Philip Ford (film director)']
  removed_titles=['Roman Polanski']

## Regressions With Selected Gap Candidate

None.
