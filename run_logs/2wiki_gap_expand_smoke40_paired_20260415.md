# Gap Expand Paired Audit

Shared queries: 40

## Overall

- F1 win/tie/lose: 2 / 34 / 4 (mean delta -0.0587)
- EM win/tie/lose: 0 / 37 / 3 (mean delta -0.075)
- Selected-gap subset: 22 queries, F1 win/tie/lose 0 / 20 / 2
- Changed-evidence subset: 19 queries, F1 win/tie/lose 0 / 17 / 2
- Unchanged-evidence subset: 21 queries, F1 win/tie/lose 2 / 17 / 2

## By Bucket

- 2_doc: count=32, F1 win/tie/lose=1 / 30 / 1, mean delta=-0.0081, selected_gap_rate=0.5625
- 4_doc: count=8, F1 win/tie/lose=1 / 4 / 3, mean delta=-0.261, selected_gap_rate=0.5

## Gap Trace

- gap_type_distribution: {'linking_relation': 40}
- fallback_used_count: 0
- selected_gap_count: 22 (rate 0.55)
- by_stop_reason: {'append_cap_reached': {'count': 11, 'mean_delta_f1': -0.0909, 'win': 0, 'lose': 1}, 'structure_below_threshold': {'count': 29, 'mean_delta_f1': -0.0465, 'win': 2, 'lose': 3}}
- top selected-gap titles in regressions: [{'title': 'Aditya Chopra', 'count': 1}, {'title': 'Roman Polanski', 'count': 1}, {'title': 'Jeethu Joseph', 'count': 1}]
- top selected-gap titles in improvements: []

## Top Improvements

- Question: Where did Coulson Wallop's father study?
  F1 0.0 -> 0.6667 (delta 0.6667); EM 0.0 -> 0.0 (delta 0.0)
  bucket=2_doc, stop_reason=structure_below_threshold, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=["Where did Coulson Wallop's father study?", "Find the linking relation between earl of portsmouth and 19 september 1774. Question: Where did Coulson Wallop's father study?"]
  added_titles=[]
  removed_titles=[]
- Question: Which film whose director was born first, El Tonto or The Heart Of Doreon?
  F1 0.0 -> 0.2609 (delta 0.2609); EM 0.0 -> 0.0 (delta 0.0)
  bucket=4_doc, stop_reason=structure_below_threshold, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=['Which film whose director was born first, El Tonto or The Heart Of Doreon?', 'Find the linking relation between 1921 american silent short western romantic drama film and in the heart of doreon. Question: Which film whose director was born first, El Tonto or The Heart Of Doreon?']
  added_titles=[]
  removed_titles=[]

## Top Regressions

- Question: Do both directors of films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?
  F1 1.0 -> 0.0 (delta -1.0); EM 1.0 -> 0.0 (delta -1.0)
  bucket=4_doc, stop_reason=structure_below_threshold, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=['Do both directors of films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?', 'Find the linking relation between clio barnard and dark river. Question: Do both directors of films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?']
  added_titles=[]
  removed_titles=[]
- Question: Which film has the director born later, Christ Walking On The Water or 45 Fathers?
  F1 1.0 -> 0.0 (delta -1.0); EM 1.0 -> 0.0 (delta -1.0)
  bucket=4_doc, stop_reason=append_cap_reached, selected_gap=True, selected_gap_titles=['Roman Polanski', 'Jeethu Joseph']
  gap_micro_queries=['Which film has the director born later, Christ Walking On The Water or 45 Fathers?', 'Find the linking relation between christ walking on the water and georges m li s. Question: Which film has the director born later, Christ Walking On The Water or 45 Fathers?']
  added_titles=['Roman Polanski', 'Jeethu Joseph']
  removed_titles=['James Tinling', 'Benjamin Christensen']
- Question: Where did Theodore Salisbury Woolsey's father study?
  F1 1.0 -> 0.0741 (delta -0.9259); EM 1.0 -> 0.0 (delta -1.0)
  bucket=2_doc, stop_reason=structure_below_threshold, selected_gap=False, selected_gap_titles=[]
  gap_micro_queries=["Where did Theodore Salisbury Woolsey's father study?", "Find the linking relation between theodore salisbury woolsey and theodore dwight woolsey. Question: Where did Theodore Salisbury Woolsey's father study?"]
  added_titles=[]
  removed_titles=[]
- Question: Which film has the director who was born later, Playing It Wild or I'Ll Be Going Now?
  F1 0.5714 -> 0.2222 (delta -0.3492); EM 0.0 -> 0.0 (delta 0.0)
  bucket=4_doc, stop_reason=structure_below_threshold, selected_gap=True, selected_gap_titles=['Aditya Chopra']
  gap_micro_queries=["Which film has the director who was born later, Playing It Wild or I'Ll Be Going Now?", "Find the linking relation between 1923 american silent western film and playing it wild. Question: Which film has the director who was born later, Playing It Wild or I'Ll Be Going Now?"]
  added_titles=['Aditya Chopra']
  removed_titles=['Ildikó Enyedi']

## Regressions With Selected Gap Candidate

- Question: Which film has the director born later, Christ Walking On The Water or 45 Fathers?
  F1 1.0 -> 0.0 (delta -1.0); EM 1.0 -> 0.0 (delta -1.0)
  bucket=4_doc, stop_reason=append_cap_reached, selected_gap=True, selected_gap_titles=['Roman Polanski', 'Jeethu Joseph']
  gap_micro_queries=['Which film has the director born later, Christ Walking On The Water or 45 Fathers?', 'Find the linking relation between christ walking on the water and georges m li s. Question: Which film has the director born later, Christ Walking On The Water or 45 Fathers?']
  added_titles=['Roman Polanski', 'Jeethu Joseph']
  removed_titles=['James Tinling', 'Benjamin Christensen']
- Question: Which film has the director who was born later, Playing It Wild or I'Ll Be Going Now?
  F1 0.5714 -> 0.2222 (delta -0.3492); EM 0.0 -> 0.0 (delta 0.0)
  bucket=4_doc, stop_reason=structure_below_threshold, selected_gap=True, selected_gap_titles=['Aditya Chopra']
  gap_micro_queries=["Which film has the director who was born later, Playing It Wild or I'Ll Be Going Now?", "Find the linking relation between 1923 american silent western film and playing it wild. Question: Which film has the director who was born later, Playing It Wild or I'Ll Be Going Now?"]
  added_titles=['Aditya Chopra']
  removed_titles=['Ildikó Enyedi']
