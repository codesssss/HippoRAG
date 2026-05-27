# V13B Gold OpenIE Schema Gap Summary

This report is diagnostic-only. It summarizes gold OpenIE audit outputs and does not change retrieval, graph construction, matching, or fallback behavior.

Counting note: status counts are obligation counts; relation mismatch tables count endpoint-hit fact relations, so they can exceed the number of audited obligations. Endpoint shape tables use representative cases retained by the audit reports.

## Status By Dataset

| dataset | audited | candidate_missing | gold_endpoint_missing | gold_endpoint_relation_mismatch | gold_exact_openie_match_available | gold_relation_present_without_endpoint_binding |
|---|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 71 | 1 | 2 | 65 | 1 | 2 |
| hotpotqa | 98 | 0 | 40 | 45 | 1 | 12 |
| musique | 161 | 12 | 55 | 90 | 0 | 4 |

## Interpretation

| item | value |
|---|---|
| main_bottleneck | qwen_openie_schema_alignment |
| candidate_missing_fraction | 0.039394 |
| endpoint_relation_mismatch_fraction | 0.606061 |
| endpoint_missing_fraction | 0.293939 |
| exact_available_fraction | 0.006061 |
| top_schema_gap_kind | lead_sentence_attribute_not_materialized |
| recommended_next_step | Define a corpus/query evidence-frame contract and test Qwen OpenIE against it; do not tune selector parameters yet. |

## Schema Gap Kinds

| gap_kind | count |
|---|---:|
| cross_family_relation_mismatch | 745 |
| same_family_relation_mismatch | 249 |
| lead_sentence_attribute_not_materialized | 189 |

## Relation Mismatch By Query Family

| query_relation_family | count |
|---|---:|
| temporal_attribute | 457 |
| other_relation | 303 |
| place_or_origin_attribute | 230 |
| event_or_competition_role | 124 |
| office_or_position_role | 40 |
| work_metadata_role | 29 |

## Relation Family Matrix

| query_family -> gold_family | count |
|---|---:|
| other_relation -> lead_or_untyped_fact | 198 |
| temporal_attribute -> other_relation | 163 |
| temporal_attribute -> lead_or_untyped_fact | 135 |
| place_or_origin_attribute -> other_relation | 105 |
| temporal_attribute -> temporal_attribute | 71 |
| event_or_competition_role -> other_relation | 70 |
| other_relation -> other_relation | 57 |
| place_or_origin_attribute -> lead_or_untyped_fact | 54 |
| place_or_origin_attribute -> place_or_origin_attribute | 51 |
| temporal_attribute -> place_or_origin_attribute | 36 |
| event_or_competition_role -> event_or_competition_role | 30 |
| temporal_attribute -> work_metadata_role | 29 |
| other_relation -> place_or_origin_attribute | 29 |
| temporal_attribute -> person_relation_role | 23 |
| work_metadata_role -> work_metadata_role | 22 |
| office_or_position_role -> office_or_position_role | 18 |
| event_or_competition_role -> place_or_origin_attribute | 15 |
| office_or_position_role -> place_or_origin_attribute | 14 |
| other_relation -> work_metadata_role | 13 |
| event_or_competition_role -> lead_or_untyped_fact | 9 |
| place_or_origin_attribute -> person_relation_role | 9 |
| office_or_position_role -> lead_or_untyped_fact | 8 |
| work_metadata_role -> other_relation | 7 |
| place_or_origin_attribute -> temporal_attribute | 6 |
| other_relation -> temporal_attribute | 6 |
| place_or_origin_attribute -> work_metadata_role | 5 |

## Top Relation Mismatch Pairs

| query_relation -> gold_relation | count |
|---|---:|
| died_on -> <empty> | 43 |
| born_in -> <empty> | 40 |
| competition -> draft | 30 |
| competition -> includ | 30 |
| locat_in -> <empty> | 30 |
| competition -> attend | 29 |
| born_on -> <empty> | 27 |
| locat_in -> locat_in | 20 |
| born_in -> occurr_dur | 20 |
| creat -> <empty> | 20 |
| died_in -> occurr_dur | 20 |
| occurr_in -> occurr_dur | 20 |
| network -> <empty> | 19 |
| radio_division -> <empty> | 19 |
| agre -> <empty> | 18 |
| involv -> <empty> | 17 |
| originat -> <empty> | 17 |
| born_on -> direct | 16 |
| died_in -> <empty> | 16 |
| from -> <empty> | 16 |
| competition -> locat_in | 15 |
| born_in -> predecessor | 14 |
| died_on -> neighbor | 14 |
| born_in -> cast_includ | 13 |
| born_in -> criticiz | 13 |
| composer -> cast_includ | 13 |
| born_in -> compos | 12 |
| co_official_language -> spoken_in | 12 |
| died_on -> discuss | 12 |
| born_on -> died_on | 11 |
| establish_in -> <empty> | 11 |
| locat_in -> part | 11 |
| tallest_build -> <empty> | 11 |
| born_on -> born_on | 10 |
| born_on -> released_on | 10 |
| born_on -> starr | 10 |
| died_in -> made | 10 |
| born_in -> occurr_in | 10 |
| died_in -> occurr_in | 10 |
| died_on -> influenc | 10 |

## Representative Endpoint Shapes

| endpoint_shape | representative_case_count |
|---|---:|
| all_variable_obligation | 15 |
| named_endpoint | 15 |
| descriptive_bound_endpoint | 10 |
| long_bound_endpoint | 2 |
| title_qualified_endpoint | 1 |
| typed_title_endpoint | 1 |

## Endpoint Shape Examples

### all_variable_obligation
- dataset=2wikimultihopqa q=33 status=candidate_missing endpoints=[] triple=['?x1', 'was born in', '?place']
- dataset=2wikimultihopqa q=34 status=gold_endpoint_missing endpoints=[] triple=['?x1', 'works at', '?place']
- dataset=2wikimultihopqa q=14 status=gold_endpoint_relation_mismatch endpoints=[] triple=['?x1', 'died in', '?place']
- dataset=2wikimultihopqa q=15 status=gold_endpoint_relation_mismatch endpoints=[] triple=['?x1', 'is from', '?country']
- dataset=2wikimultihopqa q=20 status=gold_endpoint_relation_mismatch endpoints=[] triple=['?x1', 'studied at', '?place']

### descriptive_bound_endpoint
- dataset=hotpotqa q=3 status=gold_endpoint_missing endpoints=['recently abdicated queen'] triple=['?x1', 'helped', 'recently abdicated queen']
- dataset=hotpotqa q=3 status=gold_endpoint_missing endpoints=['recently abdicated queen'] triple=['recently abdicated queen', 'imprisoned by', '?x2']
- dataset=hotpotqa q=9 status=gold_endpoint_missing endpoints=['The 2000 ICC KnockOut Trophy'] triple=['The 2000 ICC KnockOut Trophy', 'debut of', '?x1']
- dataset=hotpotqa q=24 status=gold_endpoint_missing endpoints=['This Experts Network sports analysts'] triple=['This Experts Network sports analysts', 'played in NFL for', '?x1']
- dataset=hotpotqa q=14 status=gold_endpoint_relation_mismatch endpoints=['The Secret of Kells'] triple=['The Secret of Kells', 'country of origin', '?x2']

### long_bound_endpoint
- dataset=hotpotqa q=24 status=gold_endpoint_missing endpoints=['Pro Football Hall of Fame', '2000'] triple=['Pro Football Hall of Fame', 'induction year', '2000']
- dataset=hotpotqa q=33 status=gold_endpoint_missing endpoints=['MGM Grand Garden Special Events Center'] triple=['?x1', 'is a former', 'MGM Grand Garden Special Events Center']

### named_endpoint
- dataset=2wikimultihopqa q=23 status=gold_endpoint_missing endpoints=['Lisbeth Palme'] triple=['Lisbeth Palme', 'husband', '?x1']
- dataset=2wikimultihopqa q=1 status=gold_endpoint_relation_mismatch endpoints=['Aas Ka Panchhi'] triple=['Aas Ka Panchhi', 'release date', '?date1']
- dataset=2wikimultihopqa q=1 status=gold_endpoint_relation_mismatch endpoints=['Phoolwari'] triple=['Phoolwari', 'release date', '?date2']
- dataset=2wikimultihopqa q=7 status=gold_endpoint_relation_mismatch endpoints=['Raghnall Mac Ruaidhrí'] triple=['Raghnall Mac Ruaidhrí', 'paternal grandfather', '?x1']
- dataset=2wikimultihopqa q=21 status=gold_endpoint_relation_mismatch endpoints=['John Middleton Murry'] triple=['John Middleton Murry', 'wife', '?x1']

### title_qualified_endpoint
- dataset=2wikimultihopqa q=15 status=gold_endpoint_relation_mismatch endpoints=['Aleksander Koniecpolski (1620–1659)'] triple=['Aleksander Koniecpolski (1620–1659)', 'father', '?x1']

### typed_title_endpoint
- dataset=2wikimultihopqa q=22 status=gold_endpoint_relation_mismatch endpoints=['film Billy Elliot'] triple=['film Billy Elliot', 'composer', '?x1']
