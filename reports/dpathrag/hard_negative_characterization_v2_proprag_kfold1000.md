# D-PathRAG Hard-Negative Characterization v2

- Rows: `1000`
- Top-k: `5`
- Max candidates: `100`
- Oracle pool size: `20`
- Semantic hard-negative threshold: `0.293276`

## CEE Feasibility Decision

- Recommendation: **CEE recommended**
- Oracle best support-complete gain: `+0.1280`
- Oracle Edit@2 non-gold per gold: `0.0`
- Dominant hard-negative type share: `0.5419`
- Over-edited complete queries: `649`

## Rank Bucket Distribution

| Category | Bucket | Docs | Queries | q_doc_cosine | answer_in_doc | bridge_entity_in_doc | question_coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| selector_added_non_gold | 6-10 | 633 | 539 | 0.2255 | 0.0395 | 0.0569 | 0.3348 |
| selector_added_non_gold | 11-20 | 265 | 252 | 0.2008 | 0.0340 | 0.0340 | 0.3308 |
| selector_added_non_gold | 21-50 | 245 | 193 | 0.1045 | 0.0367 | 0.0286 | 0.2829 |
| selector_added_non_gold | 51-100 | 62 | 52 | 0.1084 | 0.0968 | 0.0968 | 0.2508 |
| selector_added_gold | 6-10 | 54 | 54 | 0.2728 | 0.5185 | 0.8148 | 0.3599 |
| selector_added_gold | 11-20 | 20 | 20 | 0.2269 | 0.6000 | 0.6500 | 0.3284 |
| selector_added_gold | 21-50 | 1 | 1 | 0.3332 | 0.0000 | 0.0000 | 0.3333 |
| selector_added_gold | 51-100 | 1 | 1 | 0.4259 | 0.0000 | 1.0000 | 0.5714 |
| rank_removed_non_gold | 6-10 | 0 | 0 |  |  |  |  |
| rank_removed_non_gold | 11-20 | 0 | 0 |  |  |  |  |
| rank_removed_non_gold | 21-50 | 0 | 0 |  |  |  |  |
| rank_removed_non_gold | 51-100 | 0 | 0 |  |  |  |  |

## Hard-Negative Type Breakdown

| Type | Docs | Queries | Mean Rank | q_doc_cosine | answer_in_doc | bridge_entity_in_doc | question_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| answer_string_distractor | 49 | 41 | 22.6531 | 0.2112 | 1.0000 | 0.2041 | 0.3798 |
| bridge_entity_distractor | 48 | 43 | 19.2708 | 0.1014 | 0.0000 | 1.0000 | 0.3115 |
| lexical_hard_negative | 653 | 501 | 14.8683 | 0.2294 | 0.0000 | 0.0000 | 0.4098 |
| semantic_hard_negative | 68 | 65 | 11.3382 | 0.3735 | 0.0000 | 0.0000 | 0.2049 |
| low_signal_deep_negative | 387 | 296 | 23.7339 | 0.0979 | 0.0000 | 0.0000 | 0.1792 |

## Query-Level Failure Buckets

| Bucket | Queries | Rank Complete Rate | Selector Changed Rate | Avg Added Gold | Avg Added Non-Gold | Avg Support Complete Delta |
|---|---:|---:|---:|---:|---:|---:|
| rank_complete_and_selector_changed | 649 | 1.0000 | 1.0000 | 0.0000 | 1.5300 | -0.0524 |
| rank_incomplete_and_selector_added_gold | 76 | 0.0000 | 1.0000 | 1.0000 | 0.5789 | +0.8816 |
| rank_incomplete_but_selector_added_only_non_gold | 121 | 0.0000 | 1.0000 | 0.0000 | 1.3884 | +0.0000 |
| selector_improves_support_complete | 67 | 0.0000 | 1.0000 | 1.0000 | 0.5821 | +1.0000 |
| selector_neutral_support_complete | 899 | 0.8209 | 0.8287 | 0.0100 | 1.2380 | +0.0000 |
| selector_hurts_support_complete | 34 | 1.0000 | 1.0000 | 0.0000 | 1.5588 | -1.0000 |

## Oracle Edit Opportunity

| Variant | Support Recall | Support Complete | Selected Gold | Bridge Recall | Queries With Edit | Stop Rate | Added Gold | Added Non-Gold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rank_topk | 0.9028 | 0.7720 | 2.2270 | 0.9365 |  |  |  |  |
| selector_v1 | 0.9233 | 0.8050 | 2.2660 | 0.9416 |  |  |  |  |
| oracle_edit1 | 0.9593 | 0.8950 | 2.3630 | 0.9510 | 136 | 0.864 | 136 | 0 |
| oracle_edit2 | 0.9607 | 0.9000 | 2.3680 | 0.9510 | 136 | 0.995 | 141 | 0 |

## Gate Checks

| Gate | Pass |
|---|---:|
| oracle_support_complete_gain_ge_2pp | True |
| oracle_import_ratio_3x_better_than_v1 | True |
| dominant_hard_negative_type_ge_30pct | True |
| over_edit_rate_measurable | True |
