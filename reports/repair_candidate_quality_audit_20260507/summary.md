# Repair Candidate Quality Audit

This is an offline diagnostic. It uses gold support titles only to locate where SetR-missing support becomes visible in the fixed PropRAG pool and DBEC trace artifacts. It makes no reader or LLM calls.

## Stage Visibility

| Dataset | Q | Missing gold | Source all | rank_fill5 recover | binding-candidate recover | DBEC greedy recover | DBEC final recover | DBEC non-gold adds | rank gold replaced | near-title query | duplicate query | dF1 DBEC-SetR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 122 | 182 | 91.8% | 67.0% | 74.7% | 73.6% | 83.0% | 1.24 | 0.09 | 1.6% | 2.5% | 0.2851 |
| HotpotQA | 32 | 32 | 100.0% | 78.1% | 21.9% | 43.8% | 84.4% | 3.12 | 0.00 | 18.8% | 6.2% | 0.1091 |
| MuSiQue | 106 | 204 | 82.1% | 32.4% | 4.4% | 25.0% | 37.7% | 2.30 | 0.04 | 7.5% | 33.0% | 0.0819 |

## Missing-Gold Buckets

| Dataset | Bucket | Missing titles | Rate | dF1 DBEC-SetR |
|---|---|---:|---:|---:|
| 2Wiki | gold_recovered_by_dbec_final | 151 | 83.0% | 0.3105 |
| 2Wiki | gold_in_binding_candidates_not_selected | 17 | 9.3% | 0.4118 |
| 2Wiki | gold_absent_from_source_pool | 10 | 5.5% | 0.2000 |
| 2Wiki | gold_source_only_not_rank_or_dbec | 3 | 1.6% | 0.6667 |
| 2Wiki | gold_in_rank_fill_not_dbec | 1 | 0.5% | 0.0000 |
| HotpotQA | gold_recovered_by_dbec_final | 27 | 84.4% | 0.0843 |
| HotpotQA | gold_source_only_not_rank_or_dbec | 5 | 15.6% | 0.2431 |
| MuSiQue | gold_source_only_not_rank_or_dbec | 96 | 47.1% | 0.0167 |
| MuSiQue | gold_recovered_by_dbec_final | 77 | 37.7% | 0.1948 |
| MuSiQue | gold_absent_from_source_pool | 25 | 12.3% | 0.1000 |
| MuSiQue | gold_in_rank_fill_not_dbec | 4 | 2.0% | 0.0250 |
| MuSiQue | gold_in_binding_candidates_not_selected | 2 | 1.0% | 0.2500 |

## Query Primary Buckets

| Dataset | Primary bucket | Q | Rate | missing/query | rank recover | binding recover | DBEC recover | replaced rank gold | dF1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | all_missing_gold_recovered_by_dbec | 92 | 75.4% | 1.42 | 1.02 | 1.14 | 1.42 | 0.00 | 0.2585 |
| 2Wiki | gold_in_binding_candidates_not_selected | 17 | 13.9% | 1.76 | 1.24 | 1.59 | 0.71 | 0.65 | 0.4118 |
| 2Wiki | gold_absent_from_source_pool | 10 | 8.2% | 1.50 | 0.40 | 0.20 | 0.50 | 0.00 | 0.2000 |
| 2Wiki | gold_source_only_not_rank_or_dbec | 3 | 2.5% | 2.00 | 1.00 | 0.67 | 1.00 | 0.00 | 0.6667 |
| HotpotQA | all_missing_gold_recovered_by_dbec | 27 | 84.4% | 1.00 | 0.93 | 0.26 | 1.00 | 0.00 | 0.0843 |
| HotpotQA | gold_source_only_not_rank_or_dbec | 5 | 15.6% | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.2431 |
| MuSiQue | gold_source_only_not_rank_or_dbec | 62 | 58.5% | 1.97 | 0.42 | 0.05 | 0.48 | 0.02 | 0.0403 |
| MuSiQue | all_missing_gold_recovered_by_dbec | 26 | 24.5% | 1.54 | 1.35 | 0.08 | 1.54 | 0.00 | 0.2147 |
| MuSiQue | gold_absent_from_source_pool | 15 | 14.2% | 2.33 | 0.13 | 0.27 | 0.40 | 0.00 | 0.1000 |
| MuSiQue | gold_in_rank_fill_not_dbec | 3 | 2.8% | 2.33 | 1.00 | 0.00 | 0.33 | 1.00 | -0.3000 |

## Source Rank Depth By Missing-Gold Bucket

| Dataset | Bucket | source-visible titles | mean rank | p50 | p75 | p90 | max |
|---|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | gold_absent_from_source_pool | 0 |  |  |  |  |  |
| 2Wiki | gold_in_binding_candidates_not_selected | 17 | 16.0 | 4 | 17 | 44 | 80 |
| 2Wiki | gold_in_rank_fill_not_dbec | 1 |  | 0 | 0 | 0 | 0 |
| 2Wiki | gold_recovered_by_dbec_final | 151 | 8.7 | 2 | 5 | 27 | 87 |
| 2Wiki | gold_source_only_not_rank_or_dbec | 3 | 29.3 | 25 | 57 | 57 | 57 |
| HotpotQA | gold_recovered_by_dbec_final | 27 | 2.4 | 1 | 2 | 4 | 23 |
| HotpotQA | gold_source_only_not_rank_or_dbec | 5 | 24.6 | 14 | 30 | 62 | 62 |
| MuSiQue | gold_absent_from_source_pool | 0 |  |  |  |  |  |
| MuSiQue | gold_in_binding_candidates_not_selected | 2 | 7.0 | 5 | 9 | 9 | 9 |
| MuSiQue | gold_in_rank_fill_not_dbec | 4 | 3.0 | 4 | 4 | 4 | 4 |
| MuSiQue | gold_recovered_by_dbec_final | 77 | 6.2 | 1 | 3 | 14 | 80 |
| MuSiQue | gold_source_only_not_rank_or_dbec | 96 | 36.0 | 27 | 58 | 79 | 96 |

## MuSiQue Examples

- q805 `all_missing_gold_recovered_by_dbec` dF1=-1.0000
  - Q: Where do Greyhound buses leave from in the city where the band that recorded the album Never Too Loud formed?
  - missing: `["Never Too Loud"]`
  - rank_fill5: `["Danko Jones", "Toronto Coach Terminal", "Never Too Loud", "Philadelphia", "Alvarado Transportation Center"]`
  - DBEC: `["New York City", "Never Too Loud", "The Beatles", "Philadelphia", "Saint Paul, Minnesota"]`
  - binding candidates: `[]`
- q503 `all_missing_gold_recovered_by_dbec` dF1=0.0000
  - Q: What is the administrative territorial entity that contains the location of Eric Marcus Municipal Airport PBS station?
  - missing: `["Pima County Natural Resources, Parks and Recreation"]`
  - rank_fill5: `["Eric Marcus Municipal Airport", "Tucson, Arizona", "LaBelle Municipal Airport", "Fremont Municipal Airport (Michigan)", "Pima County Natural Resources, Parks and Recreation"]`
  - DBEC: `["Eric Marcus Municipal Airport", "ISO 3166-2:AD", "LaBelle Municipal Airport", "Fremont Municipal Airport (Michigan)", "Pima County Natural Resources, Parks and Recreation"]`
  - binding candidates: `[]`
- q210 `all_missing_gold_recovered_by_dbec` dF1=-0.0000
  - Q: When did the rx 350 model of the luxury division of the company that built Daihatsu boon change body style?
  - missing: `["1973 oil crisis"]`
  - rank_fill5: `["Lexus RX", "Daihatsu Boon", "1973 oil crisis", "Toyota Racing Development", "Toyota Matrix"]`
  - DBEC: `["Genesis Motor", "Daihatsu Boon", "Lexus RX", "1973 oil crisis", "Toyota Racing Development"]`
  - binding candidates: `[]`
- q83 `all_missing_gold_recovered_by_dbec` dF1=0.0000
  - Q: How long are the city council terms in the second largest city in the state where Yuma is located?
  - missing: `["Yuma, Colorado", "Yuma County Library District"]`
  - rank_fill5: `["Tucson, Arizona", "Tucson, Arizona", "Yuma, Colorado", "Yuma County Library District", "Delta, Colorado"]`
  - DBEC: `["Arizona", "Tucson, Arizona", "Saint Paul, Minnesota", "Yuma, Colorado", "Yuma County Library District"]`
  - binding candidates: `[]`
- q438 `all_missing_gold_recovered_by_dbec` dF1=0.0000
  - Q: The leader visiting where Steven Spielberg's grandparents are from met with whom on November 22?
  - missing: `["Dissolution of the Soviet Union"]`
  - rank_fill5: `["Steven Spielberg", "Dissolution of the Soviet Union", "Dissolution of the Soviet Union", "Vyshnivets Palace", "Pidhirtsi Castle"]`
  - DBEC: `["Balfour Declaration", "Steven Spielberg", "Australia Day 2012 protests", "Dissolution of the Soviet Union", "Dissolution of the Soviet Union"]`
  - binding candidates: `[]`
- q545 `all_missing_gold_recovered_by_dbec` dF1=0.0000
  - Q: In what region of the country where Lam Dong is located is John Phan's birthplace?
  - missing: `["South Central Coast"]`
  - rank_fill5: `["John Phan", "Lâm Đồng Province", "South Central Coast", "Khánh Hòa Province", "Kon Tum Province"]`
  - DBEC: `["Central Highlands (Tasmania)", "Lâm Đồng Province", "John Phan", "South Central Coast", "Khánh Hòa Province"]`
  - binding candidates: `["Lâm Đồng Province", "Khánh Hòa Province", "Central Highlands (Tasmania)"]`
- q559 `all_missing_gold_recovered_by_dbec` dF1=0.0000
  - Q: What was Nintendo's limit on games per developer per year on the platform, also known by a three letter abbreviation, of the video game Xexyz?
  - missing: `["Nintendo Entertainment System"]`
  - rank_fill5: `["Xexyz", "Super Nintendo Entertainment System", "Nintendo Entertainment System", "Nintendo Entertainment System", "Philips CD-i"]`
  - DBEC: `["Xbox 360", "Nintendo La Rivista Ufficiale", "Super Nintendo Entertainment System", "Nintendo Entertainment System", "Nintendo Entertainment System"]`
  - binding candidates: `["Xbox 360"]`
- q651 `all_missing_gold_recovered_by_dbec` dF1=0.0000
  - Q: What was the 1900 population of the city located within the county that also contains Helvetia?
  - missing: `["Pima County Natural Resources, Parks and Recreation"]`
  - rank_fill5: `["Helvetia, Arizona", "Tucson, Arizona", "Crescent City, California", "Delta, Colorado", "Elmo, Montana"]`
  - DBEC: `["Pima County Natural Resources, Parks and Recreation", "Helvetia, Arizona", "Crescent City, California", "Tucson, Arizona", "Delta, Colorado"]`
  - binding candidates: `["Pima County Natural Resources, Parks and Recreation"]`
- q7 `gold_absent_from_source_pool` dF1=0.0000
  - Q: When did the explorer reach the city where the headquarters of the only group larger than Vilaiyaadu Mankatha's record label is located?
  - missing: `["The Right Stuff Records", "Santa Monica, California"]`
  - rank_fill5: `["Vilaiyaadu Mankatha", "Sony Music", "Miami", "Jamnagar district", "Richmond, Virginia"]`
  - DBEC: `["Kathmandu", "Scratchie Records", "Francisco de Orellana", "The Right Stuff Records", "The Great Lakes Group"]`
  - binding candidates: `[]`
- q150 `gold_absent_from_source_pool` dF1=0.0000
  - Q: Normalization occurred in Country A that invaded Country B because the military branch was unprepared. Country B was the only communist country to have an embassy where?
  - missing: `["Police", "Prague underground (culture)"]`
  - rank_fill5: `["Josip Broz Tito", "Josip Broz Tito", "Southern Europe", "John Kerry", "1937 South American Championship"]`
  - DBEC: `["Josip Broz Tito", "Declarations of war during World War II", "John Kerry", "1937 South American Championship", "Guarani alphabet"]`
  - binding candidates: `[]`
- q723 `gold_absent_from_source_pool` dF1=0.0000
  - Q: When did the city where Souvenir's performer was born become the capitol of the state Knowles was from?
  - missing: `["Live from Austin, TX (Eric Johnson album)", "Suga Mama", "History of Austin, Texas", "Souvenir (Eric Johnson album)"]`
  - rank_fill5: `["Beyoncé", "Savannah, Georgia", "Tallahassee, Florida", "History of Georgia (U.S. state)", "Destiny's Child"]`
  - DBEC: `["History of Georgia (U.S. state)", "Alexandre de Lesseps", "Beyoncé", "Destiny's Child", "List of municipalities in Georgia (U.S. state)"]`
  - binding candidates: `[]`
- q902 `gold_absent_from_source_pool` dF1=-1.0000
  - Q: What is the capital of the county that shares a border with the county where Don Werner was born?
  - missing: `["Jerome Quinn", "John C. Petersen", "Pulaski High School"]`
  - rank_fill5: `["Don Werner", "Oklahoma City", "J. P. Hayes", "Anne Simonett", "Erik Jensen (American football)"]`
  - DBEC: `["Baranya County", "Don Werner", "Oklahoma City", "J. P. Hayes", "Anne Simonett"]`
  - binding candidates: `["Baranya County"]`

## Interpretation

- If `gold_in_binding_candidates_not_selected` dominates, the next method change should target scoring/aggregation rather than chain-walking candidate generation.
- If `gold_in_rank_fill_not_dbec` or `gold_source_only_not_rank_or_dbec` dominates, DBEC's current binding-derived candidate path is missing reachable support; chain-aware binding or candidate expansion is justified.
- If `gold_absent_from_source_pool` is large, the fixed-pool setup is the ceiling and iterative retrieval would be a different method line.

## Files

- Missing-gold rows: `reports/repair_candidate_quality_audit_20260507/missing_gold_audit.csv`
- Query rows: `reports/repair_candidate_quality_audit_20260507/query_audit.csv`
- Stage summary: `reports/repair_candidate_quality_audit_20260507/stage_summary.csv`
- Bucket summary: `reports/repair_candidate_quality_audit_20260507/bucket_summary.csv`
- Query bucket summary: `reports/repair_candidate_quality_audit_20260507/query_bucket_summary.csv`
- Source-rank depth summary: `reports/repair_candidate_quality_audit_20260507/rank_depth_summary.csv`
- Examples: `reports/repair_candidate_quality_audit_20260507/case_examples.csv`
- Full JSON: `reports/repair_candidate_quality_audit_20260507/summary.json`

