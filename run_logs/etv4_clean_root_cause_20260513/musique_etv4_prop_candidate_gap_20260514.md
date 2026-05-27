# MuSiQue ETv4 Clean vs PropRAG Candidate Gap

Diagnostic-only. Gold labels are used only for offline failure classification.

## Summary

| Metric | Value |
|---|---:|
| dataset | musique |
| count | 1000 |
| prop_pool_title_all200_only_query_count | 67 |
| prop_top5_title_all_only_query_count | 99 |
| etv4_top5_title_all_only_query_count | 95 |
| prop_pool_only_missing_title_event_count | 71 |
| prop_pool_only_missing_title_classification | {'dense200_recalled_but_displaced_by_graph_budget': 51, 'prop_specific_candidate_generation': 20} |
| prop_pool_only_dense_displaced_title_query_count | 50 |
| prop_pool_only_prop_specific_query_count | 18 |
| prop_pool_only_seed_not_admitted_query_count | 0 |
| prop_top5_only_breakdown | {'etv4_pool_missing_title_gold': 8, 'etv4_pool_has_gold_readout_miss_dense_also_has': 80, 'etv4_pool_has_gold_readout_miss_graph_needed': 11} |
| mean_prop_pool_rank_for_missing_title | 58.55 |
| mean_dense_rank_for_missing_title_when_present | 71.04 |

## Missing Title Event Classification

| Class | Count |
|---|---:|
| dense200_recalled_but_displaced_by_graph_budget | 51 |
| prop_specific_candidate_generation | 20 |

## Prop Top5-only Breakdown

| Bucket | Count |
|---|---:|
| etv4_pool_has_gold_readout_miss_dense_also_has | 80 |
| etv4_pool_has_gold_readout_miss_graph_needed | 11 |
| etv4_pool_missing_title_gold | 8 |

## Representative Missing-title Events

| qid | class | missing title | dense rank | prop rank | seed rank | admitted rank | question |
|---:|---|---|---:|---:|---:|---:|---|
| 5 | dense200_recalled_but_displaced_by_graph_budget | history of saudi arabia | 27 | 102 |  |  | When was the region immediately north of the region where Israel is located and the location of the Battle of Qurah and Umm al Maradim created? |
| 8 | prop_specific_candidate_generation | flyin' the koop |  | 49 |  |  | When was the start of the battle of the birthplace of the performer of III? |
| 13 | dense200_recalled_but_displaced_by_graph_budget | the martyrdom of saint lawrence (titian) | 94 | 29 |  |  | How many times did plague occur in the place where Crucifixion's creator died? |
| 15 | dense200_recalled_but_displaced_by_graph_budget | windjammer communications | 25 | 2 |  |  | What company succeeded the owner of Empire Sports Network? |
| 39 | dense200_recalled_but_displaced_by_graph_budget | world series | 48 | 29 |  |  | Who was second pick in the 1999 draft of the league that has a competition where they give out the MLB MVP award after it? |
| 72 | prop_specific_candidate_generation | alpena power company |  | 153 |  |  | Which county shares a border with the county where the most populous city in the state where Washington State Prison can be found is located? |
| 87 | prop_specific_candidate_generation | 1952 winter olympics |  | 87 |  |  | When was the death penalty abolished in the country near the country where the writer of The Book Thief is a citizen of? |
| 90 | dense200_recalled_but_displaced_by_graph_budget | media in pristina | 31 | 53 |  |  | Where is the headquarters of Radio Television of this country that has a co-official language that is the same as the language of Olivera Markovic? |
| 98 | dense200_recalled_but_displaced_by_graph_budget | orlando furioso (vivaldi, 1714) | 170 | 9 |  |  | How many times did the plague occur in the birth place of Concerto in C Major Op 3 6's composer? |
| 110 | dense200_recalled_but_displaced_by_graph_budget | arizona | 135 | 28 |  |  | Who won the 1993 Indy Car Race in the largest city of the state containing the city where the performer of the album Mingus Three is from? |
| 111 | prop_specific_candidate_generation | socialist party of oregon (columbia county, oregon) |  | 65 |  |  | What mountain can you see from Portland in the state where Fishing Creek Confederacy is located? |
| 133 | dense200_recalled_but_displaced_by_graph_budget | 1939 german ultimatum to lithuania | 26 | 38 |  |  | A country's military branch, the equivalent of which in the US contains the Air Defense Artillery, was unprepared for the invasion of the country occupied by the Nazi's. When was the word "Slavs" used in the national anthem of the unprepared country? |
| 170 | dense200_recalled_but_displaced_by_graph_budget | orlando furioso (vivaldi, 1714) | 98 | 34 |  |  | How many times did plague occur in the place where Bajazet's composer was born? |
| 199 | dense200_recalled_but_displaced_by_graph_budget | ahmed salah hosny | 80 | 190 |  |  | What team is the highest goal scorer in the EPL a member of? |
| 217 | dense200_recalled_but_displaced_by_graph_budget | orlando furioso (vivaldi, 1714) | 64 | 23 |  |  | How many times did plague occur in the birth city of the composer of La fida ninfa? |
| 230 | dense200_recalled_but_displaced_by_graph_budget | casa loma | 25 | 56 |  |  | What is the name of the castle in the place where the performer of Dragon Dreams was born? |
| 237 | dense200_recalled_but_displaced_by_graph_budget | 2020 afc u-23 championship qualification | 25 | 77 |  |  | Who was in charge in the country that is the natural boundary between the country that hosted the tournament and the country where That Dam is located? |
| 237 | dense200_recalled_but_displaced_by_graph_budget | myanmar | 72 | 101 |  |  | Who was in charge in the country that is the natural boundary between the country that hosted the tournament and the country where That Dam is located? |
| 238 | dense200_recalled_but_displaced_by_graph_budget | ramstein-miesenbach | 30 | 2 |  |  | Where is the headquarters of the 1st Combat Communication Squadron located? |
| 295 | dense200_recalled_but_displaced_by_graph_budget | blood (the x-files) | 110 | 164 |  |  | Who is the sibling of the screenwriter of War of the Coprophages? |
