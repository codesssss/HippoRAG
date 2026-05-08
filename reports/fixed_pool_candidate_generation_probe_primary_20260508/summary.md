# Fixed-Pool Candidate Generation Probe

This diagnostic reranks only the original PropRAG pool100. It does not retrieve new documents from the corpus and does not call the reader.

## Decision

- Decision: `mixed_fixed_pool_signal`
- Query-primary slice: `True`
- Target queries: `62`
- Target missing gold titles: `86`
- Run LLM: `True`
- LLM base URLs: `['http://localhost:8041/v1', 'http://localhost:8042/v1', 'http://localhost:8043/v1']`
- Workers: `6`
- All prompts use `/no_think`: `True`

## Policy Summary

| Policy | New@5 | New@10 | New@20 | Policy R@5 | Policy R@10 | mean rank improvement | positive-score | parse ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source_order | 0.0% | 0.0% | 0.0% | 1.2% | 18.6% | 0.0 | 0.0% |  |
| question_lexical | 5.8% | 5.8% | 8.1% | 5.8% | 9.3% | -12.3 | 57.0% |  |
| demand_lexical | 5.8% | 7.0% | 9.3% | 5.8% | 9.3% | -11.8 | 59.3% |  |
| llm_query_reform | 5.8% | 4.7% | 10.5% | 5.8% | 7.0% | -12.9 | 65.1% | 100.0% |
| llm_demand_reform | 5.8% | 8.1% | 9.3% | 5.8% | 10.5% | -12.3 | 55.8% | 100.0% |
| llm_demand_hyde | 11.6% | 14.0% | 14.0% | 11.6% | 17.4% | -2.1 | 95.3% | 100.0% |
| llm_combined | 8.1% | 10.5% | 10.5% | 8.1% | 12.8% | -5.8 | 95.3% |  |
| llm_listwise_select | 26.7% | 27.9% | 20.9% | 26.7% | 33.7% | 3.7 | 33.7% | 90.3% |

## Top Rank Improvements

- `demand_lexical` q69 `Tucson, Arizona` rank 94 -> 6 via `s3: What is the largest populated city in that state?` (body_token_overlap)
- `llm_demand_reform` q69 `Tucson, Arizona` rank 94 -> 7 via `largest populated city in [state from s2]` (body_token_overlap)
- `question_lexical` q69 `Tucson, Arizona` rank 94 -> 8 via `Who won the Indy Car Race in the largest populated city of the state where the performer of Mingus Plays Piano is from?` (body_token_overlap)
- `llm_query_reform` q69 `Tucson, Arizona` rank 94 -> 10 via `largest populated city in [state of performer of Mingus Plays Piano]` (body_token_overlap)
- `llm_listwise_select` q837 `Casa Loma` rank 88 -> 4 via `listwise_doc_id=88` (llm_listwise_selected)
- `llm_combined` q69 `Tucson, Arizona` rank 94 -> 18 via `largest populated city in [state from s2]` (body_token_overlap)
- `llm_demand_reform` q646 `Myanmar` rank 79 -> 5 via `What is the natural boundary between [host country] and [A Don's country]?` (body_token_overlap)
- `question_lexical` q646 `Myanmar` rank 79 -> 6 via `For what major conflict is the country on the natural boundary between the country that hosted the tournament and the co` (body_token_overlap)
- `llm_demand_hyde` q837 `Casa Loma` rank 88 -> 18 via `Manchester is home to several historic castles, including the Castlefield area, which features the iconic Castle Keep.` (body_token_overlap)
- `llm_demand_hyde` q87 `1952 Winter Olympics` rank 81 -> 13 via `Australia is a country located near New Zealand, which is a neighboring country in the Pacific Ocean.` (body_token_overlap)
- `llm_demand_hyde` q69 `Tucson, Arizona` rank 94 -> 28 via `The largest populated city in Mississippi is Jackson, which serves as the state capital and a major cultural hub.` (body_token_overlap)
- `llm_query_reform` q281 `Myanmar` rank 83 -> 18 via `Ajuran Empire coins and independence from expelled people` (body_token_overlap)
- `llm_demand_reform` q281 `Myanmar` rank 83 -> 18 via `Somali Muslim Ajuran Empire coins independence from people` (body_token_overlap)
- `llm_listwise_select` q568 `Casa Loma` rank 68 -> 3 via `listwise_doc_id=68` (llm_listwise_selected)
- `llm_demand_hyde` q815 `Adult contemporary music` rank 81 -> 16 via `A prominent figure at the Global Radio Division is John Carter, who has been leading the division since 2015.` (body_token_overlap)
- `demand_lexical` q281 `Myanmar` rank 83 -> 19 via `s4: Which people did the Somali Muslim Ajuran Empire make coins to proclaim independence from? Somali Muslim Ajuran Empi` (body_token_overlap)

## Interpretation Rules

- Strong fixed-pool signal requires an LLM policy to reach New@5 >= 25% and New@10 >= 35%.
- Mixed signal means New@5 >= 15%, enough to inspect but not enough to start a new method line.
- Weak signal means fixed-pool candidate generation remains difficult even under LLM reformulation or listwise pool selection.

## Files

- Probe rows: `reports/fixed_pool_candidate_generation_probe_primary_20260508/probe_rows.csv`
- Policy summary: `reports/fixed_pool_candidate_generation_probe_primary_20260508/policy_summary.csv`
- Prompt outputs: `reports/fixed_pool_candidate_generation_probe_primary_20260508/prompt_outputs.jsonl`
- Query traces: `reports/fixed_pool_candidate_generation_probe_primary_20260508/query_traces.jsonl`
- Full summary: `reports/fixed_pool_candidate_generation_probe_primary_20260508/summary.json`

