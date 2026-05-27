# Fixed-Pool Candidate Generation Probe

This diagnostic reranks only the original PropRAG pool100. It does not retrieve new documents from the corpus and does not call the reader.

## Decision

- Decision: `weak_fixed_pool_signal`
- Query-primary slice: `True`
- Target queries: `2`
- Target missing gold titles: `2`
- Run LLM: `True`
- LLM base URLs: `['http://localhost:8041/v1', 'http://localhost:8042/v1', 'http://localhost:8043/v1']`
- Workers: `3`
- All prompts use `/no_think`: `True`

## Policy Summary

| Policy | New@5 | New@10 | New@20 | Policy R@5 | Policy R@10 | mean rank improvement | positive-score | parse ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source_order | 0.0% | 0.0% | 0.0% | 0.0% | 50.0% | 0.0 | 0.0% |  |
| question_lexical | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | -29.0 | 50.0% |  |
| demand_lexical | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | -24.0 | 50.0% |  |
| llm_query_reform | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | -32.5 | 50.0% | 100.0% |
| llm_demand_reform | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | -48.0 | 0.0% | 100.0% |
| llm_demand_hyde | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | -11.0 | 100.0% | 100.0% |
| llm_combined | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | -28.0 | 100.0% |  |
| llm_listwise_select | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | -11.0 | 0.0% | 100.0% |

## Top Rank Improvements

- `source_order` q24 `Paula Santiago` rank 18 -> 18 via `` ()
- `source_order` q28 `Imperialism` rank 7 -> 7 via `` ()
- `llm_demand_hyde` q28 `Imperialism` rank 7 -> 10 via `The arguer, Nikita Khrushchev, was a prominent leader of the Soviet Union and a key figure in the Cold War era.` (body_token_overlap)
- `question_lexical` q28 `Imperialism` rank 7 -> 11 via `Where did the arguer that the country Directive 10/2 called for actions against had become an imperialist power declare ` (body_token_overlap)
- `llm_listwise_select` q24 `Paula Santiago` rank 18 -> 29 via `` ()
- `llm_combined` q28 `Imperialism` rank 7 -> 18 via `The arguer, Nikita Khrushchev, was a prominent leader of the Soviet Union and a key figure in the Cold War era.` (body_token_overlap)
- `llm_listwise_select` q28 `Imperialism` rank 7 -> 18 via `` ()
- `demand_lexical` q28 `Imperialism` rank 7 -> 21 via `s1: Which country is referred to as the one that Directive 10/2 called for actions against? Directive 10/2` (body_token_overlap)
- `llm_demand_hyde` q24 `Paula Santiago` rank 18 -> 37 via `The Italian navigator Christopher Columbus explored the eastern coast of the continent of North America during his first` (body_token_overlap)
- `llm_query_reform` q28 `Imperialism` rank 7 -> 34 via `Directive 10/2 country called for actions against` (body_token_overlap)
- `demand_lexical` q24 `Paula Santiago` rank 18 -> 52 via `` ()
- `llm_query_reform` q24 `Paula Santiago` rank 18 -> 56 via `` ()
- `llm_demand_reform` q28 `Imperialism` rank 7 -> 49 via `` ()
- `llm_combined` q24 `Paula Santiago` rank 18 -> 63 via `The Italian navigator Christopher Columbus explored the eastern coast of the continent of North America during his first` (body_token_overlap)
- `question_lexical` q24 `Paula Santiago` rank 18 -> 72 via `` ()
- `llm_demand_reform` q24 `Paula Santiago` rank 18 -> 72 via `` ()

## Interpretation Rules

- Strong fixed-pool signal requires an LLM policy to reach New@5 >= 25% and New@10 >= 35%.
- Mixed signal means New@5 >= 15%, enough to inspect but not enough to start a new method line.
- Weak signal means fixed-pool candidate generation remains difficult even under LLM reformulation or listwise pool selection.

## Files

- Probe rows: `reports/fixed_pool_candidate_generation_probe_smoke_20260508/probe_rows.csv`
- Policy summary: `reports/fixed_pool_candidate_generation_probe_smoke_20260508/policy_summary.csv`
- Prompt outputs: `reports/fixed_pool_candidate_generation_probe_smoke_20260508/prompt_outputs.jsonl`
- Query traces: `reports/fixed_pool_candidate_generation_probe_smoke_20260508/query_traces.jsonl`
- Full summary: `reports/fixed_pool_candidate_generation_probe_smoke_20260508/summary.json`

