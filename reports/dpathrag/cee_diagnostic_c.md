# CEE Diagnostic C

## Same-Remove Admission Signal

- Rows: `1000`
- Beneficial edits: `543`
- Lexical negative edits: `36230`
- Same-remove beneficial-vs-lexical pairs: `526`
- Queries with same-remove pairs: `132`

## Current Shallow CEE Error Surface

- False-positive edited queries/edits: `185` / `294`
- False-negative queries with beneficial edit missed: `109`

## Positive Add vs Lexical Negative Add Feature Means

| Feature | Positive Mean | Lexical Negative Mean | Delta |
|---|---:|---:|---:|
| `rank` | 9.924494 | 12.637179 | -2.712685 |
| `retriever_score` | 0.00309 | 0.002667 | 0.000423 |
| `title_question_jaccard` | 0.082693 | 0.070589 | 0.012104 |
| `body_question_jaccard` | 0.067502 | 0.090874 | -0.023372 |
| `question_token_coverage` | 0.367745 | 0.35091 | 0.016835 |
| `q_doc_cosine` | 0.279297 | 0.211095 | 0.068202 |
| `answer_in_doc` | 0.589319 | 0.0 | 0.589319 |
| `bridge_entity_in_doc` | 0.869245 | 0.0 | 0.869245 |
| `log_doc_chars` | 6.269793 | 5.618658 | 0.651135 |

## DAEC Feature Audit

- Status: `no_direct_joinable_cache_found`
- Decision: Use CEE-pairwise v0 first; add DAEC features only after a fold-safe per-query/per-doc cache is confirmed.

| Candidate Artifact | Score |
|---|---:|
| `docs/cee_v2_pairwise_admission_plan_20260427.md` | 7 |
| `scripts/dpathrag_cee_diagnostic_c.py` | 7 |
| `docs/dpathrag_method_and_pilot_plan_20260426.md` | 6 |
| `outputs_step0_general_nvembed_2wikimultihopqa/eval_reports/dtc_embed_nvembed_ser_lam1p0_smoke5_anchor2_8042.json` | 5 |
| `reports/setr_pathA/setr_2wikimultihopqa_dense_k100_limit100.eval.json` | 5 |
| `reports/setr_pathA/setr_2wikimultihopqa_dense_k100_limit100_doc160.eval.json` | 5 |
| `reports/setr_pathA/setr_2wikimultihopqa_dense_k20_limit100.eval.json` | 5 |
| `reports/setr_pathA/setr_2wikimultihopqa_dense_k50_limit100.eval.json` | 5 |
| `reports/setr_pathA/setr_2wikimultihopqa_dense_k50_limit100_doc320.eval.json` | 5 |
| `reports/setr_pathA/setr_2wikimultihopqa_proprag_k100_limit100.eval.json` | 5 |
| `reports/setr_pathA/setr_2wikimultihopqa_proprag_k100_limit100_doc160.eval.json` | 5 |
| `reports/setr_pathA/setr_2wikimultihopqa_proprag_k20_limit100.eval.json` | 5 |
| `reports/setr_pathA/setr_2wikimultihopqa_proprag_k50_limit100.eval.json` | 5 |
| `reports/setr_pathA/setr_2wikimultihopqa_proprag_k50_limit100_doc320.eval.json` | 5 |
| `reports/setr_pathA/setr_windowed_2wikimultihopqa_dense_k100_limit100.eval.json` | 5 |
| `reports/setr_pathA/setr_windowed_2wikimultihopqa_dense_k50_limit100.eval.json` | 5 |
| `reports/setr_pathA/setr_windowed_2wikimultihopqa_proprag_k100_limit100.eval.json` | 5 |
| `reports/setr_pathA/setr_windowed_2wikimultihopqa_proprag_k50_limit100.eval.json` | 5 |
| `scripts/eval_causal_qwen3.py` | 5 |
| `outputs_step0_general_hotpotqa/eval_reports/setwise_bridge_beam_set_closure_exactonly_100_legacy_reserve3_dedup.json` | 4 |
