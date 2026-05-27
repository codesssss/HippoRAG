# AREC-RAG MuSiQue limit=100 Smoke Summary

- Status: `completed_smoke_with_red_flags`
- Dataset: `musique`
- Limit: `100`
- Pool: `run_logs/dense_pool_exports_full1000_20260424/musique_dense_pool100.json`
- Verifier: `cross-encoder/nli-deberta-v3-base`
- CoT prompt/query source: `local_ircot_style_frozen_prompt`

## Main Metrics

| Stage | AREC Missing Hit | Raw Missing Hit | CoT Missing Hit | Initial SC@5 | Final SC@5 | Initial Recall@5 | Final Recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Day 0 silver oracle | 0.0320 | 0.0080 | 0.0220 | 0.2900 | 0.5000 | 0.6458 | 0.7675 |
| Day 2 generated | 0.0160 | 0.0080 | 0.0220 | 0.2900 | 0.2400 | 0.6458 | 0.4892 |

## Closure Signal

| Stage | Generated Closure | Silver Oracle Closure | Initial SC@5 | Active Obligations |
|---|---:|---:|---:|---:|
| Day 1 generated | 0.8342 | 2.1344 | 0.2900 | 2.7600 |

## Interpretation

- Day 0 is positive under silver-oracle obligations: AREC residual retrieval beats raw question and local CoT-style retrieval.
- Generated obligations are the bottleneck: Day 2 generated AREC loses to CoT-style retrieval and degrades final Support-Complete@5.
- Current gate decision: do not promote AREC full implementation yet. Rerun only after improving/fixing obligation generation and replacing local CoT prompt with official IRCoT prompt/config.

## Caveats

- Silver oracle obligations are generated from gold docs by frozen LLM, not manual oracle annotations.
- CoT baseline is local frozen IRCoT-style prompt, not confirmed official IRCoT released prompt/config.
- Final reader was not rerun after projection in this smoke harness; final answer metrics are placeholders from the initial answer.
- NLI verifier used `cross-encoder/nli-deberta-v3-base` from local cache.
