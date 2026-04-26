# D-PathRAG 2Wiki Split And Pool Audit

## Dataset

- Dataset: `2wikimultihopqa`
- Local samples: `1000` from `reproduce/dataset/2wikimultihopqa.json`
- Audited samples: `1000`
- Local corpus: `6119` from `reproduce/dataset/2wikimultihopqa_corpus.json`
- Split status: `local_eval_subset_only`
- Has train/dev/test locally: `False`

## Schema

- `_id` coverage: `1.0`
- `type` coverage: `1.0`
- `question` coverage: `1.0`
- `context` coverage: `1.0`
- `supporting_facts` coverage: `1.0`
- `evidences` coverage: `1.0`
- `evidences_id` coverage: `1.0`
- `answer` coverage: `1.0`

## Support / Evidence Stats

- Avg supporting facts: `2.471`
- Avg support titles: `2.47`
- Avg evidence path length: `2.507`
- Avg context docs: `10.0`

## Pool Compatibility

### dense

- Exists: `True`
- Path: `run_logs/dense_pool_exports_full1000_20260424/2wikimultihopqa_dense_pool100.json`
- Records: `1000`
- Question mismatches: `0`
- Gold-title mismatches: `0`
- Support recall@5 / @20 / @100: `0.7238` / `0.799` / `0.877`
- Support complete@5 / @20 / @100: `0.429` / `0.558` / `0.706`

### proprag

- Exists: `True`
- Path: `run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json`
- Records: `1000`
- Question mismatches: `0`
- Gold-title mismatches: `0`
- Support recall@5 / @20 / @100: `0.9028` / `0.9607` / `0.9872`
- Support complete@5 / @20 / @100: `0.772` / `0.9` / `0.968`

## Recommendation

- Can build smoke cache: `True`
- Build a D-PathRAG smoke cache from the local 1000-example subset, but acquire/prepare official 2Wiki train/dev before E2E training.

## Notes

- The local 1000-example file is suitable for smoke/cache audit, not for full E2E selector training.
- D-PathRAG Phase 1 must use HuggingFace differentiable reader training; vLLM/Qwen is only for later hard-selection evaluation.
- The local audit does not validate full training feasibility unless an official train split is present.
