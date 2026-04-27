# DAEC-ALR Step-1 Single-Edit Gate

## Purpose

Test the frozen DAEC-ALR Step-1 plan: single-edit repair inside the existing PropRAG top100 pool.

## Configuration

- dataset: `2wikimultihopqa`
- DAEC report: `run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json`
- pool JSON: `run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json`
- rows: `200`
- variants: `no_edit_daec_only, binding_gate_only, consistency_gate_only, double_gate_no_skip, double_gate_skip`
- reader mode: `openai_compatible_chat`
- model/cache key model: `qwen3-8b-train`

## Variant Summary

| Variant | F1 | ΔF1 | CI95 ΔF1 | Support Complete | Accepted | Edit Subset F1 Pre/Post | W→C | C→W | Added G/NG | NG/G |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_edit_daec_only` | 0.5881 | 0.0000 | [0.0000, 0.0000] | 0.8550 | 0.000 | 0.0000/0.0000 | 0 | 0 | 0/0 | n/a |
| `binding_gate_only` | 0.5245 | -0.0636 | [-0.1159, -0.0190] | 0.6450 | 1.000 | 0.5887/0.5245 | 9 | 25 | 13/187 | 14.38 |
| `consistency_gate_only` | 0.5757 | -0.0125 | [-0.0371, 0.0110] | 0.8150 | 0.230 | 0.2910/0.2343 | 3 | 5 | 4/42 | 10.50 |
| `double_gate_no_skip` | 0.5757 | -0.0125 | [-0.0371, 0.0110] | 0.8150 | 0.230 | 0.2910/0.2343 | 3 | 5 | 4/42 | 10.50 |
| `double_gate_skip` | 0.5757 | -0.0125 | [-0.0371, 0.0110] | 0.8150 | 0.230 | 0.2910/0.2343 | 3 | 5 | 4/42 | 10.50 |

## Main Decision

- main variant: `double_gate_skip`
- decision: `RED`
- F1 delta: `-0.012461`
- accepted edit rate: `0.230000`
- support_complete: `0.815000`

## Notes

- The binding gate uses a fixed trace-local demand/binding proxy because the exported DAEC report does not contain enough embedding state to recompute exact DAEC candidate scores.
- The proxy protects covered requirement positions, replaces low-utility selected docs, and logs component scores for every candidate.
- No retrieval outside the cached pool is performed.
