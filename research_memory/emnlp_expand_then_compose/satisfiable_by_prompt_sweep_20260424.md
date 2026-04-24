# DtC `satisfiable_by` Prompt Sweep - 2026-04-24

## Question

We added a `satisfiable_by: document | inference` field to the DtC decomposition prompt to replace a regex-based repairable filter with an LLM-produced requirement taxonomy.

The paper-safety motivation was sound, but pilot-100 showed a performance drop. This sweep tested whether the drop came from the downstream policy or from perturbing the decomposition prompt itself.

## Compared Variants

All runs use the same NV-Embed / Qwen3-8B protocol:

- `limit=100`
- `qa_top_k=5`
- `setwise_pool_k=100`
- `setwise_anchor_count=2`
- `setwise_reserve_top_m=0`
- `dtc_rank_weight=0.2`
- `dtc_repairable_filter_enabled=true`
- `dtc_enable_dependency_binding=true`
- no instruction prefix for NV embeddings

Variants:

- `old_repairable`: old decomposition prompt, no `satisfiable_by` field.
- `sat_by_strict`: prompt includes `satisfiable_by`; `inference` always vetoes repairability.
- `binding_override`: `inference` veto is overridden when a dependent requirement has resolved binding.
- `grounded_override`: `inference` veto is overridden when the requirement has an explicit anchor or resolved binding.
- `regex_only`: prompt includes `satisfiable_by`, but downstream ignores that field and uses the old regex/role fallback.

## Results

| Dataset | Variant | Selector F1 | Delta F1 | Selector R@5 | Wins/Losses/Ties | Changed |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | old_repairable | 0.5477 | +0.0946 | 0.8850 | 12/1/87 | 24 |
| 2Wiki | sat_by_strict | 0.5289 | +0.0739 | 0.8675 | 10/2/88 | 20 |
| 2Wiki | binding_override | 0.5368 | +0.0837 | 0.8800 | 11/1/88 | 23 |
| 2Wiki | grounded_override | 0.5406 | +0.0828 | 0.8800 | 11/2/87 | 24 |
| 2Wiki | regex_only | 0.5368 | +0.0837 | 0.8800 | 11/1/88 | 22 |
| Hotpot | old_repairable | 0.7054 | +0.0310 | 0.9550 | 4/1/95 | 23 |
| Hotpot | sat_by_strict | 0.7094 | +0.0350 | 0.9450 | 4/0/96 | 19 |
| Hotpot | binding_override | 0.7054 | +0.0310 | 0.9500 | 4/1/95 | 22 |
| Hotpot | grounded_override | 0.7054 | +0.0310 | 0.9500 | 4/1/95 | 23 |
| Hotpot | regex_only | 0.7054 | +0.0310 | 0.9500 | 4/1/95 | 22 |
| MuSiQue | old_repairable | 0.4484 | +0.0632 | 0.7283 | 9/2/89 | 48 |
| MuSiQue | sat_by_strict | 0.4379 | +0.0560 | 0.6875 | 6/0/94 | 35 |
| MuSiQue | binding_override | 0.4380 | +0.0556 | 0.7067 | 10/3/87 | 45 |
| MuSiQue | grounded_override | 0.4387 | +0.0557 | 0.7125 | 10/3/87 | 46 |
| MuSiQue | regex_only | 0.4347 | +0.0523 | 0.7117 | 9/3/88 | 45 |

Average selector F1:

- `old_repairable`: 0.5672
- `sat_by_strict`: 0.5587
- `binding_override`: 0.5601
- `grounded_override`: 0.5616
- `regex_only`: 0.5590

## Conclusion

The degradation is not mainly caused by downstream `satisfiable_by` policy.

Evidence: `regex_only` still drops versus `old_repairable`, even though it ignores the `satisfiable_by` field in repairable filtering. Therefore the prompt/schema change itself perturbs the LLM decomposition output.

The safe implementation choice is:

- Keep `satisfiable_by` parsing and downstream policy support.
- Make the decomposition prompt field optional.
- Default `--dtc_include_satisfiable_by=false` for main experiments to preserve the validated prompt.
- Use `--dtc_include_satisfiable_by=true` only for diagnostic/appendix taxonomy runs.

If a paper-safe taxonomy must be used online, `grounded_override` is the best available policy among the tested variants, but it still underperforms the old prompt on 2Wiki and MuSiQue.
