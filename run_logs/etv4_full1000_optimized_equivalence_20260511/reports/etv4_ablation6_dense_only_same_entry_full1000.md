# ETv4 Ablation #6: Dense-Only Same-Entry Full1000

| Dataset         | Rows | Dense R@5 | ETv4 R@5 | ΔR@5      | Dense all-gold@5 | ETv4 all-gold@5 | Δall-gold@5 | Top5 changed | Mean overlap | Gold gains | Gold losses |
| --------------- | ---- | --------- | -------- | --------- | ---------------- | --------------- | ----------- | ------------ | ------------ | ---------- | ----------- |
| 2wikimultihopqa | 1000 | 0.758500  | 0.920000 | +0.161500 | 0.493000         | 0.797000        | +0.304000   | 811          | 0.724167     | 375        | 5           |
| musique         | 1000 | 0.682500  | 0.724917 | +0.042417 | 0.372000         | 0.432000        | +0.060000   | 793          | 0.759226     | 149        | 49          |
| hotpotqa        | 1000 | 0.933500  | 0.950500 | +0.017000 | 0.871000         | 0.906000        | +0.035000   | 693          | 0.840012     | 52         | 17          |

Protocol:
- Dense top5 is `source_prior_prefix_doc_indices` saved by the same ETv4 dense-seeded run.
- ETv4 top5 is `retrieved_doc_indices_top5` from the optimized clean mainline run.
- No reader QA is run here; this is retrieval-only attribution.

Source root: `run_logs/etv4_full1000_optimized_equivalence_20260511`
