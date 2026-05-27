# ETv4 Ablation #6: MuSiQue Per-Hop Same-Entry Dense vs ETv4

| Hop | Rows | Dense R@5 | ETv4 R@5 | Delta R@5 | Dense all-gold@5 | ETv4 all-gold@5 | Delta all-gold@5 | Top5 changed | Mean overlap | Gold gains | Gold losses | Gain/loss |
| --- | ---- | --------- | -------- | --------- | ---------------- | --------------- | ---------------- | ------------ | ------------ | ---------- | ----------- | --------- |
| 2   | 518  | 0.768340  | 0.827220 | +0.058880 | 0.555985         | 0.673745        | +0.117761        | 405          | 0.791598     | 70         | 11          | 6.36      |
| 3   | 316  | 0.683544  | 0.688819 | +0.005274 | 0.265823         | 0.243671        | -0.022152        | 248          | 0.751733     | 33         | 29          | 1.14      |
| 4   | 166  | 0.412651  | 0.474398 | +0.061747 | 0.000000         | 0.036145        | +0.036145        | 140          | 0.672476     | 46         | 9           | 5.11      |
| all | 1000 | 0.682500  | 0.724917 | +0.042417 | 0.372000         | 0.432000        | +0.060000        | 793          | 0.759226     | 149        | 49          | 3.04      |

Protocol:
- Hop count is `len(gold_doc_indices)`.
- Dense top5 is `source_prior_prefix_doc_indices` saved by the same ETv4 dense-seeded run.
- ETv4 top5 is `retrieved_doc_indices_top5` from the optimized clean mainline run.
- No reader QA is run here; this is retrieval-only attribution.

Source report: `run_logs/etv4_full1000_optimized_equivalence_20260511/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json`
