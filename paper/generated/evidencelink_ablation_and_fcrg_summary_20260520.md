# EvLink Ablation and FCRG Summary

Generated on 2026-05-20 from completed local artifacts in `run_logs/`.

## Protocol Notes

- Component ablations use the unified `GPT-4o-mini` reader reports and report `R@5`, `F1`, and `All@5`.
- Mechanism controls are retrieval-only/PCEC diagnostics and report `R@5` and `All@5`; no `GPT-4o-mini` reader was run for these controls.
- FCRG uses dense query-only as the reference ranker. Hard FCRG filters to fact-linked gold support documents that dense query-only misses from top-5.

## Component Ablations

| Variant | Hotpot R@5 | Hotpot F1 | Hotpot All@5 | 2Wiki R@5 | 2Wiki F1 | 2Wiki All@5 | MuSiQue R@5 | MuSiQue F1 | MuSiQue All@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full EvLink | 96.50 | 75.57 | 93.30 | 96.23 | 74.68 | 89.50 | 73.51 | 48.91 | 45.30 |
| w/o evidence-need mining | 95.20 | 74.81 | 90.90 | 92.80 | 71.11 | 81.10 | 72.95 | 48.99 | 43.90 |
| w/o Evidence-Coverage-Aware Retrieval Optimization | 95.05 | 74.78 | 90.50 | 93.50 | 72.79 | 82.60 | 71.83 | 48.33 | 42.20 |
| w/o evidence-linked transitions | 96.15 | 74.97 | 92.70 | 93.85 | 72.98 | 82.00 | 72.52 | 47.45 | 43.40 |

### Component Takeaways

- Removing evidence-need mining mainly hurts 2Wiki: `R@5 -3.43`, `F1 -3.57`, `All@5 -8.40`.
- Removing Evidence-Coverage-Aware Retrieval Optimization hurts all three datasets, especially 2Wiki `All@5 -6.90` and MuSiQue `All@5 -3.10`.
- Replacing evidence-linked transitions with a phrase-source-only graph keeps the rest of the protocol fixed but drops 2Wiki `All@5` from `89.50` to `82.00`.

## Mechanism Controls

These controls test whether the gain comes from source-grounded edge semantics rather than simply using a denser or larger document graph.

| Variant | Hotpot R@5 | Hotpot All@5 | 2Wiki R@5 | 2Wiki All@5 | MuSiQue R@5 | MuSiQue All@5 | Avg R@5 | Avg All@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full EvLink | 96.55 | 93.40 | 96.30 | 89.70 | 76.42 | 51.00 | 89.76 | 78.03 |
| budget dense-doc KNN | 95.25 | 90.80 | 87.52 | 69.70 | 73.13 | 44.90 | 85.30 | 68.47 |
| strict edge-count dense | 95.15 | 90.60 | 83.80 | 62.70 | 72.15 | 43.30 | 83.70 | 65.53 |
| degree-matched shuffled | 94.75 | 89.80 | 77.48 | 51.90 | 71.14 | 41.10 | 81.12 | 60.93 |

### Mechanism-Control Takeaways

- The strict edge-count control matches each query's transition edge count to Full EvLink but replaces fact-certified transitions with dense document links.
- Full EvLink still beats strict edge-count dense by `+6.06` Avg `R@5` and `+12.50` Avg `All@5`.
- The 2Wiki gap is largest: Full EvLink beats strict edge-count dense by `+12.50` `R@5` and `+27.00` `All@5`.
- This supports the claim that source-grounded fact-certified edges, rather than graph density or edge count, drive the retrieval gain.

## Hard FCRG

Hard FCRG evaluates only fact-linked gold support documents that dense query-only misses from top-5.

| Method | Hard FCRG | Method@5 | Fact docs |
|---|---:|---:|---:|
| EvLink pool | 0.2807 | 57.12 | 1320 |
| EvLink delivered | 0.2995 | 67.58 | 1320 |
| w/o evidence-linked transitions delivered | 0.2675 | 58.56 | 1320 |
| Dense-doc KNN transitions delivered | 0.0911 | 38.41 | 1320 |
| Edge-count matched dense transitions delivered | 0.0642 | 30.30 | 1320 |
| Degree-matched shuffled transitions delivered | 0.0059 | 15.68 | 1320 |
| HippoRAG2 pool | 0.1303 | 32.12 | 1320 |
| PropRAG pool | 0.2516 | 58.18 | 1320 |

### FCRG Takeaways

- EvLink delivered has the highest Hard FCRG: `0.2995`.
- EvLink delivered promotes `67.58%` of dense-missed fact-linked gold support documents into top-5.
- Edge-count matched dense transitions reach only `0.0642` Hard FCRG and `30.30%` Method@5.
- Shuffled transitions are near zero, showing that arbitrary document connectivity does not recover fact-conditioned bridge evidence.

## Artifact Locations

- Full EvLink PCEC: `run_logs/all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514/pcec/etv4/evals/`
- Full EvLink reader QA: `run_logs/all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514/reader_qa/etv4/reports/`
- Evidence-need and coverage ablations: `run_logs/etv4_ablation_relation_coverage_all32_gpt4omini_full1000_20260516/`
- Phrase/source-only ablation: `run_logs/evidencelink_phrase_source_only_unified_qwen32b_gpt4omini_full1000_20260519/`
- Dense/shuffled mechanism controls: `run_logs/evidencelink_doc_transition_controls_retrieval_full1000_20260520/`
- Strict edge-count dense control: `run_logs/evidencelink_edge_count_matched_dense_control_full1000_20260520/`
- Default FCRG: `run_logs/evidencelink_fcrg_20260520/`
- Hard FCRG: `run_logs/evidencelink_fcrg_hard_dense_miss_top5_20260520/`
