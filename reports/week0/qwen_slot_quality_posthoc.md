# Qwen Slot Quality Post-hoc Diagnostics

- Status: `completed`
- Examples: `997`
- Alignment threshold: `0.35`

## Aggregate Metrics
- avg_gold_slots: `2.6479`
- avg_predicted_slots: `2.7834`
- slot_count_exact: `0.6309`
- slot_count_within_one: `0.9428`
- aligned_answer_type_accuracy: `0.1803`
- dependency_presence_accuracy: `0.6908`
- position_chain_accuracy: `0.5575`
- avg_gold_dependent_slots: `1.5165`
- avg_predicted_internal_dependencies: `1.5196`

## Interpretation

position_chain_accuracy checks whether a generated dependent slot refers to a previous generated output variable, avoiding brittle exact-name matching against oracle x1/x2 labels.

The original exact variable-grounding metric remains useful as a schema strictness check, but this post-hoc metric is the better Week-0 decision signal for latent-slot feasibility.
