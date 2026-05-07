# DAEC Binding Posterior & Verifier Probe — Negative Results (2026-05-06)

## Summary

Two experiments attempted to improve DAEC's latent binding selection. Both failed and converge on the same root cause: no train-free signal available in the current pipeline can safely distinguish correct from incorrect entity bindings.

## Experiment 1: Grounded Posterior Ablation

**Hypothesis**: When multiple bindings achieve tied noisy-OR objectives (saturation), an upstream φ-derived grounding score can break ties in favor of correct bindings.

**Mechanism**: `binding_selection_score = effective_objective × grounding_score`, where grounding_score = mean of `phi[upstream_req_idx, binding_idx, entity_doc_pos]` across binding assignments.

**Result**: Every binding flip was wrong. All 4 EM losses on 2Wiki came from grounding favoring prominent-but-wrong entities over correct-but-obscure ones (e.g., Claude Autant-Lara grounding=0.719 over correct Frank Lloyd grounding=0.184 for "Madame La Presidente" director).

**Root cause**: Embedding cosine similarity ≠ factual entailment. The embedding space encodes topical relatedness, not predicate satisfaction.

## Experiment 2: Saturation-Aware Verifier Probe

**Hypothesis**: Text co-mention (entity appears in upstream doc or companion doc) is a safer structural signal than embedding similarity.

**Mechanism**: Offline probe; only flips bindings when objectives are exactly tied AND base binding lacks co-mention support while an alternative has it.

**Result**: At most 3 flips on 2Wiki (extraction_or_companion mode), with 1 rescue canceled by 1 regression. Selected_companion mode was too inert to produce any meaningful change.

**Root cause**: Entity co-mention ≠ relation entailment. Co-mention walks one-hop relations indiscriminately (Q14: William the Silent → William I of Nassau through family co-mention, but wrong predicate).

## Convergent Conclusion

| Signal | Failure Mode | Root Cause |
|---|---|---|
| Embedding φ (grounded posterior) | Favors salient-but-wrong entities | Cosine similarity ≠ factual entailment |
| Text co-mention (verifier probe) | Follows one-hop relations blindly | Co-mention ≠ relation entailment |

Both fail because they detect entity **relevance** but cannot verify the specific **relation** (directed_by, born_in, father_of) that the upstream demand requires.

## Paper Implications

- DAEC's contribution is train-free decomposition-aware evidence selection, not binding self-verification.
- Honest limitations: (1) binding correctness depends on decomposition/extraction quality; (2) embedding φ is topical similarity, not entailment; (3) coverage objective saturates even with wrong bindings.
- The base LLM binding (dep_score ranking from extraction) is already correct 82-87% of the time on 2Wiki; the correction surface is small and the available signals are too noisy to exploit it safely.

## Files

- Grounded ablation launcher: `run_logs/launch_daec_binding_grounded_limit100_20260506.sh`
- Grounded ablation results: `run_logs/daec_binding_grounded_limit100_20260506/`
- Verifier probe script: `scripts/audit_daec_binding_verifier.py`
- Verifier probe design: `docs/daec_saturation_binding_verifier_probe_20260506.md`
- Verifier probe reports: `reports/daec_binding_verifier_20260506/`
