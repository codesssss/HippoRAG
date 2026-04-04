# PCRS-RAG V2 Q6 Beam-State Audit

Date: 2026-04-04

## Goal

Audit the `B` run transition after `Riverside Plaza` is admitted, without adding any new probe logic.

Target question:

- `Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`

Target run:

- `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_offlinehybrid_varfix_clean_b_q6factual_20260404.json`

Question to resolve:

1. Does the post-`Riverside Plaza` beam state really absorb its entities?
2. Is `Minneapolis` still at `structure_score = 0` on the next step?
3. Is step-5 scoring using the updated covered set or an old snapshot?

## Short Answer

- `Riverside Plaza` is admitted correctly at step 4 and gets real positive gain.
- On step 5, `Minneapolis` and `Mississippi River` still have `structure_score = 0.0`.
- The code path rules out a stale covered-set snapshot bug: the next beam state explicitly stores `next_covered`, and the next iteration scores against `state["covered_entities"]`.

So the remaining q6 bottleneck is not beam-state update timing. It is further downstream:

- either entity continuity still does not line up (`minneapolis minnesota` vs `minneapolis`)
- or the structure scorer is too strict about what counts as a bridge edge after the first bridge has been admitted

## Evidence

### B run selection steps

The q6 selected titles are:

- `Southeast Library`
- `Colorado River (Texas)`
- `Gulf of Mexico`
- `Riverside Plaza`
- `The Hague City Hall`

The step sequence in the B run is:

1. `Southeast Library`
2. `Colorado River (Texas)`
3. `Gulf of Mexico`
4. `Riverside Plaza`
5. `The Hague City Hall`

### Step 4

At step 4, `Riverside Plaza` is not just exposed; it is promoted all the way through:

- `source rank = 1`
- `shortlist rank = 1`
- `combined_score = 0.7647`
- `structure_score = 1.0`
- `support_completeness_gain = 0.1191`
- `utility_margin_gain = 0.0449`

### Step 5

On the immediately following step, the watch-title trace still shows:

- `Minneapolis`
  - `structure_score = 0.0`
  - `support_completeness_gain = 0.0`
  - `utility_margin_gain = 0.0`
- `Mississippi River`
  - `structure_score = 0.0`
  - `support_completeness_gain = 0.0`
  - `utility_margin_gain = 0.0`

So the first bridge admission does not propagate into a downstream structure signal for the next hop.

## Runtime Entity Audit

The q6 seed-entity preview in the B run is:

- `cedar bayou`
- `gulf of mexico`
- `mississippi river`
- `perdido river`

The runtime entity sets for relevant docs show:

- `Riverside Plaza` includes:
  - `ralph rapson`
  - `riverside plaza`
  - `minneapolis minnesota`
- `Minneapolis` includes:
  - `minneapolis`
  - `mississippi river`
  - `minnesota river`
- `Mississippi River` includes:
  - `mississippi river`
  - `gulf of mexico`
  - `minnesota`

This matters because after step 4 the beam state should now include:

- the original seeds, including `mississippi river`
- plus `Riverside Plaza` entities such as `ralph rapson`, `riverside plaza`, and `minneapolis minnesota`

That makes a pure “selected doc never entered covered state” explanation unlikely.

## Code Audit

### Covered state is initialized from seeds plus reserved docs

In [eval_causal_qwen3.py](/mnt/nvme/code/HippoRAG/scripts/eval_causal_qwen3.py#L2212), the requirement beam normalizes the seed entities.

Then in [eval_causal_qwen3.py](/mnt/nvme/code/HippoRAG/scripts/eval_causal_qwen3.py#L2228), `initial_covered` is created from the seeds and updated with reserved-doc entities.

### Candidate scoring uses the current state's covered entities

During expansion, candidate scoring is called with `covered_entities=state["covered_entities"]` in [eval_causal_qwen3.py](/mnt/nvme/code/HippoRAG/scripts/eval_causal_qwen3.py#L2442).

The same current-state covered set is also passed into need-unit feature extraction / bridge-bonus binding input in [eval_causal_qwen3.py](/mnt/nvme/code/HippoRAG/scripts/eval_causal_qwen3.py#L2470).

### After choosing a candidate, the next state explicitly updates covered entities

When a candidate is accepted, the code does:

- `next_covered = set(state["covered_entities"])`
- `next_covered.update(set(candidate["doc_entities"]))`

in [eval_causal_qwen3.py](/mnt/nvme/code/HippoRAG/scripts/eval_causal_qwen3.py#L2614).

The expanded beam state then stores that updated set as:

- `"covered_entities": next_covered`

in [eval_causal_qwen3.py](/mnt/nvme/code/HippoRAG/scripts/eval_causal_qwen3.py#L2637).

This means the next iteration uses the updated covered set, not the previous snapshot.

## Why `structure_score` can still stay at 0

The structure scorer does not simply reward entity overlap. It first expands seeds into `reachable_scores`, then only counts explicit bridge edges if:

- the edge source is supported by a seed or reachable node
- and the edge target is in `reachable_scores`

See:

- [causal_utils.py](/mnt/nvme/code/HippoRAG/src/hipporag/utils/causal_utils.py#L323)
- [causal_utils.py](/mnt/nvme/code/HippoRAG/src/hipporag/utils/causal_utils.py#L370)
- [causal_utils.py](/mnt/nvme/code/HippoRAG/src/hipporag/utils/causal_utils.py#L438)

Important consequence:

- seed entities themselves are excluded from `reachable_scores`
- pure overlap like `minneapolis minnesota` vs `minneapolis` does not automatically close the chain
- a doc can therefore remain at `structure_score = 0` even when the beam state's covered entities have already been updated

## Conclusion

This audit rules out the highest-priority timing suspicion:

> The B-run q6 failure after `Riverside Plaza` is not caused by a stale beam-state covered set.

What remains is narrower:

- the updated covered set contains the right first-bridge information
- but that information still does not get converted into a downstream structure signal for `Minneapolis` or `Mississippi River`

So the next q6 question is no longer “did the beam state update?” It is:

> why does the updated covered state still fail to satisfy the structure scorer's bridge condition for the next hop?

The two most likely remaining explanations are:

1. continuity mismatch such as `minneapolis minnesota` vs `minneapolis`
2. structure scoring semantics that do not reward edges whose target is already in the seed/covered set
