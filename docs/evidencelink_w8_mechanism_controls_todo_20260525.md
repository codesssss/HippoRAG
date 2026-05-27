# W8 Mechanism-Control Revision TODO

Date: 2026-05-25
Target file: `paper/sections/05_experiments.tex`

## Goal

Address the W8 reviewer concern that the current mechanism controls in
Table 3 preserve EvLink's own readout machinery. The revision should make
the causal scope explicit and add the existing raw traversal readout
control as a separate diagnostic, without running new experiments.

## Core Decision

Use the existing `no_coverage` artifacts as the readout-substitution
control:

```text
run_logs/etv4_ablation_relation_coverage_all32_gpt4omini_full1000_20260516/no_coverage/evals/
```

This is not a "noisy-OR saturation only" ablation. It uses:

```text
reader_budget_k = 5
prefix_budget_m = 5
residual_budget = 0
changed_count = 0
admit_count = 0
```

So the correct interpretation is:

```text
EvLink source-grounded transition topology + raw traversal-order top-5
```

Do not describe this row as only disabling noisy-OR saturation or only
removing a saturation term.

## Numbers To Add

Add a readout-substitution row to Table 3 using retrieval-only / PCEC
diagnostic metrics:

| Variant | HotpotQA R@5 | HotpotQA All@5 | 2Wiki R@5 | 2Wiki All@5 | MuSiQue R@5 | MuSiQue All@5 | Avg R@5 | Avg All@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EvLink edges, raw traversal top-$K$ | 95.05 | 90.50 | 93.50 | 82.60 | 74.17 | 46.60 | 87.6 | 73.2 |

Full EvLink and current topology controls remain:

| Variant | Avg All@5 |
| --- | ---: |
| Full EvLink | 78.0 |
| EvLink edges, raw traversal top-$K$ | 73.2 |
| Dense-doc KNN | 68.5 |
| Edge-count dense-doc KNN | 65.5 |
| Same-seed dense-doc KNN | 68.5 |
| Degree-matched shuffled | 60.9 |

The key RQ3 comparison is not only `78.0 -> 73.2`. That only says the
coverage-aware readout helps. The stronger mechanism comparison is:

```text
EvLink topology with the weakest readout: 73.2 Avg All@5
Dense topology with the full EvLink readout: 68.5 Avg All@5
```

This supports the claim that source-grounded evidence-link topology carries
signal independently of the coverage-aware readout.

## Table 3 Structure

Do not keep the current caption claim that only topology differs across all
controls. After adding the raw traversal row, Table 3 should be split into
two blocks.

Suggested LaTeX structure:

```tex
\midrule
\multicolumn{9}{l}{\emph{Topology substitution (EvLink readout fixed)}} \\
\midrule
Dense-doc KNN & 95.3 & 90.8 & 87.5 & 69.7 & 73.1 & 44.9 & 85.3 & 68.5 \\
Edge-count dense-doc KNN & 95.2 & 90.6 & 83.8 & 62.7 & 72.2 & 43.3 & 83.7 & 65.5 \\
Same-seed dense-doc KNN & 95.3 & 90.8 & 88.1 & 70.0 & 73.4 & 44.8 & 85.6 & 68.5 \\
Degree-matched shuffled & 94.8 & 89.8 & 77.5 & 51.9 & 71.1 & 41.1 & 81.1 & 60.9 \\
\midrule
\multicolumn{9}{l}{\emph{Readout substitution (EvLink topology fixed)}} \\
\midrule
EvLink edges, raw traversal top-$K$ & 95.05 & 90.50 & 93.50 & 82.60 & 74.17 & 46.60 & 87.6 & 73.2 \\
\bottomrule
```

Suggested caption:

```tex
\caption{Mechanism controls under retrieval-only PCEC diagnostics.
The top block fixes the EvLink readout and varies document transition
topology. The bottom block fixes EvLink's source-grounded transition
topology and replaces coverage-aware readout with the raw traversal-order
top-$K$. Row-specific definitions are discussed in the text.}
```

## RQ3 Text Rewrite

Revise the mechanism-control discussion so it explicitly makes the
asymmetric comparison:

```tex
A complementary readout substitution isolates the topology contribution
from the readout contribution. With \methodname{}'s source-grounded
transition topology fixed, replacing the coverage-aware readout with the
raw traversal-order top-$K$ still reaches $73.2$ average All@5. This is
$4.7$ points above Dense-doc KNN under the full \methodname{} readout
($68.5$), and also above the edge-count and same-seed dense controls.
Thus source-grounded transitions carry retrieval signal independently of
the coverage-aware readout machinery.
```

Keep the existing topology-control paragraph, but adjust it so it no longer
claims Table 3 is purely a topology-only table after the new row is added.

## Table 2 Interaction

Do not modify Table 2's existing `w/o coverage-aware reranking` row. Its
configuration and numbers are correct as published. The W8 revision only
adds a new row to Table 3; Table 2 is untouched.

The risk to manage is that Table 3's new row and Table 2's existing row
sound similar but are different experimental configurations. They must
not share a name.

| Where | Row name | Configuration | MuSiQue All@5 |
| --- | --- | --- | ---: |
| Table 2 | `w/o coverage-aware reranking` | `prefix_budget_m=4`, `residual_budget=1`; one residual slot, but ranked by traversal order rather than noisy-OR coverage | 42.2 |
| Table 3 (new) | `EvLink edges, raw BFS top-K` | `prefix_budget_m=5`, `residual_budget=0`; no fine-grained retrieval at all, pool top-5 returned as is | 46.6 |

Naming guidance:

- Table 2 row: keep `w/o coverage-aware reranking` as written.
- Table 3 new row: name it differently, for example
  `EvLink edges, raw BFS top-$K$` or `EvLink edges, no fine-grained retrieval`.
- Avoid `w/o coverage-aware readout` for either row, since that wording fits
  both configurations and would force a footnote to disambiguate.

If a reviewer asks why the two rows differ on MuSiQue (`42.2` vs `46.6`),
the answer is the residual budget, not metric provenance: Table 2's row
still admits one residual swap under traversal-order utility, and that
swap can hurt MuSiQue when the question-side scorer is disabled but the
admission slot remains open. Table 3's new row closes the residual slot
entirely, so the pool top-5 is delivered unchanged.

## Source Configuration For The New Row

The `no_coverage` artifact is the source for Table 3's new row, not for
Table 2.

Verified configuration of the `no_coverage` artifact:

```text
reader_budget_k = 5
prefix_budget_m = 5
residual_budget = reader_budget_k - prefix_budget_m = 0
changed_count = 0
admit_count = 0
```

Verified mechanism in `evidenceflow/native_readout.py`:

- `baseline_positions = list(range(min(reader_budget_k, len(pool_docs))))`,
  i.e. pool positions 0..4 when the pool has at least 5 entries.
- `retained_prefix_positions = list(range(min(prefix_budget_m, len(baseline_positions))))`,
  which equals `baseline_positions` when `prefix_budget_m == reader_budget_k`.
- `finalized_positions` first emits `selected_positions` (capped by `reader_budget_k`),
  then back-fills any remaining slots from pool order. With `prefix_budget_m == reader_budget_k`,
  the safe-projection layer reports `safe_max_swaps = 0`, so `selected_positions`
  cannot displace any baseline position.
- Net effect: `final_positions == baseline_positions == pool top-5`. This is
  what the `changed_count=0` and `admit_count=0` summary confirms.

Therefore Table 3's new row is correctly described as
`EvLink source-grounded transition topology + raw BFS top-K`. R@5 and
All@5 are pure retrieval metrics computed from `final_titles`, so the
`46.6` MuSiQue All@5 reflects the actual retrieved set under this
configuration; it is not an artifact of which evaluator was used.

## Artifact Evidence

Main readout-substitution files:

```text
run_logs/etv4_ablation_relation_coverage_all32_gpt4omini_full1000_20260516/no_coverage/evals/hotpotqa_pcec_no_coverage_prefix5_residual0_pool100_limit1000.json
run_logs/etv4_ablation_relation_coverage_all32_gpt4omini_full1000_20260516/no_coverage/evals/2wikimultihopqa_pcec_no_coverage_prefix5_residual0_pool100_limit1000.json
run_logs/etv4_ablation_relation_coverage_all32_gpt4omini_full1000_20260516/no_coverage/evals/musique_pcec_no_coverage_prefix5_residual0_pool100_limit1000.json
```

Launcher:

```text
run_logs/launch_etv4_ablation_relation_coverage_all32_gpt4omini_full1000_20260516.sh
```

Relevant code:

```text
evidenceflow/contract.py
evidenceflow/dbec_utility.py
evidenceflow/native_readout.py
evidenceflow/run_native_pool.py
scripts/dtc_embed_utils.py
```

Key code facts (governing the new Table 3 row only):

- `DEFAULT_READER_BUDGET_K = 5`
- `DEFAULT_PREFIX_BUDGET_M = 4` (this is the default; `no_coverage` overrides it to 5)
- `residual_budget = reader_budget_k - prefix_budget_m`
- `no_coverage` sets `prefix_budget_m = 5`, so `residual_budget = 0`
- `safe_max_swaps = residual_budget(...) = 0` under that override
- `finalized_positions` falls back to baseline pool top-5 when no swap occurs
- therefore the new Table 3 row equals "EvLink topology + pool top-5", not
  "EvLink topology + Table 2's `w/o coverage-aware reranking` configuration"

## Implementation Checklist

- [ ] Do not modify Table 2. Its `w/o coverage-aware reranking` row stays as published.
- [ ] Add a new row to Table 3 named `EvLink edges, raw BFS top-$K$` (or `EvLink edges, no fine-grained retrieval`) using the retrieval-only PCEC values above.
- [ ] Split Table 3 into a topology-substitution block and a readout-substitution block via two `\midrule` headers.
- [ ] Replace Table 3 caption so it no longer says only topology differs across all rows.
- [ ] Rewrite the RQ3 mechanism paragraph to emphasize the asymmetric comparison `73.2 > 68.5`.
- [ ] Keep FCRG (Table 4) as a separate positive bridge-recovery diagnostic; do not make it carry the whole mechanism claim.
- [ ] Do not compile unless explicitly requested.
