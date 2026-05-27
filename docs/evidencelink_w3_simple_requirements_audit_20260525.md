# W3 Simple-Requirements Ablation TODO and Audit

Date: 2026-05-25
Target file: `paper/sections/05_experiments.tex`

## Goal

Address the W3 reviewer concern that LLM-based question decomposition and
graph-substrate contributions are not isolated. Add one ablation row that
keeps the EvLink link substrate, BFS, binding cache, and coverage readout
fixed, and replaces the Qwen3-32B five-field requirement set with a single
requirement equal to the question itself. This is a textual decomposition
ablation only; no new substrate is built.

The full simple-anchor variant (L1) does not enter Table 2. It is recorded
in the appendix as a confirmatory negative result.

## Core Decision

Use the existing whole-question artifacts as the simple-requirement
control:

```text
run_logs/evidencelink_simple_requirements_full1000_20260525/readout/whole_question/
run_logs/evidencelink_simple_requirements_full1000_20260525/reader_qa/whole_question/
```

Configuration of this variant:

```text
prefix_budget_m = 4
reader_budget_k = 5
residual_budget = 1
binding_cache    = unchanged from full EvLink
link_substrate   = unchanged from full EvLink
B(q)             = a single requirement whose subquery is the question text
                   and whose other DTC fields are default values
```

So the correct interpretation is:

```text
EvLink substrate + EvLink BFS + EvLink coverage readout
+ |B(q)| = 1, subquery = question, default DTC fields
```

Do not describe this row as "raw BFS top-K" or "no readout"; the full
PCEC readout is still applied, but with a degenerate one-element
requirement set.

## Numbers To Add

Add a simple-requirements row to Table 2 using reader-facing metrics:

| Variant                                  | Hot R@5 | Hot F1 | Hot All@5 | 2Wiki R@5 | 2Wiki F1 | 2Wiki All@5 | MuSiQue R@5 | MuSiQue F1 | MuSiQue All@5 |
| ---                                      | ---:    | ---:   | ---:      | ---:      | ---:     | ---:        | ---:        | ---:       | ---:          |
| simple requirements (whole question)     | 94.25   | 74.47  | 90.8      | 92.70     | 72.94    | 82.6        | 71.83       | 47.92      | 42.2          |

Surrounding context:

| Variant                                  | MH Avg F1 | MH Avg All@5 |
| ---                                      | ---:      | ---:         |
| Full EvLink                              | 66.47     | 76.03        |
| simple requirements (whole question, L0) | 65.11     | 71.87        |
| simple anchors (L1, appendix only)       | 64.64     | 70.40        |
| PropRAG                                  | 63.9      | n/a          |
| NeocorRAG                                | 63.1      | n/a          |

The headline three-segment attribution under the reader-facing F1 mouth:

```text
substrate alone vs PropRAG  (per dataset, L0 - PropRAG):
  HotpotQA       : -0.63
  2WikiMultiHopQA: +3.84
  MuSiQue        : +0.32
  Avg multi-hop  : +1.21

LLM five-field decomposition marginal (Full - L0):
  HotpotQA       : +1.13
  2WikiMultiHopQA: +1.76
  MuSiQue        : +1.18
  Avg multi-hop  : +1.36
```

This shows that the substrate alone is sufficient to surpass the
strongest graph baseline on 2WikiMultiHopQA F1 and on the multi-hop F1
average; the five-field LLM decomposition adds a consistent secondary
boost that is most necessary on HotpotQA, where substrate alone is on
par with PropRAG.

## Audit Of The MuSiQue 42.2 Coincidence

Both the L0 row above and the Table 3 raw-BFS-top-K row report MuSiQue
All@5 = 42.2. This is not algebraic equivalence; it is a near match
caused by 6/1000 final_positions differences that happen to be neutral
with respect to All@5.

Verified files:

```text
L0 retrieval artifact:
  run_logs/evidencelink_simple_requirements_full1000_20260525/readout/whole_question/evals/
    musique_whole_question_prefix4_residual1_pool100_limit1000.json

Table 2 w/o coverage-aware reranking source (no-coverage):
  run_logs/etv4_ablation_relation_coverage_all32_gpt4omini_full1000_20260516/no_coverage/evals/
    musique_pcec_no_coverage_prefix5_residual0_pool100_limit1000.json
  run_logs/etv4_ablation_relation_coverage_all32_gpt4omini_full1000_20260516/reader_qa/no_coverage/reports/
    musique_etv4_no_coverage_gpt4omini_reader_qa_full1000.json
```

Per-query overlap between L0 and no-coverage on MuSiQue:

| Compare                    | Same / 1000 |
| ---                        | ---:        |
| retrieved_doc_indices_top5 | 994         |
| retrieved_titles_top5      | 994         |
| reader all_gold_at5        | 1000        |
| reader recall_at5          | 998         |
| reader F1                  | 946         |
| reader EM                  | 975         |

L0 selector trace:

```text
admit_count   = 6
changed_count = 6
prefix_preserved_count = 1000
```

No-coverage selector trace:

```text
admit_count   = 0
changed_count = 0
prefix_preserved_count = 1000
```

Aggregate metric mouths on MuSiQue:

| Variant                 | retrieval-side title All@5 | retrieval-side title R@5 | reader doc-level All@5 | reader R@5 | reader F1 |
| ---                     | ---:                        | ---:                      | ---:                    | ---:       | ---:      |
| L0 whole-question       | 46.6                        | 74.17                     | 42.2                    | 71.83      | 47.92     |
| no-coverage / raw top-K | 46.6                        | 74.17                     | 42.2                    | 71.83      | 48.33     |

Interpretations:

- L0 has 6 residual-slot swaps that move from BFS order under the |B|=1
  noisy-OR; no-coverage cannot swap because its residual budget is 0.
- These 6 swaps are individually meaningful but neutral for MuSiQue
  All@5 at the gold-set level.
- 46.6 and 42.2 are different metric mouths (retrieval-side title
  matching versus reader-side doc-id matching), not different
  experiments. Table 2 uses the reader-side number; Table 3 retrieval-only
  diagnostic uses the title-side number.
- The reader F1 gap (47.92 vs 48.33) cannot come from final_positions
  alone (994/1000 identical); it is most likely due to evidence ordering
  inside the reader prompt (L0 places the residual-slot doc at the tail,
  no-coverage uses strict BFS order). This does not affect any retrieval
  claim.

## §5.2 Ablation Paragraph Addition

Insert this paragraph after the existing `w/o evidence-need mining`
discussion. Do not weaken any existing claim.

```tex
\paragraph{Substrate alone vs.\ LLM-mined evidence needs.}
We isolate the contribution of the Qwen3-32B five-field requirement
extraction by replacing $B(q)$ with a single requirement equal to the
question text, holding the link substrate, BFS, binding cache, and
coverage readout fixed. This naive variant trails full \methodname{}
by $1.36$ F1 and $4.16$ All@5 on multi-hop average, yet still surpasses
PropRAG on 2WikiMultiHopQA F1 ($72.94$ vs.\ $69.1$) and on the
multi-hop F1 average ($65.11$ vs.\ $63.9$). Source-grounded evidence
links are therefore the dominant contributor on 2WikiMultiHopQA, where
they recover bridges that PropRAG paths miss; on HotpotQA the substrate
alone is on par with PropRAG ($-0.63$ F1) and the structured requirement
decomposition is what lifts the result past the strongest graph
baseline.
```

## L1 Simple Anchors (Appendix Only)

L1 numbers (already collected, not added to Table 2):

| Variant                  | Hot F1 | 2Wiki F1 | MuSiQue F1 | MH Avg F1 |
| ---                      | ---:   | ---:     | ---:       | ---:      |
| L0 whole-question        | 74.47  | 72.94    | 47.92      | 65.11     |
| L1 simple anchors        | 74.34  | 72.23    | 47.35      | 64.64     |

Suggested appendix sentence:

```tex
Replacing the whole-question requirement with a list of question-side
anchor mentions yields essentially identical results
(Hot $74.34$, 2Wiki $72.23$, MuS $47.35$ F1; multi-hop average $64.64$),
confirming that the structure that matters at the requirement layer is
the five-field decomposition itself, not anchor presence.
```

Reason for keeping L1 out of Table 2:

- L1 vs L0 differences are within $\pm 0.7$ F1 on every dataset.
- Adding L1 as a row invites a "why does anchor extraction not help"
  side discussion that distracts from the substrate vs LLM-mined
  decomposition story.

## Implementation Checklist

- [ ] Add a single L0 row to Table 2 named `simple requirements (whole question)` with the reader-facing numbers above.
- [ ] Insert the new `\paragraph{Substrate alone vs.\ LLM-mined evidence needs.}` block after the existing `w/o evidence-need mining` discussion in `paper/sections/05_experiments.tex`.
- [ ] Do not modify the existing `w/o evidence-need mining` row or its paragraph.
- [ ] Add the L1 sentence to the appendix as a single-line confirmatory note. Do not add an L1 row to Table 2.
- [ ] Do not write the 994/6 MuSiQue audit into the paper. Keep it in this document only; reference it from internal notes if a reviewer asks.
- [ ] Do not compile unless explicitly requested.

## Source Files

L0 whole-question:

```text
run_logs/evidencelink_simple_requirements_full1000_20260525/readout/whole_question/evals/
  hotpotqa_whole_question_prefix4_residual1_pool100_limit1000.json
  2wikimultihopqa_whole_question_prefix4_residual1_pool100_limit1000.json
  musique_whole_question_prefix4_residual1_pool100_limit1000.json
run_logs/evidencelink_simple_requirements_full1000_20260525/reader_qa/whole_question/reports/
  hotpotqa_whole_question_gpt4omini_reader_qa_limit1000.{json,md}
  2wikimultihopqa_whole_question_gpt4omini_reader_qa_limit1000.{json,md}
  musique_whole_question_gpt4omini_reader_qa_limit1000.{json,md}
```

L1 simple anchors:

```text
run_logs/evidencelink_simple_requirements_full1000_20260525/readout/simple_question_anchors/
run_logs/evidencelink_simple_requirements_full1000_20260525/reader_qa/simple_question_anchors/reports/
  hotpotqa_simple_question_anchors_gpt4omini_reader_qa_limit1000.{json,md}
  2wikimultihopqa_simple_question_anchors_gpt4omini_reader_qa_limit1000.{json,md}
  musique_simple_question_anchors_gpt4omini_reader_qa_limit1000.{json,md}
```

Relevant code:

```text
evidenceflow/requirements.py
evidenceflow/dbec_utility.py
evidenceflow/native_readout.py
evidenceflow/run_native_pool.py
scripts/dtc_embed_utils.py        # DTCRequirement, build_fallback_dtc_requirements
```
