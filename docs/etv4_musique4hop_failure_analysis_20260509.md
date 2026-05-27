# ETv4 MuSiQue 4-Hop Failure Analysis

Date: 2026-05-09

## Scope

This note diagnoses why `evidence_transition_graphragv3_variable_flow` improves
MuSiQue limit100 only weakly and leaves 4-hop QA unchanged.  ETv3 is treated as
frozen.  All proposed V4 changes must be separate from ETv3.

## Baselines

Config:

- Dataset: MuSiQue limit100
- LLM reader: `qwen3-8b-train`
- Reader endpoint: `http://localhost:8043/v1`
- Embedding substrate: NV-Embed-v2
- ETv3 report:
  `run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_limit100_20260509/musique/reports/musique_evidence_transition_graphragv3_variable_flow_retrieval.json`

ETv3 reader-only QA:

| subset | EM | F1 | R@5 | answer-string@5 |
| --- | ---: | ---: | ---: | ---: |
| all | 0.3600 | 0.4282 | 0.7092 | 0.6000 |
| 2-doc | 0.5000 | 0.5837 | 0.8333 | 0.7083 |
| 3-doc | 0.3000 | 0.3800 | 0.6889 | 0.6333 |
| 4-doc | 0.1364 | 0.1545 | 0.4659 | 0.3182 |

## What Failed

A direct coverage-style readout over the ETv3 candidate universe was tested as
a diagnostic.  Strategies included query-token coverage, anchored coverage,
edge-first coverage, flow-first coverage, and source/graph interleaving.  None
improved the frozen ETv3 readout.

| subset | current R@5 | best diagnostic R@5 | conclusion |
| --- | ---: | ---: | --- |
| 2-doc | 0.8333 | 0.6250 | coverage readout breaks shallow cases |
| 3-doc | 0.6889 | 0.4889 | coverage readout breaks 3-hop cases |
| 4-doc | 0.4659 | 0.3523 | coverage readout does not solve long chains |
| all | 0.7092 | 0.5242 | not viable as V4 mainline |

This falsifies the simple story that ETv4 only needs a better static
`top5 = coverage(source_prior, graph_tail)` objective.

## Root Cause

For MuSiQue 4-doc queries, ETv3 candidate200 has headroom but final top5 cannot
choose the correct branch.

| diagnostic | value |
| --- | ---: |
| candidate200 mean gold recall | 0.9205 |
| candidate200 all-gold | 16/22 |
| source-prior top5 mean gold recall | 0.4318 |
| graph/admissible tail mean gold recall | 0.4886 |
| source-prior union graph tail mean gold recall | 0.9205 |
| final top5 mean gold recall | 0.4659 |
| final top5 all-gold | 0/22 |

Feature audit on 4-hop documents shows why static graph/readout objectives fail:
selected non-gold documents look stronger than missed gold under the available
unsupervised signals.

| document group | candidate rank | query-token count | degree | evidence edges | role-bridge edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| selected gold | 2.29 | 3.39 | 22.66 | 4.93 | 6.51 |
| missed gold | 40.73 | 2.30 | 23.90 | 6.10 | 9.78 |
| selected non-gold | 8.16 | 3.10 | 27.04 | 8.81 | 11.15 |
| non-gold pool | 102.47 | 1.26 | 14.95 | 3.79 | 5.15 |

The hard negatives are not weak.  They are often higher ranked, more lexical,
and more graph-connected than the missed gold.  Therefore a clean graph-only
readout has no reliable local signal to prefer the missed support.

Manual examples confirm the mechanism:

- Query 30 requires composing `Springfield` from the school branch with
  `Illinois` from the screenwriter branch, then selecting `Springfield,
  Illinois`.  ETv3 instead keeps high-rank state/city distractors such as
  Missouri or Indiana.
- Query 91 requires resolving `Erskine College -> South Carolina -> Columbia ->
  Forest Acres -> Richland County`.  ETv3 selects lexical distractors such as
  `College Place, Washington`.

This is a latent dependency-binding / branch-consistency failure, not a lack of
edge strength.

## Reader-Budget Diagnostic

Since answer evidence often appears shortly after top5, a reader-context
diagnostic was run:

`reader_context = unique(current_ETv3_top5 + ETv3_candidate_prefix, limit=10)`

The top5 retrieval metrics are unchanged, but reader QA uses 10 passages.

| subset | ETv3 EM | ETv3 F1 | reader@10 EM | reader@10 F1 | effect |
| --- | ---: | ---: | ---: | ---: | --- |
| all | 0.3600 | 0.4282 | 0.3700 | 0.4247 | EM +0.0100, F1 -0.0035 |
| 2-doc | 0.5000 | 0.5837 | 0.4792 | 0.5486 | clear F1 drop |
| 3-doc | 0.3000 | 0.3800 | 0.3000 | 0.3711 | slight F1 drop |
| 4-doc | 0.1364 | 0.1545 | 0.2273 | 0.2273 | clear 4-hop gain |

This supports a reader/budget direction, but it is not a full solution.  It
raises 4-hop QA by exposing mid-rank answer evidence, while hurting shorter
queries because extra context adds noise.  As a global policy it is diagnostic,
not a mainline improvement.

## V4 Decision

Do not turn the failed coverage-style readout into ETv4.

The clean conclusion is:

1. ETv3 candidate generation has useful 4-hop headroom.
2. ETv3 final top5 fails because dependency binding is unresolved.
3. Static graph strength, query-token coverage, and edge coverage cannot
   reliably distinguish missed gold from hard non-gold.
4. Reader-context expansion helps 4-hop but is only a diagnostic unless the
   paper explicitly changes the evaluation contract from fixed top5 evidence
   selection to larger reader context.

The honest V4 candidates are:

- `ETv4-reader-context`: keep ETv3 retrieval frozen, expose a larger deterministic
  reader context such as `current top5 + candidate prefix to 10`.  This is a
  reader/budget variant, not a better top5 selector.
- `ETv4-state-binding`: change the algorithmic object from document readout to
  dependency-state binding.  This requires preserving branch state or using a
  DBEC-like binding module; it should not be presented as a graph-order tweak.

Current data supports `ETv4-reader-context` as a modest diagnostic and supports
`ETv4-state-binding` as the deeper research direction.  It does not support a
new graph-only coverage assembly mainline.
