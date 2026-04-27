# CPAG Day-1 Agreement Graph Failure - 2026-04-27

## Context

CPAG was tested as a training-free LLM-assisted method line after LCPS was rejected as too brittle. The intended paper-flavored hypothesis was:

```text
LLM local extraction + robust graph operator
not LLM reasoning + brittle program execution
```

CPAG builds a query-local proposition/entity agreement graph over multi-pool retrieval results and selects evidence by agreement closure.

## Implementation

```text
script: scripts/run_cpag_agreement.py
test:   tests/dpathrag/test_cpag_agreement.py
report: reports/cpag/cpag_day1_agreement_gate.md
json:   reports/cpag/cpag_day1_agreement_gate.json
```

The implementation uses existing Qwen OpenIE cache as the LLM-local extraction artifact:

```text
outputs/2wikimultihopqa/openie_results_ner_qwen3-8b.json
```

No task-supervised training was used.

## Setup

```text
dataset: 2Wiki dev fold 0, first 200 queries
pools: PropRAG top20 + Dense top20
BM25: not available in local dpathrag cache, so not used
dedup cap: 60 docs/query
average query-local pool size: 31.15
average cross-pool docs/query: 8.835
average agreed entities/query: 60.045
reader: data/dpathrag/models/flan_t5_base_gold_k5_5k
```

Variants:

```text
proprag_rank: PropRAG top5
dense_rank: Dense top5
rrf: reciprocal-rank fusion over PropRAG/Dense
cpag_pure: pure greedy agreement coverage
cpag: anchor-first agreement coverage
```

The first smoke run showed that pure agreement chased high-degree family/location hubs. CPAG was therefore made conservative:

```text
select query-anchor documents first
then greedily close proposition/entity agreement coverage
```

`cpag_pure` was retained as a diagnostic variant.

## Support Results

| Variant | Support Recall | Support Complete | Selected Gold |
|---|---:|---:|---:|
| PropRAG rank | 0.8888 | 0.7050 | 2.155 |
| Dense rank | 0.7712 | 0.4900 | 1.840 |
| RRF | 0.7650 | 0.4900 | 1.820 |
| CPAG pure | 0.5687 | 0.3100 | 1.305 |
| CPAG anchor-first | 0.7412 | 0.4700 | 1.735 |

CPAG vs PropRAG rank:

```text
support_complete delta: -0.235
added_gold: 3
added_non_gold: 87
non_gold/gold: 29.0
```

## Reader Results

| Variant | EM | F1 | Support Recall | Support Complete |
|---|---:|---:|---:|---:|
| PropRAG rank | 0.4100 | 0.4720 | 0.8888 | 0.7050 |
| RRF | 0.3650 | 0.4162 | 0.7650 | 0.4900 |
| CPAG pure | 0.3200 | 0.3580 | 0.5687 | 0.3100 |
| CPAG anchor-first | 0.3500 | 0.3897 | 0.7412 | 0.4700 |

Reader F1 delta:

```text
CPAG anchor-first vs PropRAG rank: -0.0823
CPAG pure vs PropRAG rank: -0.1140
RRF vs PropRAG rank: -0.0558
```

## Key Diagnostic

Cross-pool document presence is discriminative in isolation:

```text
gold-vs-non-gold cross-pool count AUC = 0.772391
```

However, this local signal does not produce a useful evidence assembly operator. The agreement graph has many high-degree wrong nodes, especially family/location/movie-neighbor hubs. Pure agreement selection chases those hubs. Anchor-first selection reduces the collapse but still replaces too many PropRAG gold supports with non-gold cross-pool neighbors.

This is the same high-level pattern observed in previous branches:

```text
local signal exists
global selection object fails
reader F1 drops when non-gold imports increase
```

## Implementation Audit

The initial CPAG result had one suspicious symptom:

```text
RRF support_complete = 0.490
PropRAG support_complete = 0.705
delta = -0.215
```

This could have indicated a bug in doc identity alignment, deduplication, RRF, or support metrics. A dedicated audit was added:

```text
script: scripts/audit_cpag_implementation.py
test: tests/dpathrag/test_cpag_audit.py
report: reports/cpag/audit200/cpag_implementation_audit.md
json: reports/cpag/audit200/cpag_implementation_audit.json
```

Audit checks:

```text
doc_id/title overlap between PropRAG top20 and Dense top20
manual RRF score vs implementation output
manual support_complete vs reported support_complete
cross-pool count for known shared docs inside CPAG graph
```

Full dev200 audit:

```text
sample_queries = 200
avg_doc_id_overlap@20 = 8.84
avg_title_overlap@20 = 8.835
all_checks_pass = true
failure_count = 0
warning_count = 15
prop_top1_rrf_rank_gt3_count = 12
```

Conclusion:

```text
The CPAG/RRF failure is not an implementation artifact.
```

The RRF drop is a real methodological finding. Standard RRF over-rewards documents shared by PropRAG and Dense, and those shared documents are often non-gold lexical/semantic neighbors. In 12/200 dev queries, the PropRAG top-1 document is pushed below RRF rank 3 because several cross-pool shared distractors receive two reciprocal-rank contributions.

This validates the CPAG failure interpretation:

```text
cross-pool agreement is locally discriminative,
but agreement is not selective enough to preserve PropRAG gold supports.
```

## Per-Type Pattern

CPAG is especially bad on the slice where PropRAG still has difficult but useful structure:

```text
bridge_comparison:
  PropRAG support_complete = 0.2766
  CPAG support_complete = 0.0000

comparison:
  PropRAG support_complete = 1.0000
  CPAG support_complete = 0.9608

compositional:
  PropRAG support_complete = 0.7595
  CPAG support_complete = 0.4557

inference:
  PropRAG support_complete = 0.7391
  CPAG support_complete = 0.3913
```

The only slice where CPAG nearly preserves support is comparison, where question anchors are explicit and both gold pages are often already in top ranks. It does not recover the hard bridge/compositional residual cases.

## Interpretation

The CPAG hypothesis was:

```text
gold support facts should have stronger cross-pool proposition/entity agreement than lexical hard negatives
```

The measured result is subtler:

```text
cross-pool count separates gold from non-gold weakly-to-moderately,
but agreement closure is not selective enough to assemble support sets.
```

Reason:

```text
agreement is not proof
cross-pool redundancy is not specificity
entity co-occurrence creates hubs
high-degree wrong propositions outcompete missing gold supports
```

This means CPAG is not the next HippoRAG/PropRAG-style breakthrough. It has the right broad paradigm, but the particular agreement operator is not aligned with the residual error regime after strong PropRAG.

## Decision

```text
STOP_CPAG_AGREEMENT_FAIL
```

Do not continue CPAG by:

```text
hand-tuning agreement weights
adding more heuristic penalties
adding BM25 only to hope consensus improves
turning CPAG into another learned reranker without a new formulation
```

The only plausible reopen condition is a different agreement object, such as:

```text
typed relation-level agreement with calibrated entity canonicalization
claim contradiction/specificity filtering
reader-robust proposition-led prompting after evidence selection is fixed
```

For the current paper cycle, CPAG should be included as a negative training-free LLM-operator diagnostic.

## Paper-Ready Finding

We tested a training-free multi-pool proposition agreement graph as a robust alternative to brittle program execution and learned edit admission. Although cross-pool document presence is moderately discriminative for gold support (`AUC=0.772`), deterministic agreement closure fails as an evidence assembly operator: it adds only `3` gold supports while importing `87` non-gold documents relative to PropRAG top-5, reducing support-complete from `0.705` to `0.470` and reader F1 from `0.472` to `0.390`. This shows that proposition/entity agreement is not sufficient to solve the residual hard-negative problem on a strong PropRAG substrate.
