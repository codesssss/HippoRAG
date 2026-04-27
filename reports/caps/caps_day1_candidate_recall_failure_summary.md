# CAPS Day-1 Candidate Recall Failure Summary

Date: 2026-04-27

## Decision

```text
STOP_CANDIDATE_RECALL_FAIL
```

CAPS Day 0 passed the NLI verifier sanity check, but Day 1 fails at the first blocking gate: candidate answer recall is not high enough to justify proof search.

Do not proceed to:

```text
oracle-answer proof separability
non-oracle proof-selection pilot
CAPS eval800/eval1000
CAPS expansion to HotpotQA/MuSiQue
```

unless a substantially stronger candidate answer generator is introduced.

## Setup

Data:

```text
2Wiki PropRAG dev fold rows 0-199
PropRAG top20 + Dense top20 union, deduplicated
```

Candidate answer sources:

```text
heuristic title/entity/date/number/yes-no spans
Flan-T5 reader answer on PropRAG top-5
Flan-T5 reader answer on Dense top-5
Flan-T5 reader answer on PropRAG/Dense union top-5
Flan-T5 single-passage local answers for top-12 union docs
Flan-T5 pairwise local answers for top-5 union doc pairs
```

Cache:

```text
data/dpathrag/cache/caps/caps_day1_answer_cache.jsonl
4000 cached reader answer generations
```

## Main Result

After adding single-passage and pairwise reader answer sources:

```text
rows = 200
answer_cap = 20
recall@5 = 0.670
recall@10 = 0.740
recall@20 = 0.750
avg_candidate_count = 20.0
gate = recall@5 >= 0.85
decision = fail
```

The initial version without single/pair reader answers was worse:

```text
recall@5 = 0.595
recall@10 = 0.660
recall@20 = 0.665
```

So local reader answers help, but not enough.

## Pool Oracle Check

The union pool itself is not the main bottleneck:

```text
gold answer string appears in PropRAG/Dense union top60 title/text: 180/200 = 0.900
gold answer string appears in union top60 titles only: 91/200 = 0.455
```

This means the answer is often present somewhere in retrieved text, but the current candidate generator does not extract or promote it reliably.

## Interpretation

CAPS still has a coherent modeling premise:

```text
answer-conditioned proof search may avoid the evidence-edit admission failures seen in D-PathRAG / CEE / C-CEE.
```

But the current Day-1 implementation cannot supply a high-recall candidate answer set. Since proof search can only verify candidates it sees, continuing to proof scoring would conflate proof failure with candidate-generation failure.

The failure is therefore:

```text
candidate answer generation bottleneck, not NLI proof-verifier bottleneck
```

Day 0 remains positive:

```text
cross-encoder/nli-deberta-v3-base
template obligation NLI AUC = 0.903
95% CI = [0.8616, 0.9401]
```

Day 1 stops the current CAPS v1:

```text
multi-source reader/entity candidate recall@20 = 0.750
target recall@5 = 0.850
```

## Paper Use

Use this as a concise negative diagnostic if CAPS is not reopened:

> Answer-conditioned proof search is promising at the verifier level: template proof obligations are separable by an NLI verifier. However, a training-free candidate answer generator over PropRAG/Dense union pools only recalls the gold answer in 67% of dev examples at top-5 and 75% at top-20, despite the answer string appearing in 90% of union top60 contexts. This shifts the bottleneck from proof verification to candidate answer extraction.

## Reopen Conditions

Only reopen CAPS if one of the following is available:

```text
stronger candidate answer generator with recall@5 >= 0.85 on dev
structured answer extraction tuned for 2Wiki relation templates
LLM candidate list generation with controlled cost and no answer leakage
existing model/pipeline that can produce high-recall answer candidates from union contexts
```

Do not continue by only tuning proof scoring, NLI calibration, or proof set cover. Those components are downstream of the failed candidate recall gate.
