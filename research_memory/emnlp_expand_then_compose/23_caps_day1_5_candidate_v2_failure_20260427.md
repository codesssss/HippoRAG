# CAPS Day-1.5 Candidate Generator v2 Failure - 2026-04-27

## Context

CAPS Day 1 failed candidate answer recall:

```text
v1 Recall@5 = 0.670
v1 Recall@10 = 0.740
v1 Recall@20 = 0.750
```

However, the gold answer string appears in PropRAG/Dense union top60 text in:

```text
180/200 = 0.900
```

Therefore Day 1.5 tested whether a local LLM candidate-list generator could repair answer proposal recall.

## Endpoint Discovery

The old assumed endpoint `http://localhost:8039/v1` was not active.

Active local OpenAI-compatible LLM endpoints:

```text
http://localhost:8041/v1 -> qwen3-8b-train
http://localhost:8042/v1 -> qwen3-8b-train
http://localhost:8043/v1 -> qwen3-8b-train
http://localhost:8045/v1 -> qwen3-32b-judge
http://localhost:8088/v1 -> Qwen3-32B
```

The actual run used:

```text
endpoint = http://localhost:8043/v1
model = qwen3-8b-train
```

## Implementation

```text
scripts/caps_day1_5_candidate_v2.py
tests/dpathrag/test_caps_day1_5.py
reports/caps/caps_day1_5_candidate_v2_summary.md
```

The prompt was set to `/no_think` and JSON-only output. The parser:

```text
removes <think>...</think>
parses {"answers": [...]}
extracts short entity variants from explanatory candidates
```

Example fix:

```text
"Andy Summers was born on 31 December 1942"
-> also add "Andy Summers"
```

## Result

LLM-only:

```text
Recall@5 = 0.455
Recall@10 = 0.480
Recall@20 = 0.485
```

LLM + v1 reader candidates + string extraction:

```text
Recall@5 = 0.635
Recall@10 = 0.665
Recall@20 = 0.840
```

The union improves top-20 recall relative to v1, but it fails both gate criteria:

```text
Recall@5 >= 0.85 -> fail
Recall@10 >= 0.90 -> fail
```

## Interpretation

CAPS is not failing because proof verification is weak:

```text
Day0 NLI verifier AUC = 0.903
```

CAPS is failing because candidate answer generation remains too weak:

```text
gold answer in union top60 text = 0.900
best candidate Recall@10 = 0.665
best candidate Recall@20 = 0.840
```

The current LLM candidate-list prompt does not solve the problem. It often outputs explanatory phrases or plausible alternatives, and when unioned with v1/string candidates it raises recall only deep in the list, not at the top.

## Decision

```text
CAPS proof search should not run with the current candidate generators.
```

Do not continue by tuning:

```text
NLI proof scoring
proof set cover
answer prior
non-oracle proof selection
```

Those are downstream of candidate recall.

Only reopen CAPS if a materially stronger candidate proposal mechanism reaches:

```text
Recall@10 >= 0.90 on dev
```

Likely reopen routes:

```text
relation-template answer extraction
supervised answer proposal
dataset-specific structured extraction from 2Wiki relation types
stronger LLM candidate generator with verified high recall
```
