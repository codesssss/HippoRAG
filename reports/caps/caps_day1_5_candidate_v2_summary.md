# CAPS Day-1.5 Candidate Generator v2 Summary

Date: 2026-04-27

## Decision

```text
STOP_CANDIDATE_V2_RECALL_FAIL
```

The Day-1.5 rescue attempt found working local LLM endpoints and tested an LLM-based candidate answer generator, but candidate recall still does not meet the gate.

Do not proceed to CAPS proof scoring with the current candidate generators.

## Available LLM Endpoint

The originally assumed endpoint `http://localhost:8039/v1` was unavailable. Active local OpenAI-compatible endpoints were found:

```text
http://localhost:8041/v1 -> qwen3-8b-train
http://localhost:8042/v1 -> qwen3-8b-train
http://localhost:8043/v1 -> qwen3-8b-train
http://localhost:8045/v1 -> qwen3-32b-judge
http://localhost:8088/v1 -> Qwen3-32B
```

The run used:

```text
endpoint = http://localhost:8043/v1
model = qwen3-8b-train
```

## Prompt / Parser Fix

Initial smoke exposed Qwen3 think-mode output. The prompt was changed to:

```text
/no_think
Return valid JSON only:
{"answers": ["answer 1", "answer 2"]}
```

The parser now:

```text
strips <think>...</think>
parses JSON {"answers": [...]}
recovers short entity variants from lines like "Andy Summers was born on ..."
```

## Results

### LLM-only candidates

```text
rows = 200
Recall@5 = 0.455
Recall@10 = 0.480
Recall@20 = 0.485
avg candidate count = 7.87
decision = STOP_CANDIDATE_V2_RECALL_FAIL
```

LLM-only candidate generation is much worse than the Day-1 v1 generator.

### LLM + v1 reader candidates + string extraction

```text
rows = 200
Recall@5 = 0.635
Recall@10 = 0.665
Recall@20 = 0.840
avg candidate count = 20.0
decision = STOP_CANDIDATE_V2_RECALL_FAIL
```

The union improves top-20 recall compared with Day-1 v1:

```text
Day-1 v1 final Recall@20 = 0.750
Day-1.5 union Recall@20 = 0.840
```

But the ranking remains too weak:

```text
Day-1.5 union Recall@5 = 0.635
Day-1.5 union Recall@10 = 0.665
```

This fails both planned gates:

```text
Recall@5 >= 0.85 -> fail
Recall@10 >= 0.90 -> fail
```

## Interpretation

Day 1.5 confirms the bottleneck is candidate answer proposal/ranking:

```text
union top60 contains gold answer string in 90% of dev examples
NLI proof verifier AUC is 0.903
candidate generators cannot reliably surface and rank the gold answer
```

The LLM list generator does not solve this. It often produces explanatory phrases or plausible but wrong candidates, and adding it to v1/string extraction improves only top-20 recall while hurting top-rank precision.

This is still not a proof-scoring failure. It is a candidate-answer generation failure.

## Final CAPS Status

```text
CAPS Day0 verifier sanity: pass
CAPS Day1 candidate recall v1: fail
CAPS Day1.5 LLM candidate rescue: fail
CAPS proof search / eval: do not run
```

Reopen CAPS only with a materially different answer proposal mechanism, not another proof scorer:

```text
relation-template answer extraction
supervised/high-recall answer proposal model
dataset-specific structured extraction from evidence triples or Wikidata-like relations
stronger LLM candidate generator verified to reach Recall@10 >= 0.90 on dev
```
