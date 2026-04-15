# GPT-5.4 Late Judge Integration Issue - Review Request

## TL;DR

The GPT-5.4 late rerank judge experiment **did not successfully test the judge's effectiveness** because the judge failed to intervene in ranking on every single query. While API requests returned 200 OK, all late rerank attempts failed with parse errors, resulting in `override_count=0` and final results identical to the heuristic baseline.

---

## What I Implemented

**Added independent late-rerank judge pipeline:**
- Location: `scripts/eval_causal_qwen3.py:66`
- Supports two backends:
  - `responses` (OpenAI Responses API)
  - `chat_completions` (OpenAI Chat Completions API)

**Integrated as state-level reranker:**
- Location: `scripts/eval_causal_qwen3.py:573`
- Operates on completed beam finalists (does not modify beam search)
- Hooks into `bridge_beam + set_closure` pipeline at `scripts/eval_causal_qwen3.py:2461`

**Added CLI parameters:**
- `--setwise_late_rerank_judge_backend`
- `--setwise_late_rerank_judge_model`
- `--setwise_late_rerank_judge_base_url`
- `--setwise_late_rerank_judge_api_key_env`
- `--setwise_late_rerank_judge_reasoning_effort`

---

## Experiments Run

**Datasets:**
- 2WikiMultiHopQA-100
- MuSiQue-100
- HotpotQA-100

**Configuration:**
- Provider: `https://api-vip.codex-for.me/v1`
- Backend: `responses`
- Model: `gpt-5.4`
- Reasoning effort: `xhigh`

**Note:** `chat_completions` backend returns `400 Bad Request` from this provider, so it was not used.

---

## Results

### 2WikiMultiHopQA-100
- Baseline: EM 0.38, F1 0.4332
- Selector: EM 0.47, F1 0.5116
- **Identical to heuristic baseline**
- File: `outputs_step0_general_2wikimultihopqa/eval_reports/setwise_bridge_beam_set_closure_gpt54judge_100_legacy_reserve3_dedup.json`

### MuSiQue-100
- Baseline: EM 0.26, F1 0.3466
- Selector: EM 0.26, F1 0.3312
- **Identical to heuristic baseline**
- File: `outputs_step0_general_musique/eval_reports/setwise_bridge_beam_set_closure_gpt54judge_100_legacy_reserve3_dedup.json`

### HotpotQA-100
- Baseline: EM 0.59, F1 0.7114
- Selector: EM 0.59, F1 0.6948
- **No evidence of judge intervention**
- File: `outputs_step0_general_hotpotqa/eval_reports/setwise_bridge_beam_set_closure_gpt54judge_100_legacy_reserve3_dedup.json`

---

## Critical Evidence: Judge Never Succeeded

**MuSiQue-100:**
```json
"late_rerank_apply_count": 84,
"late_rerank_override_count": 0,
"late_rerank_parse_failure_count": 84,
"late_rerank_error_count": 84
```

**2WikiMultiHopQA-100:**
```json
"late_rerank_apply_count": 40,
"late_rerank_override_count": 0,
"late_rerank_parse_failure_count": 40,
"late_rerank_error_count": 40
```

**HotpotQA-100:**
```json
"late_rerank_apply_count": 37,
"late_rerank_override_count": 0,
"late_rerank_parse_failure_count": 37,
"late_rerank_error_count": 37
```

**Interpretation:**
- Judge requests were sent (not a network issue)
- Every single request failed with parse error
- Final output = heuristic baseline
- **This experiment does NOT test judge effectiveness**

---

## Why I Suspect Provider/Structured-Output Compatibility Issue

**Evidence from logs:**
1. API requests succeed: `POST https://api-vip.codex-for.me/v1/responses 200 OK`
   - Examples: `run_logs/gpt54judge_2wiki100.log:183`
   - Examples: `run_logs/gpt54judge_musique100.log:251`
   - Examples: `run_logs/gpt54judge_hotpot100.log:200`

2. Network, auth, and endpoint are working

3. Current implementation uses: `client.responses.parse(..., text_format=PydanticModel)`

4. **Hypothesis:** Provider returns content that doesn't match OpenAI SDK's structured parse expectations, causing SDK/wrapper layer to throw exceptions

5. `chat_completions` path also fails (provider returns 400)

**Note on confusing log messages:**
- Some logs show `Rerank parse failed on attempt 1...` from `src.hipporag.rerank`
- These are OLD warnings from existing candidate-selection code
- **NOT related to GPT-5.4 late judge failures**
- Real evidence is in `selector_summary`:
  - `late_rerank_apply_count`
  - `late_rerank_override_count`
  - `late_rerank_parse_failure_count`
  - `late_rerank_error_count`

---

## Current Conclusion

**This experiment CANNOT answer:** "Does a stronger judge help?"

**This experiment CAN answer:** "GPT-5.4 + provider responses endpoint failed to produce usable late-rerank decisions under current integration"

**This is NOT model failure. This is integration/API compatibility failure.**

---

## What I Need Reviewers to Check

### Key Code Locations

1. **`scripts/eval_causal_qwen3.py:87`** - `OpenAICompatibleLateRerankJudge` class
2. **`scripts/eval_causal_qwen3.py:139`** - `_infer_with_responses` method
3. **`scripts/eval_causal_qwen3.py:573`** - `rerank_completed_evidence_sets_with_llm` function

### Specific Questions

**Q1: Should we avoid `responses.parse(...)`?**
- Should we switch to `responses.create(...)` and manually extract/validate JSON from raw text?

**Q2: Does this provider's `/responses` implementation only "appear compatible"?**
- Does it actually support structured parse, or just return plain text?
- If the latter, we cannot rely on SDK's parsed object

**Q3: Should the most robust late judge implementation be:**
- Plain text / JSON prompt
- `responses.create(...)`
- Local `json.loads` + repair logic
- **NOT** `text_format=PydanticModel`

---

## Recommended Next Steps

1. **Stop running more 100-sample experiments**
2. **Fix compatibility layer first**
3. **Re-run with conservative implementation:**
   - MuSiQue-40
   - 2Wiki-40
4. **Only proceed to 100-sample runs after `override_count > 0`**

---

## Appendix: Implementation Details

### Current Judge Prompt Structure
```python
prompt = f"""You are evaluating evidence sets for multi-hop question answering.

Question: {query}

Candidate Evidence Sets (ranked by heuristic):
{format_candidates(finalists)}

Task: Rerank these sets by evidence completeness and reasoning chain quality.
Output: JSON with "best_set_index" field.
"""
```

### Current Parse Logic
```python
response = client.responses.parse(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    text_format=LateRerankDecision,  # Pydantic model
    reasoning_effort=reasoning_effort,
)
decision = response.parsed  # Fails here
```

### Proposed Alternative
```python
response = client.responses.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    reasoning_effort=reasoning_effort,
)
raw_text = response.choices[0].message.content
decision = parse_and_repair_json(raw_text)  # Manual extraction
```
