# CAPS Day-1.5 Candidate Generator v2

- Status: `completed`
- Decision: `STOP_CANDIDATE_V2_RECALL_FAIL`
- LLM endpoint: `http://localhost:8043/v1`
- LLM model: `qwen3-8b-train`
- Include v1 candidates: `False`
- Include string extraction: `False`
- Rows: `200`
- Recall@5: `0.455`
- Recall@10: `0.48`
- Recall@20: `0.485`
- Avg candidate count: `7.87`

## Gates

- Top-5 success: `Recall@5 >= 0.85`
- Top-10 review: `Recall@10 >= 0.9`

Health: `{'available': True, 'status': 200, 'models': ['qwen3-8b-train']}`
