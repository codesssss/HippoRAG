# CAPS Day-1.5 Candidate Generator v2

- Status: `completed`
- Decision: `STOP_CANDIDATE_V2_RECALL_FAIL`
- LLM endpoint: `http://localhost:8043/v1`
- LLM model: `qwen3-8b-train`
- Include v1 candidates: `True`
- Include string extraction: `True`
- Rows: `200`
- Recall@5: `0.635`
- Recall@10: `0.665`
- Recall@20: `0.84`
- Avg candidate count: `20.0`

## Gates

- Top-5 success: `Recall@5 >= 0.85`
- Top-10 review: `Recall@10 >= 0.9`

Health: `{'available': True, 'status': 200, 'models': ['qwen3-8b-train']}`
