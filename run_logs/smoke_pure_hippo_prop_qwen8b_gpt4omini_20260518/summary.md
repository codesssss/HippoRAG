# Pure HippoRAG / PropRAG Qwen3-8B no-think + GPT-4o-mini Reader

Reader is fixed to GPT-4o-mini; upstream retrieval pools are pure HippoRAG/PropRAG exports, not DAEC selector outputs.

| Dataset | Method | Count | R@5 | R@20 | EM | F1 | Mean reader docs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | hipporag_qwen8b_no_think_top5 | 2 | 0.8313 | 0.9048 | 1.0000 | 1.0000 | 5.0000 |
