# Pure HippoRAG / PropRAG Qwen3-8B no-think + GPT-4o-mini Reader

Reader is fixed to GPT-4o-mini; upstream retrieval pools are pure HippoRAG/PropRAG exports, not DAEC selector outputs.

| Dataset | Method | Count | R@5 | R@20 | EM | F1 | Mean reader docs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | hipporag_qwen8b_no_think_top5 | 1000 | 0.8313 | 0.9048 | 0.5600 | 0.6264 | 5.0000 |
| hotpotqa | hipporag_qwen8b_no_think_top5 | 1000 | 0.9230 | 0.9885 | 0.5960 | 0.7289 | 5.0000 |
| musique | hipporag_qwen8b_no_think_top5 | 1000 | 0.7011 | 0.8662 | 0.3320 | 0.4430 | 5.0000 |
| 2wikimultihopqa | proprag_qwen8b_no_think_top5 | 1000 | 0.9028 | 0.9607 | 0.6050 | 0.6842 | 5.0000 |
| hotpotqa | proprag_qwen8b_no_think_top5 | 1000 | 0.9500 | 0.9925 | 0.6070 | 0.7426 | 5.0000 |
| musique | proprag_qwen8b_no_think_top5 | 1000 | 0.7372 | 0.9007 | 0.3640 | 0.4743 | 5.0000 |
