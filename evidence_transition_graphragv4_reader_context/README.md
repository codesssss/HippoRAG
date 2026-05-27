# Evidence Transition GraphRAG v4 Reader Context

This folder is a diagnostic V4 line over frozen ETv3 retrieval.  It does not
change ETv3 candidate generation or ETv3 top5 selection.

## Boundary

| Item | Setting |
| --- | --- |
| Folder | `evidence_transition_graphragv4_reader_context` |
| Method line | `evidence_transition_graphragv4_reader_context` |
| Base retriever | `evidence_transition_graphragv3_variable_flow` |
| Retrieval top5 | frozen ETv3 top5 |
| Reader context | frozen top5 + ETv3 candidate prefix, unique to `k=10` by default |
| Graph readout change | none |
| Gate / dataset routing | none |

This version exists because MuSiQue 4-hop diagnostics showed that graph-only
coverage readout variants fail, while many answer-bearing passages sit in
candidate positions 6-10.  Treat this as a reader/budget diagnostic, not as a
claim that ETv3 graph selection has been fixed.

## Usage

```bash
env PYTHONPATH=. /mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  evidence_transition_graphragv4_reader_context/run_reader_qa.py \
  --retrieval-reports run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_limit100_20260509/musique/reports/musique_evidence_transition_graphragv3_variable_flow_retrieval.json \
  --reader-context-k 10 \
  --max-queries 100 \
  --llm-name qwen3-8b-train \
  --llm-base-url http://localhost:8043/v1 \
  --qwen-disable-thinking \
  --save-dir run_logs/evidence_transition_graphragv4_reader_context_qwen8b_nv2_limit100_20260509/reader_runtime \
  --output-json run_logs/evidence_transition_graphragv4_reader_context_qwen8b_nv2_limit100_20260509/reports/evidence_transition_graphragv4_reader_context_qa.json \
  --output-md run_logs/evidence_transition_graphragv4_reader_context_qwen8b_nv2_limit100_20260509/reports/evidence_transition_graphragv4_reader_context_qa.md
```

The wrapper writes transformed QA-input reports under
`<output-json-parent>/v4_reader_context_inputs/`, then calls ETv3's package-local
reader-only QA runner.  The first five reader passages remain the frozen ETv3
top5, so top5 retrieval accounting is unchanged.

## Current Limit100 Diagnostic

On MuSiQue limit100 with Qwen3-8B reader:

| subset | ETv3 EM | ETv3 F1 | V4 reader-context EM | V4 reader-context F1 |
| --- | ---: | ---: | ---: | ---: |
| all | 0.3600 | 0.4282 | 0.3700 | 0.4247 |
| 2-doc | 0.5000 | 0.5837 | 0.4792 | 0.5486 |
| 3-doc | 0.3000 | 0.3800 | 0.3000 | 0.3711 |
| 4-doc | 0.1364 | 0.1545 | 0.2273 | 0.2273 |

The gain is concentrated in 4-hop queries and comes with F1 loss on shorter
queries.  This supports reader/budget as a real bottleneck, but as a global
policy it is diagnostic rather than a mainline improvement.  The remaining
failure still points to dependency-binding rather than graph-order tuning.
