# Evidence Transition GraphRAG v3 + DBEC Latest

This folder is an isolated combination line.  Its default is now the
baseline-stable DBEC projection, because the full-rebuild DBEC selector hurt
MuSiQue 2-hop queries.

| Item | Setting |
| --- | --- |
| Folder | `evidence_transition_graphragv3_dbec_latest` |
| Method line | `evidence_transition_graphragv3_dbec_latest` |
| Base retriever | frozen `evidence_transition_graphragv3_variable_flow` |
| Candidate pool | ETv3 top5 prefix, then ETv3 candidate universe to pool@100 |
| Selector | `daec_noisyor_safe_llm` |
| Reader context | selector top5 |
| Identifiability gate | off |
| AG-STO graph prior | off |
| Dataset routing / weighted fusion | none |
| Projection | start from ETv3 top5, apply strict positive DBEC-objective swaps |

The purpose is to test whether DBEC/DAEC can rescue the parts of ETv3 where the
candidate universe has support evidence but ETv3's native top5 composition
misses it.  It is not an ETv3 retrieval change.

The previous full-rebuild selector is still available with
`--selector daec_noisyor_llm` for reproduction.

## Usage

If the ETv3 pool JSON already exists:

```bash
env PYTHONPATH=. /mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  evidence_transition_graphragv3_dbec_latest/run_eval.py \
  --dataset musique \
  --limit 100 \
  --pool-json run_logs/etv3_dbec_latest_limit100_20260510/pools/musique_etv3_pool100_limit100.json \
  --output-root run_logs/etv3_dbec_latest_limit100_20260510 \
  --llm-name qwen3-8b \
  --llm-request-name qwen3-8b-train \
  --llm-base-url http://localhost:8043/v1 \
  --qwen-disable-thinking \
  --embedding-name VLLM/nvidia/NV-Embed-v2 \
  --embedding-base-url http://localhost:8019/v1/embeddings
```

If the pool needs to be exported first, provide the frozen ETv3 retrieval report
and matching OpenIE file:

```bash
env PYTHONPATH=. /mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  evidence_transition_graphragv3_dbec_latest/run_eval.py \
  --dataset musique \
  --limit 100 \
  --output-root run_logs/etv3_dbec_latest_limit100_20260510 \
  --retrieval-report run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_limit100_20260509/musique/reports/musique_evidence_transition_graphragv3_variable_flow_retrieval.json \
  --openie-results run_logs/evidence_transition_qwen8b_nv2_limit100_musique_20260509/musique/index/openie_results_ner_qwen3-8b-train.json \
  --llm-name qwen3-8b \
  --llm-request-name qwen3-8b-train \
  --llm-base-url http://localhost:8043/v1 \
  --qwen-disable-thinking \
  --embedding-name VLLM/nvidia/NV-Embed-v2 \
  --embedding-base-url http://localhost:8019/v1/embeddings
```

## Current MuSiQue Limit100 Result

Qwen3-8B reader, NV-Embed-v2 embeddings, ETv3 pool@100:

| subset | ETv3 EM | ETv3 F1 | full-rebuild DBEC F1 | stable DBEC EM | stable DBEC F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 0.3600 | 0.4282 | 0.4382 | 0.3900 | 0.4547 |
| 2-doc | 0.5000 | 0.5837 | 0.5489 | 0.5208 | 0.5987 |
| 3-doc | 0.3000 | 0.3800 | 0.4356 | 0.3333 | 0.4111 |
| 4-doc | 0.1364 | 0.1545 | 0.2000 | 0.1818 | 0.2000 |

The stable projection removes the 2-hop loss: the 2-doc slice moves from
`-0.0347` F1 under full rebuild to `+0.0150` F1.  It also improves overall F1
from `0.4382` to `0.4547`.  The 4-hop bottleneck is still not solved; stable
DBEC only lifts 4-doc F1 to `0.2000`.

There is also a more conservative thresholded diagnostic
(`--daec-safe-min-objective-gain 0.02 --daec-safe-min-swap-gain 0.01`) with
overall F1 `0.4575` and 4-doc F1 `0.2455`.  Keep that as a diagnostic until it
is validated beyond MuSiQue limit100.
