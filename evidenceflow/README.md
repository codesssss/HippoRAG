# EvidenceFlow

This package is the canonical implementation of the current EvidenceFlow
ETv4 + PCEC retrieval stack.  Source-grounded OpenIE units certify
query-local document transition graphs, and PCEC performs the fixed-budget
evidence-coverage readout that produces the reader top-5.

Canonical full1000 protocol and results:

```text
docs/pcec_full1000_protocol_20260511.md
```

## Boundary

| Item | Setting |
| --- | --- |
| Folder | `evidenceflow` |
| Compatibility import | `evidence_transition_graphragv4_composition` |
| Paper-facing name | Preservation-Constrained Evidence Composition |
| Base expander | frozen ETv3 variable-flow snapshot |
| Readout | `prefix_residual_admission` |
| Default reader budget | `K=5` |
| Default preservation prefix | `m=4` |
| Default residual budget | `K-m=1` |
| Candidate pool | ETv3 pool100 by default |
| Admission objective | DBEC frozen-binding noisy-OR marginal coverage |
| Ordering | Preserve ET prefix order; admitted residual fills the displaced slot |

EvidenceFlow is not a fact-node PageRank implementation in this mainline.  The
frozen expander admits a compact document graph from dense/textual seeds and
source-grounded OpenIE transition witnesses; PCEC then optimizes the final
reader prefix.  PCEC is not a weighted reranker and not a dataset router.  It is
a train-free fixed-budget composition layer:

```text
S*_m = argmax_{S subset P, |S| = K} U_b*(S)
       subject to Prefix_m(S0) subset S
```

For the default `K=5,m=4`, the constrained problem has a one-slot exact solver:
keep the ET top-4 prefix and admit one residual document only when the frozen
DBEC utility improves over the ET top-5 baseline.

## Run

Dry-run the default MuSiQue command:

```bash
python evidenceflow/run_eval.py \
  --dataset musique \
  --limit 1000 \
  --pool-k 100 \
  --prefix-budget-m 4 \
  --dry-run
```

Use a historical ETv3 pool and binding cache explicitly:

```bash
python evidenceflow/run_eval.py \
  --dataset 2wikimultihopqa \
  --limit 1000 \
  --pool-json run_logs/etv3_dbec_latest_full1000_20260510/pools/2wikimultihopqa_etv3_pool100_limit1000.json \
  --binding-cache-path run_logs/etv3_dbec_latest_full1000_20260510/evals/2wikimultihopqa_etv3_pool100_dbec_latest.binding_cache.json \
  --qwen-disable-thinking
```

The output JSON is postprocessed with `pcec_contract`, `pcec_config`,
`pcec_summary`, and per-query `pcec_readout` traces.

## Native Pool Readout

Run native PCEC readout on an existing ETv3 pool without subprocess-running the
legacy DBEC evaluator:

```bash
python evidenceflow/run_native_pool.py \
  --dataset musique \
  --limit 1000 \
  --pool-k 100 \
  --prefix-budget-m 4 \
  --reader-budget-k 5 \
  --qwen-disable-thinking
```

The native runner consumes:

- historical ETv3 pool JSON,
- frozen DBEC requirement report,
- DBEC binding cache,
- HippoRAG passage embeddings/entities.

It then writes a reader-compatible top-5 report with per-query `pcec_readout`
traces.  Pool document IDs are remapped through HippoRAG's
`passage_node_key_to_doc_idx`; using raw pool JSON doc IDs directly will
misalign DBEC passage embeddings.

Verified native-pool parity on 2026-05-10:

| Dataset | Exact top5 parity |
| --- | ---: |
| 2WikiMultiHopQA | 1000/1000 |
| HotpotQA | 1000/1000 |
| MuSiQue | 1000/1000 |

## Fresh Frozen E2E Retrieval

Run the Phase 4 native retrieval pipeline:

```bash
python evidenceflow/run_fresh_e2e.py \
  --dataset musique \
  --max-queries 1000 \
  --candidate-pool-k 100 \
  --reader-budget-k 5 \
  --prefix-budget-m 4 \
  --qwen-disable-thinking
```

This path replaces the historical ETv3 pool JSON with
`FrozenETv3Expander`, while keeping frozen historical DBEC requirements:

```text
query
-> FrozenETv3Expander
-> frozen historical DBEC requirements
-> native DBEC utility adapter
-> PCEC PrefixResidualReadout
-> reader-compatible top5 retrieval report
```

It imports only from the package-local frozen snapshot under
`evidenceflow/frozen_etv3_variable_flow/`; it
does not import the live ETv3 branch and does not subprocess-run
`scripts/eval_causal_qwen3.py`.

Legacy imports and top-level runner paths under
`evidence_transition_graphragv4_composition/` are compatibility shims.  New
code should import or execute `evidenceflow/...` directly.

Verified Phase 4 parity:

| Dataset | Fresh pool parity | Fresh vs native top5 parity | Changed queries | Binding misses | Title-all@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2WikiMultiHopQA | 1000/1000 | 1000/1000 | 483 | 0 | 0.856 |
| HotpotQA | 1000/1000 | 1000/1000 | 203 | 0 | 0.932 |
| MuSiQue | 1000/1000 | 1000/1000 | 522 | 0 | 0.485 |

## Fresh E2E Reader QA

Reader QA must use reports whose `retrieved_doc_indices_top5` are external
corpus doc ids, not DBEC-remapped passage ids.  The reader-safe reports are:

```text
run_logs/pcec_fresh_e2e_reader_qa/retrieval_reports_reader_docids/
```

Final full1000 reader QA outputs:

```text
run_logs/pcec_fresh_e2e_reader_qa/reports_reader_docids/
```

| Dataset | R@5 | All-gold@5 | EM | F1 |
| --- | ---: | ---: | ---: | ---: |
| 2WikiMultiHopQA | 0.9500 | 0.8550 | 0.6230 | 0.7009 |
| HotpotQA | 0.9630 | 0.9290 | 0.6300 | 0.7480 |
| MuSiQue | 0.7291 | 0.4360 | 0.3430 | 0.4454 |

Fresh LLM requirement decomposition, pool-size sensitivity, and extra reranker
baselines remain outside this gate.
