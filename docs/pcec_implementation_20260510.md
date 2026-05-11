# Preservation-Constrained Evidence Composition Implementation

Date: 2026-05-10

Canonical full1000 experiment protocol and final results:

```text
docs/pcec_full1000_protocol_20260511.md
```

## Method Identity

Paper-facing method name:

```text
Preservation-Constrained Evidence Composition
```

Engineering method line:

```text
evidence_transition_graphragv4_composition
```

The method keeps ETv3 variable-flow retrieval as the expander and changes the
reader-facing readout objective:

```text
ETv3 = EvidenceTransition + TopKReadout
PCEC = EvidenceTransition + PrefixResidualReadout
```

## Formulation

PCEC treats multi-hop readout as hard-constrained fixed-budget set selection:

```text
S*_m = argmax_{S subset P, |S| = K} U_b*(S)
       subject to Prefix_m(S0) subset S
```

where:

- `P` is the ET candidate pool.
- `S0` is the ET top-K baseline reader set.
- `K=5` is the reader budget used in this paper.
- `m=4` is the default preservation prefix.
- `b*` is selected by the DBEC best-binding protocol and then frozen.
- `U_b*(S)` is DBEC frozen-binding noisy-OR demand-binding coverage.

For default `K=5,m=4`, this becomes an exact one-slot solver:

```text
x* = argmax_{x in P \\ S0} U_b*(Prefix_4(S0) union {x})
return Prefix_4(S0) union {x*} if it improves U_b*(S0), else S0
```

The implementation maps this solver onto the existing safe DBEC selector with:

```text
--setwise_selector daec_noisyor_safe_llm
--daec_safe_preserve_top_m 4
--daec_safe_max_swaps 1
--daec_safe_min_objective_gain 0.0
--daec_safe_min_swap_gain 1e-6
```

## Implemented Artifacts

New package:

```text
evidence_transition_graphragv4_composition/
```

Key files:

- `contract.py`: method identity and guardrail flags.
- `readout.py`: PCEC trace adapter, summaries, title metrics.
- `run_eval.py`: integrated runner from ET pool to PCEC reader evaluation.
- `pcec_types.py`: native query/readout data boundaries.
- `requirements.py`: frozen historical DBEC requirement provider.
- `dbec_utility.py`: library-level DBEC frozen-binding utility adapter.
- `native_readout.py`: native PrefixResidualReadout implementation.
- `pool_alignment.py`: corpus alignment and HippoRAG passage-id remapping.
- `run_native_pool.py`: native pool runner that reproduces top4/max1 without
  subprocess-running `scripts/eval_causal_qwen3.py`.
- `expander.py`: frozen ETv3 variable-flow expander adapter.
- `run_fresh_e2e.py`: fresh frozen-expansion retrieval runner that writes the
  same reader-compatible report schema as `run_native_pool.py`.
- `README.md`: concise method and command documentation.

New parity tool:

```text
scripts/verify_pcec_parity.py
scripts/verify_pcec_frozen_expander.py
scripts/verify_pcec_fresh_pool_parity.py
scripts/verify_pcec_fresh_vs_native_pool_parity.py
```

New tests:

```text
tests/test_pcec_composition.py
tests/test_pcec_native_readout.py
tests/test_pcec_frozen_expander.py
```

Frozen ETv3 expander snapshot:

```text
evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/
```

This snapshot is the first step toward a native algorithmic E2E PCEC pipeline.
It prevents PCEC from depending on the live ETv3 branch while that branch
continues to change.

## Native Pool Status

`run_eval.py` remains as a Version A bridge for backward-compatible reader
experiments, but the Step 1-3 native pool path is now implemented:

```text
historical ETv3 pool JSON
-> frozen historical DBEC requirements
-> native DBEC utility adapter
-> PCEC PrefixResidualReadout
-> reader-compatible top5 retrieval report
```

The native runner is:

```text
evidence_transition_graphragv4_composition/run_native_pool.py
```

It may call `dtc_embed_utils.select_daec_noisyor_positions` as a library utility.
It does not subprocess-run `scripts/eval_causal_qwen3.py`.

Critical implementation detail: native pool readout must remap pool document IDs
through HippoRAG's `passage_node_key_to_doc_idx`, matching historical
`apply_setwise_selector` behavior.  Directly using `pool_doc_ids` from the pool
JSON misaligns `passage_embeddings[doc_id]` and changes DBEC demand scores.

Reader QA is intentionally not part of the Phase 3 native-pool gate.

## Fresh Frozen E2E Retrieval Status

Phase 4 replaces the historical ETv3 pool JSON with a package-local frozen ETv3
expander adapter, while keeping frozen historical DBEC requirements:

```text
query
-> FrozenETv3Expander
-> frozen historical DBEC requirements
-> native DBEC utility adapter
-> PCEC PrefixResidualReadout
-> reader-compatible top5 retrieval report
```

The expander entry point is:

```text
evidence_transition_graphragv4_composition/expander.py
```

The fresh retrieval runner is:

```text
evidence_transition_graphragv4_composition/run_fresh_e2e.py
```

The expander imports only from:

```text
evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/
```

It does not import the live `evidence_transition_graphragv3_variable_flow`
branch and does not subprocess-run `scripts/eval_causal_qwen3.py`.

## Validation Commands

Dry-run default MuSiQue PCEC:

```bash
/mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  evidence_transition_graphragv4_composition/run_eval.py \
  --dataset musique \
  --limit 1000 \
  --pool-k 100 \
  --prefix-budget-m 4 \
  --dry-run \
  --qwen-disable-thinking
```

Dry-run HotpotQA `m=3` ablation:

```bash
/mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  evidence_transition_graphragv4_composition/run_eval.py \
  --dataset hotpotqa \
  --limit 1000 \
  --pool-k 100 \
  --prefix-budget-m 3 \
  --dry-run \
  --qwen-disable-thinking
```

Parity check:

```bash
/mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  scripts/verify_pcec_parity.py \
  --pcec-json run_logs/evidence_transition_graphragv4_composition/evals/musique_pcec_prefix4_residual1_pool100_limit1000.json \
  --reference-json run_logs/etv3_dbec_additive_only_full1000_20260510/evals/musique_etv3_pool100_dbec_additive_only_limit1000.json \
  --dataset musique \
  --output-json reports/pcec_parity_20260510/musique_pcec_parity.json \
  --output-md reports/pcec_parity_20260510/musique_pcec_parity.md
```

Native pool run:

```bash
/mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  evidence_transition_graphragv4_composition/run_native_pool.py \
  --dataset musique \
  --limit 1000 \
  --pool-k 100 \
  --prefix-budget-m 4 \
  --reader-budget-k 5 \
  --qwen-disable-thinking
```

Frozen expander verification:

```bash
/mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  scripts/verify_pcec_frozen_expander.py
```

Fresh pool parity:

```bash
/mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  scripts/verify_pcec_fresh_pool_parity.py \
  --datasets 2wikimultihopqa,hotpotqa,musique \
  --limit 1000 \
  --pool-k 100 \
  --reader-budget-k 5 \
  --et-candidate-pool-k 200 \
  --fresh-pool-output-root run_logs/pcec_fresh_pool_parity/fresh_pools \
  --output-json run_logs/pcec_fresh_pool_parity/fresh_pool_parity_full1000.json \
  --output-md run_logs/pcec_fresh_pool_parity/fresh_pool_parity_full1000.md \
  --progress-every 50 \
  --jobs 3
```

Fresh frozen E2E retrieval:

```bash
/mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  evidence_transition_graphragv4_composition/run_fresh_e2e.py \
  --dataset musique \
  --max-queries 1000 \
  --candidate-pool-k 100 \
  --reader-budget-k 5 \
  --prefix-budget-m 4 \
  --qwen-disable-thinking
```

Fresh E2E vs native-pool parity:

```bash
/mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  scripts/verify_pcec_fresh_vs_native_pool_parity.py \
  --datasets 2wikimultihopqa,hotpotqa,musique \
  --limit 1000 \
  --pool-k 100 \
  --reader-budget-k 5 \
  --prefix-budget-m 4 \
  --output-json run_logs/pcec_fresh_e2e/fresh_vs_native_pool_parity_full1000.json \
  --output-md run_logs/pcec_fresh_e2e/fresh_vs_native_pool_parity_full1000.md
```

## Acceptance Gates

1. Unit tests pass:

```text
tests/test_pcec_composition.py
tests/test_pcec_native_readout.py
```

2. Dry-run command maps PCEC defaults to:

```text
daec_safe_preserve_top_m = 4
daec_safe_max_swaps = 1
qa_top_k = 5
setwise_pool_k = 100
```

3. Integrated PCEC `prefix4/residual1` has exact top-5 title parity with the
historical top4/max1 artifacts before reader numbers are treated as final.

4. Final full1000 QA should be regenerated from the PCEC runner, not copied
from historical DBEC wrapper outputs.

5. Phase 4 fresh frozen E2E retrieval must have exact top-5 parity with the
native-pool runner before reader QA is run from the fresh pipeline.

## Native Pool Parity Results

All results below are selector/top5 parity only; no reader QA was run in this
implementation step.

| Dataset | Native output | Historical reference | Exact top5 parity | Binding misses | Title-all@5 |
| --- | --- | --- | ---: | ---: | ---: |
| 2WikiMultiHopQA | `run_logs/evidence_transition_graphragv4_composition_native_pool/evals/2wikimultihopqa_pcec_native_pool_prefix4_residual1_pool100_limit1000.json` | `run_logs/etv3_dbec_preservation_grid_full1000_20260510/evals/2wikimultihopqa_etv3_pool100_dbec_top4_max1_limit1000.json` | 1000/1000 | 0 | 0.856 |
| HotpotQA | `run_logs/evidence_transition_graphragv4_composition_native_pool/evals/hotpotqa_pcec_native_pool_prefix4_residual1_pool100_limit1000.json` | `run_logs/etv3_dbec_preservation_grid_full1000_20260510/evals/hotpotqa_etv3_pool100_dbec_top4_max1_limit1000.json` | 1000/1000 | 0 | 0.932 |
| MuSiQue | `run_logs/evidence_transition_graphragv4_composition_native_pool/evals/musique_pcec_native_pool_prefix4_residual1_pool100_limit1000.json` | `run_logs/etv3_dbec_additive_only_full1000_20260510/evals/musique_etv3_pool100_dbec_additive_only_limit1000.json` | 1000/1000 | 0 | 0.485 |

Parity reports:

```text
run_logs/evidence_transition_graphragv4_composition_native_pool/evals/2wikimultihopqa_pcec_native_pool_prefix4_residual1_pool100_limit1000.parity.md
run_logs/evidence_transition_graphragv4_composition_native_pool/evals/hotpotqa_pcec_native_pool_prefix4_residual1_pool100_limit1000.parity.md
run_logs/evidence_transition_graphragv4_composition_native_pool/evals/musique_pcec_native_pool_prefix4_residual1_pool100_limit1000.parity.md
```

## Fresh Frozen E2E Retrieval Results

All results below are retrieval/top5 results only; no reader QA was run in
Phase 4.

Fresh pool parity against historical ETv3 pool JSON:

| Dataset | Fresh pool all-exact | Fresh baseline top5 exact |
| --- | ---: | ---: |
| 2WikiMultiHopQA | 1000/1000 | 1000/1000 |
| HotpotQA | 1000/1000 | 1000/1000 |
| MuSiQue | 1000/1000 | 1000/1000 |

Report artifacts:

```text
run_logs/pcec_fresh_pool_parity/fresh_pool_parity_full1000.json
run_logs/pcec_fresh_pool_parity/fresh_pool_parity_full1000.md
```

Fresh E2E retrieval outputs:

| Dataset | Fresh E2E output | Changed queries | Binding misses | Title-all@5 |
| --- | --- | ---: | ---: | ---: |
| 2WikiMultiHopQA | `run_logs/pcec_fresh_e2e/evals/2wikimultihopqa_pcec_fresh_e2e_prefix4_residual1_pool100_limit1000.json` | 483 | 0 | 0.856 |
| HotpotQA | `run_logs/pcec_fresh_e2e/evals/hotpotqa_pcec_fresh_e2e_prefix4_residual1_pool100_limit1000.json` | 203 | 0 | 0.932 |
| MuSiQue | `run_logs/pcec_fresh_e2e/evals/musique_pcec_fresh_e2e_prefix4_residual1_pool100_limit1000.json` | 522 | 0 | 0.485 |

Fresh E2E vs native-pool parity:

| Dataset | Exact final top5 parity |
| --- | ---: |
| 2WikiMultiHopQA | 1000/1000 |
| HotpotQA | 1000/1000 |
| MuSiQue | 1000/1000 |

Parity artifacts:

```text
run_logs/pcec_fresh_e2e/fresh_vs_native_pool_parity_full1000.json
run_logs/pcec_fresh_e2e/fresh_vs_native_pool_parity_full1000.md
```

Remaining out-of-scope items:

- Fresh LLM requirement decomposition.
- Pool-size sensitivity.
- Extra reranker baselines.
- Paper writing.

## Fresh E2E Reader QA Results

Phase 5 reader QA uses reader-safe retrieval reports under:

```text
run_logs/pcec_fresh_e2e_reader_qa/retrieval_reports_reader_docids/
```

These reports keep the PCEC final title selection unchanged, but ensure
`retrieved_doc_indices_top5` stores external corpus doc ids for reader context
construction.  DBEC-remapped passage ids are preserved separately as
`pcec_dbec_doc_indices_top5`.

Final full1000 reader QA outputs:

```text
run_logs/pcec_fresh_e2e_reader_qa/reports_reader_docids/2wikimultihopqa_pcec_fresh_e2e_reader_qa_full1000.json
run_logs/pcec_fresh_e2e_reader_qa/reports_reader_docids/hotpotqa_pcec_fresh_e2e_reader_qa_full1000.json
run_logs/pcec_fresh_e2e_reader_qa/reports_reader_docids/musique_pcec_fresh_e2e_reader_qa_full1000.json
```

| Dataset | Count | R@5 | All-gold@5 | EM | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2WikiMultiHopQA | 1000 | 0.9500 | 0.8550 | 0.6230 | 0.7009 |
| HotpotQA | 1000 | 0.9630 | 0.9290 | 0.6300 | 0.7480 |
| MuSiQue | 1000 | 0.7291 | 0.4360 | 0.3430 | 0.4454 |

The first attempted reader QA outputs in
`run_logs/pcec_fresh_e2e_reader_qa/reports/` should not be used for
HotpotQA or MuSiQue because they consumed DBEC-remapped doc ids as reader
corpus ids.

## Paper Framing Notes

Use this wording:

```text
We use DBEC's best-binding protocol to select b*, then perform constrained
readout under U_b*.
```

Do not write the implementation as:

```text
max_b U_b(S)
```

unless the code is changed to optimize bindings during readout.

Default `m=4` is scoped to `K=5`.  The paper should frame it as a robust
operating point on the preservation/admission frontier, not a universal
constant.

The failed cross-signal agreement retention remains an analysis result, not a
mainline component.  It should support the claim that ET transition rank is a
high-SNR retention proxy in this expander, rather than motivate weighted fusion.
