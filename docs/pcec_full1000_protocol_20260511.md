# PCEC Full1000 Experiment Protocol

Date: 2026-05-11

This document is the canonical record for the PCEC full1000 retrieval and
reader-QA experiment.  It records the valid artifacts, the exact protocol, the
known invalid artifacts, and the comparison against ETv3, ProPRAG, and HippoRAG.

## Scope

Method:

```text
Preservation-Constrained Evidence Composition (PCEC)
```

Engineering line:

```text
evidence_transition_graphragv4_composition
```

Default configuration:

```text
candidate_pool_k      = 100
et_candidate_pool_k   = 200
reader_budget_k       = 5
prefix_budget_m       = 4
residual_budget       = 1
reader_model          = qwen3-8b-train
embedding_model       = nvidia/NV-Embed-v2
fresh_requirement     = no
requirement_provider  = frozen historical DBEC report
```

Pipeline:

```text
query
-> FrozenETv3Expander
-> frozen historical DBEC requirements
-> native DBEC utility adapter
-> PCEC PrefixResidualReadout
-> reader-safe top5 report
-> Qwen3-8B reader QA
```

The expander is package-local and frozen:

```text
evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/
```

The PCEC fresh path must not import the live
`evidence_transition_graphragv3_variable_flow` branch and must not subprocess-run
`scripts/eval_causal_qwen3.py`.

## Critical ID Boundary

There are two document-id spaces:

```text
external corpus doc id       -> used by reader QA to fetch passages
HippoRAG/DBEC passage doc id -> used internally by DBEC utility scoring
```

For reader QA, `retrieved_doc_indices_top5` must contain external corpus doc
ids.  DBEC-remapped ids are preserved separately as:

```text
pcec_dbec_doc_indices_top5
```

Do not use these first-attempt reader QA outputs for HotpotQA or MuSiQue:

```text
run_logs/pcec_fresh_e2e_reader_qa/reports/
```

They consumed DBEC-remapped ids as reader corpus ids, producing invalid
HotpotQA/MuSiQue retrieval metrics.  The valid final reader QA outputs are:

```text
run_logs/pcec_fresh_e2e_reader_qa/reports_reader_docids/
```

## Artifact Registry

Fresh E2E retrieval reports:

```text
run_logs/pcec_fresh_e2e/evals/2wikimultihopqa_pcec_fresh_e2e_prefix4_residual1_pool100_limit1000.json
run_logs/pcec_fresh_e2e/evals/hotpotqa_pcec_fresh_e2e_prefix4_residual1_pool100_limit1000.json
run_logs/pcec_fresh_e2e/evals/musique_pcec_fresh_e2e_prefix4_residual1_pool100_limit1000.json
```

Reader-safe retrieval reports:

```text
run_logs/pcec_fresh_e2e_reader_qa/retrieval_reports_reader_docids/2wikimultihopqa_pcec_fresh_e2e_prefix4_residual1_pool100_limit1000.json
run_logs/pcec_fresh_e2e_reader_qa/retrieval_reports_reader_docids/hotpotqa_pcec_fresh_e2e_prefix4_residual1_pool100_limit1000.json
run_logs/pcec_fresh_e2e_reader_qa/retrieval_reports_reader_docids/musique_pcec_fresh_e2e_prefix4_residual1_pool100_limit1000.json
```

Final reader QA outputs:

```text
run_logs/pcec_fresh_e2e_reader_qa/reports_reader_docids/2wikimultihopqa_pcec_fresh_e2e_reader_qa_full1000.json
run_logs/pcec_fresh_e2e_reader_qa/reports_reader_docids/hotpotqa_pcec_fresh_e2e_reader_qa_full1000.json
run_logs/pcec_fresh_e2e_reader_qa/reports_reader_docids/musique_pcec_fresh_e2e_reader_qa_full1000.json
```

Parity reports:

```text
run_logs/pcec_fresh_pool_parity/fresh_pool_parity_full1000.json
run_logs/pcec_fresh_pool_parity/fresh_pool_parity_full1000.md
run_logs/pcec_fresh_e2e/fresh_vs_native_pool_parity_full1000.json
run_logs/pcec_fresh_e2e/fresh_vs_native_pool_parity_full1000.md
```

Baseline comparison sources:

```text
run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_full_20260510/reports/
run_logs/agsto_qwen8b_nvembed_sync_full1000_20260505/summary_8b.md
```

## Protocol

### 1. Verify PCEC Unit Tests

```bash
/mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python -m pytest \
  tests/test_pcec_frozen_expander.py \
  tests/test_pcec_composition.py \
  tests/test_pcec_native_readout.py
```

Expected:

```text
12 passed
```

### 2. Verify Fresh Pool Parity

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

Acceptance:

```text
2wikimultihopqa all_exact = 1000/1000
hotpotqa        all_exact = 1000/1000
musique         all_exact = 1000/1000
```

### 3. Run Fresh PCEC Retrieval

Command template:

```bash
/mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  evidence_transition_graphragv4_composition/run_fresh_e2e.py \
  --dataset DATASET \
  --max-queries 1000 \
  --candidate-pool-k 100 \
  --reader-budget-k 5 \
  --prefix-budget-m 4 \
  --qwen-disable-thinking \
  --progress-every 50
```

Datasets:

```text
2wikimultihopqa
hotpotqa
musique
```

### 4. Verify Fresh E2E vs Native-Pool Parity

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

Acceptance:

```text
2wikimultihopqa final_top5 exact = 1000/1000
hotpotqa        final_top5 exact = 1000/1000
musique         final_top5 exact = 1000/1000
```

### 5. Generate Reader-Safe Retrieval Reports

```bash
/mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  scripts/repair_pcec_reader_doc_ids_from_fresh_pool.py \
  --datasets 2wikimultihopqa,hotpotqa,musique
```

Expected changed rows:

```text
2wikimultihopqa changed_rows = 0
hotpotqa        changed_rows = 1000
musique         changed_rows = 1000
```

Reason: 2Wiki happened to use matching ids, while HotpotQA/MuSiQue required
external corpus doc ids for reader context construction.

### 6. Reader-Safe Skip-QA Sanity Check

Before full reader QA, run `--skip-qa` on the reader-safe reports and verify the
retrieval metrics are nonzero and match the PCEC retrieval level.

Expected sanity metrics:

```text
hotpotqa  R@5 = 0.9630    all_gold@5 = 0.9290
musique   R@5 = 0.729083  all_gold@5 = 0.4360
```

If either dataset reports `R@5=0`, the reader report is using the wrong doc-id
space and must not be used.

### 7. Run Reader QA

Reader service map:

```text
2wikimultihopqa -> http://localhost:8041/v1
hotpotqa        -> http://localhost:8042/v1
musique         -> http://localhost:8043/v1
```

Command template:

```bash
env PYTHONPATH=. /mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python \
  evidence_transition_graphragv4_composition/frozen_etv3_variable_flow/run_reader_qa.py \
  --retrieval-reports READER_SAFE_RETRIEVAL_JSON \
  --max-queries 1000 \
  --qa-top-k 5 \
  --save-dir run_logs/pcec_fresh_e2e_reader_qa/reader_runtime_reader_docids/DATASET \
  --llm-name qwen3-8b-train \
  --llm-base-url READER_BASE_URL \
  --max-new-tokens 2048 \
  --embedding-name nvidia/NV-Embed-v2 \
  --embedding-base-url http://localhost:8019/v1/embeddings \
  --qwen-disable-thinking \
  --output-json run_logs/pcec_fresh_e2e_reader_qa/reports_reader_docids/DATASET_pcec_fresh_e2e_reader_qa_full1000.json \
  --output-md run_logs/pcec_fresh_e2e_reader_qa/reports_reader_docids/DATASET_pcec_fresh_e2e_reader_qa_full1000.md
```

### 8. Extract Metrics

Metrics are read from:

```text
datasets[0].methods[...].metrics
```

The primary reader metrics are:

```text
ExactMatch
F1
r5
all_gold_at5
```

## PCEC Results

```text
+-----------------+-------+--------+------------+--------+--------+
| Dataset         | Count | R@5    | All-gold@5 | EM     | F1     |
+-----------------+-------+--------+------------+--------+--------+
| 2WikiMultiHopQA | 1000  | 0.9500 | 0.8550     | 0.6230 | 0.7009 |
| HotpotQA        | 1000  | 0.9630 | 0.9290     | 0.6300 | 0.7480 |
| MuSiQue         | 1000  | 0.7291 | 0.4360     | 0.3430 | 0.4454 |
+-----------------+-------+--------+------------+--------+--------+
```

## Comparison: F1

```text
+----------+----------+---------+--------+--------+-----------+--------------+------------+
| Dataset  | HippoRAG | ProPRAG | ETv3   | PCEC   | PCEC-ETv3 | PCEC-ProPRAG | PCEC-Hippo |
+----------+----------+---------+--------+--------+-----------+--------------+------------+
| 2Wiki    | 0.5850   | 0.6457  | 0.6601 | 0.7009 | +0.0408   | +0.0552      | +0.1159    |
| HotpotQA | 0.7010   | 0.7227  | 0.7330 | 0.7480 | +0.0150   | +0.0253      | +0.0470    |
| MuSiQue  | 0.3947   | 0.4266  | 0.4319 | 0.4454 | +0.0135   | +0.0188      | +0.0507    |
| Avg      | 0.5602   | 0.5983  | 0.6083 | 0.6314 | +0.0231   | +0.0331      | +0.0712    |
+----------+----------+---------+--------+--------+-----------+--------------+------------+
```

## Comparison: EM

```text
+----------+----------+---------+--------+--------+-----------+--------------+------------+
| Dataset  | HippoRAG | ProPRAG | ETv3   | PCEC   | PCEC-ETv3 | PCEC-ProPRAG | PCEC-Hippo |
+----------+----------+---------+--------+--------+-----------+--------------+------------+
| 2Wiki    | 0.5240   | 0.5750  | 0.5860 | 0.6230 | +0.0370   | +0.0480      | +0.0990    |
| HotpotQA | 0.5770   | 0.5950  | 0.6170 | 0.6300 | +0.0130   | +0.0350      | +0.0530    |
| MuSiQue  | 0.3110   | 0.3300  | 0.3320 | 0.3430 | +0.0110   | +0.0130      | +0.0320    |
| Avg      | 0.4707   | 0.5000  | 0.5117 | 0.5320 | +0.0203   | +0.0320      | +0.0613    |
+----------+----------+---------+--------+--------+-----------+--------------+------------+
```

## Comparison: R@5

```text
+----------+----------+---------+--------+--------+-----------+--------------+------------+
| Dataset  | HippoRAG | ProPRAG | ETv3   | PCEC   | PCEC-ETv3 | PCEC-ProPRAG | PCEC-Hippo |
+----------+----------+---------+--------+--------+-----------+--------------+------------+
| 2Wiki    | 0.8313   | 0.9028  | 0.9015 | 0.9500 | +0.0485   | +0.0472      | +0.1187    |
| HotpotQA | 0.9230   | 0.9500  | 0.9505 | 0.9630 | +0.0125   | +0.0130      | +0.0400    |
| MuSiQue  | 0.7011   | 0.7372  | 0.7184 | 0.7291 | +0.0107   | -0.0081      | +0.0280    |
| Avg      | 0.8185   | 0.8633  | 0.8568 | 0.8807 | +0.0239   | +0.0174      | +0.0622    |
+----------+----------+---------+--------+--------+-----------+--------------+------------+
```

## PCEC vs ETv3: All-Gold@5

```text
+----------+--------+--------+---------+
| Dataset  | ETv3   | PCEC   | Delta   |
+----------+--------+--------+---------+
| 2Wiki    | 0.7060 | 0.8550 | +0.1490 |
| HotpotQA | 0.9050 | 0.9290 | +0.0240 |
| MuSiQue  | 0.4200 | 0.4360 | +0.0160 |
| Avg      | 0.6770 | 0.7400 | +0.0630 |
+----------+--------+--------+---------+
```

## Interpretation

1. PCEC improves over frozen ETv3 on all three datasets at reader level.
   Average deltas are `+0.0231 F1`, `+0.0203 EM`, and `+0.0239 R@5`.

2. PCEC also exceeds the same-stack bare ProPRAG and HippoRAG baselines on
   average reader QA.  Average F1 deltas are `+0.0331` over ProPRAG and
   `+0.0712` over HippoRAG.

3. MuSiQue remains the constrained case.  PCEC improves MuSiQue reader F1 over
   ETv3 and ProPRAG, but its R@5 is slightly below ProPRAG by `-0.0081`.  This
   supports the selector-reader mismatch claim: stronger set recall is useful
   but does not fully determine reader QA.

4. The main selector-level gain is on 2Wiki, where all-gold@5 increases from
   `0.7060` to `0.8550`.  This is the cleanest evidence for
   preservation-constrained residual admission.

## Reproducibility Gates

Treat a run as valid only if all gates pass:

```text
1. PCEC unit tests pass.
2. Fresh pool parity is 1000/1000 on all datasets.
3. Fresh E2E vs native-pool final top5 parity is 1000/1000 on all datasets.
4. Reader-safe reports use external corpus doc ids in retrieved_doc_indices_top5.
5. Skip-QA sanity metrics are nonzero and match retrieval-level metrics.
6. Reader QA uses reports_reader_docids outputs, not reports outputs.
```

## Paper-Safe Summary

Use this wording:

```text
Under the same Qwen3-8B reader and NV-Embed retrieval stack, PCEC improves over
the frozen ETv3 expander by +2.31 average F1 across 2WikiMultiHopQA, HotpotQA,
and MuSiQue.  It also outperforms same-stack bare ProPRAG and HippoRAG by +3.31
and +7.12 average F1, respectively.  The largest gain appears on 2Wiki, while
MuSiQue shows smaller but positive reader-level gains despite remaining close to
the selector-level coverage ceiling.
```
