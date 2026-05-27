# ETv3 Variable-Flow Full1000 Results

Date: 2026-05-10

## Scope

This note records the completed full1000 run for
`evidence_transition_graphragv3_variable_flow` under the controlled Qwen3-8B
and NV-Embed-v2 substrate.

ETv3 is evaluated here as a native method-owned retrieval/readout line.  It does
not consume the exported PropRAG pool and should not be directly framed as
losing to PropRAG-pool composer systems such as `Prop+DAEC` or SetR-style
selectors.  Those are stronger two-stage references: strong pool plus
composition selector.

## Configuration

| Item | Setting |
| --- | --- |
| Method | `evidence_transition_graphragv3_variable_flow` |
| Run root | `run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_full_20260510` |
| Rows | full1000 per dataset |
| Datasets | `2wikimultihopqa`, `hotpotqa`, `musique` |
| Reader / extraction LLM | `qwen3-8b-train` |
| Thinking | disabled |
| Embedding | `nvidia/NV-Embed-v2` |
| Embedding endpoint | `http://localhost:8019/v1/embeddings` |
| Candidate pool | method-owned candidate universe, `candidate_pool_k=200` |
| Reader evidence budget | `top_k=5` |
| Fresh-index policy | rebuild per dataset in this output root; no legacy index reuse |

Launcher:

```text
run_logs/launch_etv3_full_qwen8b_nv2_20260510.sh
```

Completion:

```text
2026-05-10T12:49:34+08:00
```

## Final Results

| Dataset | R@5 | all_gold@5 | answer_hit@5 | EM | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 0.9015 | 0.7060 | 0.8330 | 0.5860 | 0.6601 |
| HotpotQA | 0.9505 | 0.9050 | 0.9130 | 0.6170 | 0.7330 |
| MuSiQue | 0.7184 | 0.4200 | 0.6530 | 0.3320 | 0.4319 |

Full QA outputs:

| Dataset | QA output |
| --- | --- |
| 2Wiki | `run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_full_20260510/reports/2wikimultihopqa_evidence_transition_graphragv3_variable_flow_qa.json` |
| HotpotQA | `run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_full_20260510/reports/hotpotqa_evidence_transition_graphragv3_variable_flow_qa.json` |
| MuSiQue | `run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_full_20260510/reports/musique_evidence_transition_graphragv3_variable_flow_qa.json` |

Full retrieval outputs:

| Dataset | Retrieval output |
| --- | --- |
| 2Wiki | `run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_full_20260510/runs/2wikimultihopqa/2wikimultihopqa/reports/2wikimultihopqa_evidence_transition_graphragv3_variable_flow_retrieval.json` |
| HotpotQA | `run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_full_20260510/runs/hotpotqa/hotpotqa/reports/hotpotqa_evidence_transition_graphragv3_variable_flow_retrieval.json` |
| MuSiQue | `run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_full_20260510/runs/musique/musique/reports/musique_evidence_transition_graphragv3_variable_flow_retrieval.json` |

## Comparison Against Existing Full1000 References

The table below uses the existing aligned full1000 references from
`run_logs/aligned_full1000_master_comparison_20260430.md`.

| Dataset | Method | Regime | R@5 | EM | F1 | Reading |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 2Wiki | PropRAG top5 | Prop pool order | 0.9028 | 0.5790 | 0.6503 | Bare PropRAG reference |
| 2Wiki | ETv3 variable-flow | native ET readout | 0.9015 | 0.5860 | 0.6601 | Bare ETv3 is PropRAG-top5 level |
| 2Wiki | Prop+DAEC-L1 | Prop pool + composer | 0.9215 | 0.6130 | 0.6828 | Strong two-stage reference |
| HotpotQA | PropRAG top5 | Prop pool order | 0.9500 | 0.5940 | 0.7212 | Bare PropRAG reference |
| HotpotQA | ETv3 variable-flow | native ET readout | 0.9505 | 0.6170 | 0.7330 | Bare ETv3 beats bare PropRAG and approaches Prop+DAEC |
| HotpotQA | Prop+DAEC-L1 | Prop pool + composer | 0.9460 | 0.6180 | 0.7370 | Strong two-stage reference |
| MuSiQue | PropRAG top5 | Prop pool order | 0.7131 | 0.3300 | 0.4266 | Bare PropRAG reference |
| MuSiQue | ETv3 variable-flow | native ET readout | 0.7184 | 0.3320 | 0.4319 | Bare ETv3 is slightly above bare PropRAG |
| MuSiQue | Prop+DAEC | Prop pool + composer | 0.7433 | 0.3390 | 0.4404 | Strong two-stage reference |

## Interpretation

ETv3 is positive as a native method-owned retriever/readout:

- On 2Wiki, bare ETv3 reaches bare PropRAG top5 level: `F1 0.6601` versus
  PropRAG top5 `0.6503`.
- On HotpotQA, bare ETv3 is strong: `F1 0.7330`, above bare PropRAG top5
  `0.7212` and close to Prop+DAEC-L1 `0.7370`.
- On MuSiQue, bare ETv3 is slightly above bare PropRAG top5: `F1 0.4319`
  versus `0.4266`, but the long-chain bottleneck remains.

The result should not be summarized as "ETv3 loses to DAEC/SetR" without the
pool distinction.  DAEC and SetR-style numbers in the existing comparison are
PropRAG-pool composition systems.  The fair statement is:

```text
Bare ETv3 is competitive with bare PropRAG top5 across all three datasets.
Strong PropRAG-pool composers still provide additional gains, especially on
2Wiki and MuSiQue.  The next fair test is ETv3 pool + DBEC/DAEC composition.
```

## Dataset-Specific Notes

2Wiki:

- `R@5 = 0.9015`, essentially tied with PropRAG top5 `0.9028`.
- `EM/F1 = 0.5860 / 0.6601`, slightly above PropRAG top5
  `0.5790 / 0.6503`.
- Remaining gap to Prop+DAEC-L1 should be treated as a composition-layer
  opportunity, not evidence that the native ETv3 substrate is weak.

HotpotQA:

- Strongest dataset for ETv3.
- `EM/F1 = 0.6170 / 0.7330`, near Prop+DAEC-L1
  `0.6180 / 0.7370`.
- This mainly validates that ETv3 does not hurt shallow multi-hop cases.
  HotpotQA is not the primary evidence for resolving long dependency chains.

MuSiQue:

- `R@5 = 0.7184` and `F1 = 0.4319`, slightly above bare PropRAG top5.
- `all_gold@5 = 0.4200` remains low, confirming the long-chain fixed-budget
  bottleneck.
- This is consistent with the earlier limit100 diagnosis: MuSiQue failures are
  not fixed by graph readout alone and need either better composition over the
  ETv3 pool or a carefully selected auxiliary reader context.

## Native Failure Audit

Follow-up audit:

```text
reports/etv3_native_full1000_audit_20260510/etv3_native_full1000_failure_audit.md
```

This audit is diagnostic-only: it reads the completed ETv3 retrieval and QA
JSONs, does not call an LLM, and does not modify ETv3/ETv4 method code.

Main audit findings:

- Across all datasets, ETv3 has substantial candidate-universe headroom:
  `all_gold@200` is `0.9750` on 2Wiki, `0.9940` on HotpotQA, and `0.8610`
  on MuSiQue.
- `R@k` and `all_gold@k` diverge strongly on long-chain settings.  On MuSiQue,
  `R@5 = 0.7184` but `all_gold@5 = 0.4200`; on MuSiQue 4-doc, `R@5 = 0.4623`
  but `all_gold@5 = 0.0181`.  This is the set-level composition gap: partial
  support recall is not enough for multi-hop QA.
- The MuSiQue long-chain bottleneck is sharply top5-composition driven:
  4-doc queries have `all_gold@5 = 0.0181`, but `all_gold@200 = 0.6687`.
- MuSiQue 4-doc top5 reader failure is not the dominant failure mode: only
  3 / 166 4-doc queries have all gold evidence in top5; 108 / 166 have all gold
  evidence somewhere in ETv3 candidate200 but not in top5.
- The 108 MuSiQue 4-doc candidate-complete/top5-incomplete cases are not only a
  top10/top20 budget problem: by the worst-ranked missing gold document, only
  28 / 108 are within rank20, while 49 / 108 require rank51-200.  The next
  method therefore needs a composition objective over deeper candidate ranks,
  not just a small reader-budget increase.
- This supports an ETv3-pool composition audit next.  It does not by itself
  prove that ETv4 should be state-binding; that requires a residual audit after
  ETv3-pool + stable DBEC/DAEC full1000.

## Follow-Up

The clean next comparison is:

```text
ETv3 pool + baseline-stable DBEC/DAEC
vs
PropRAG pool + DBEC/DAEC / SetR-style references
```

This separates the two questions:

1. Is ETv3 a competitive native retrieval/readout substrate?
2. Does the ETv3 candidate universe support the same composition gains as the
   PropRAG candidate universe?

Current evidence answers question 1 positively.  Question 2 is still open.
